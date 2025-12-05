import argparse
import os
from os.path import join, abspath

from tqdm import tqdm
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6"  # set before torch import
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import random
import transformers
from torchvision import transforms
from img_proc import expand_4d
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from openvla_dataloader import get_dataloader
import datetime

# spaa import
from img_proc import expand_4d, center_crop as cc

# from train_network import train_eval_pcnet, get_model_train_cfg
from perc_al.differential_color_functions import rgb2lab_diff, ciede2000_diff


def myAttack(
    dataset,
    pcnet,
    vla,
    train_dataloader,
    val_dataloader,
    device,
    name,
    root,
    inner_loop=2000,
    outer_loop=50,
    targeted=False,
    p_thr=0.25,
    d_thr=11.0,
    eval_freq=500,
):

    # learning rates
    adv_lr = 1e-3  # 5e-4
    col_lr = 1e-4  # stealth usually benefit from gentle steps

    adv_w = 1  # weight for adversarial loss
    stealth_loss = ["prjl2", "caml2"]  # using both caml2 and camdE, prjl2 ignored
    prjl2_w = (
        0.1 if "prjl2" in stealth_loss else 0
    )  # weight for pixel difference between prj_adv and im_gray
    caml2_w = (
        1 if "caml2" in stealth_loss else 0
    )  # weight for pixel difference between cam_infer and cam_scene
    camdE_w = (
        1 if "camdE" in stealth_loss else 0
    )  # weight for color fidelity(visual realism) between cam_infer and cam_scene

    iters = outer_loop
    inner_loop = inner_loop
    B = 1
    eval_batch_size = 10
    maskidx = [0, 1, 2, 3, 4, 5]  # attack all dof except gripper
    # maskidx = [0, 6]  # attack only base rotation and gripper open/close

    # => creates a residual image of learnable perturbatiosn on the input images
    prj_brightness = 0.2
    prj_im_sz = (224, 224)
    im_gray, prj_adv = create_prj_adv(B, prj_brightness, prj_im_sz, device)
    adv_optimizer = transformers.AdamW(
        [prj_adv], lr=adv_lr, weight_decay=0.0
    )  # disable weight_decay
    col_optimizer = transformers.AdamW([prj_adv], lr=col_lr, weight_decay=0.0)
    print("prj_adv initialised:", prj_adv.shape)
    save_image(prj_adv, f"{root}/initial.png")

    # scheduler = transformers.get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup,
    #     num_training_steps=int(iters / accumulate_steps),
    #     num_cycles=0.5,
    #     last_epoch=-1,
    # )

    val_CE_loss = []
    val_MSE_Distance = []
    val_UAD = []
    val_col_loss = []
    train_CE_loss = []
    train_MSE_distance_loss = []
    train_UAD = []

    prj_adv_best = prj_adv.clone()
    stealth_adv_best = prj_adv.clone()
    best_adv = prj_adv.clone()
    col_loss_best = float("inf")
    # best_info = None  # (outer_iter, inner_iter, UAD, MSE_distance, col_loss)
    MSE_Distance_best = 10000
    stealth_loss_best = 10000

    mean = [
        torch.tensor([0.484375, 0.455078125, 0.40625]),  # imageNet normalization mean
        torch.tensor([0.5, 0.5, 0.5]),  # CLIP normalization mean
    ]
    std = [
        torch.tensor([0.228515625, 0.2236328125, 0.224609375]),
        torch.tensor([0.5, 0.5, 0.5]),
    ]
    wandb.init(
        project="SAA_Stealthy_Adversarial_Attack_standardized",
        name=name,
        dir=root,
        tags=["uada", "perturbation"],
    )
    wandb.config.update(
        {
            "adv_lr": adv_lr,
            "stealth_lr": col_lr,
            "attack_target": maskidx,
            "p_thr": p_thr,
            "d_thr": d_thr,
            "inner_loop": inner_loop,
            "outer_loop": outer_loop,
            "libero_model": dataset,
            "eval_freq": eval_freq,
            "eval_batch_size": eval_batch_size,
        }
    )

    for i in tqdm(range(0, iters)):
        train_relative_distance = {f"{idx}": [] for idx in maskidx}
        data = next(iter(train_dataloader))
        labels = data["labels"].to(device)
        attention_mask = data["attention_mask"].to(device)
        input_ids = data["input_ids"].to(device)
        cam_scene = data["pixel_values"]
        labels = mask_labels(action_tokenizer, labels, maskidx)
        success_count = 0
        for j in range(inner_loop):
            iteration = inner_loop * i + j
            # cam_infer = apply_projection(pcnet, prj_adv, cam_scene, mean, std)
            cam_scene_t, cam_infer_t, vla_scene = apply_perturbation(
                prj_adv, cam_scene, mean, std, root
            )

            # ----------------y_pred ← F(T(x+δ)) -------------
            output: CausalLMOutputWithPast = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=vla_scene.to(torch.bfloat16).to(device),
                labels=labels,
                use_cache=False,
            )

            # --------------compute loss & back propagation-----------------
            # adv loss
            celoss = output.loss
            # MSE_distance  = distance between output and max discrepency
            # UAD = average deviation from ground truth
            MSE_Distance, UAD = weighted_loss(output.logits, labels, maskidx)
            # MSE_Distance = MSE_Distance + 1 / celoss
            # 1/celoss might cause explosion when celoss is 0
            eps = 1e-6
            alpha = 1.0  # tune
            MSE_Distance = MSE_Distance + alpha / (celoss.detach() + eps)

            # stealth loss
            prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)
            col_loss = prjl2_w * prjl2
            caml2 = torch.norm(cam_scene_t - cam_infer_t, dim=1).mean(1).mean(1)
            col_loss += caml2_w * caml2
            # color fidelity to human eye (CIE deltaE 2000 loss)
            # camdE = (
            #     ciede2000_diff(
            #         rgb2lab_diff(cam_infer_t, device),
            #         rgb2lab_diff(cam_scene_t, device),
            #         device,
            #     )
            #     .mean(1)
            #     .mean(1)
            # )
            # col_loss += camdE_w * camdE
            col_loss = col_loss.mean()

            stealth_loss_ok = (col_loss * 255 < d_thr).bool()
            adv_loss_ok = (UAD > p_thr).bool()
            is_better_loss = adv_loss_ok & (col_loss.item() < col_loss_best)

            # ---------------update------------------
            # if j % log_freq == 0:
            # print("mse_distance:", MSE_Distance.item(), "uad:", UAD.item())
            # print("color loss and caml2:", col_loss.item(), caml2.item())

            if stealth_loss_ok or not adv_loss_ok:  # focus on attack
                if adv_loss_ok:
                    success_count += 1
                    print("Success achieved at iteration:", iteration)
                loss = MSE_Distance
                # train_CE_loss.append(celoss.item())
                # train_MSE_distance_loss.append(MSE_Distance.item())
                # train_UAD.append(UAD.item())
                optimizer = adv_optimizer
            else:
                loss = col_loss
                optimizer = col_optimizer

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            log_patch_grad = prj_adv.grad.detach().mean().item()
            optimizer.step()
            with torch.no_grad():
                prj_adv.clamp_(0.0, 1.0)  # maintain prj_adv as leaf

            # vla.zero_grad()
            # torch.cuda.empty_cache()

            action_logits = output.logits[
                :, vla.vision_backbone.featurizer.patch_embed.num_patches : -1
            ]
            action_preds = action_logits.argmax(dim=2)
            action_gt = labels[:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_preds[mask].cpu().numpy()
                )
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(
                    action_gt[mask].cpu().numpy()
                )
            )
            # NAD logs the dimension under attack(Specified by maskidx)
            train_relative_distance = calculate_relative_distance(
                continuous_actions_pred,
                continuous_actions_gt,
                maskidx,
                train_relative_distance,
            )
            train_logdata = {
                # "train/lr/adv": adv_optimizer.param_groups[0]["lr"],
                # "train/lr/col": col_optimizer.param_groups[0]["lr"],
                "train/adv_loss/UAD": UAD.item(),
                "train/adv_loss/MSE_Distance": MSE_Distance.item(),
                # "train/stealth_loss/caml2": caml2.item(),
                "train/stealth_loss/col_loss": col_loss.item(),
                "train/patch_grad": log_patch_grad,
                "train/step": iteration,
            }
            for key, value in train_relative_distance.items():
                property_name = f"train/NAD_{key}"
                train_logdata[property_name] = sum(value) / len(value)
            wandb.log(train_logdata)

            if (iteration + 1) % eval_freq == 0:
                # val_num_sample = 0
                avg_MSE_Distance = 0
                avg_UAD = 0
                avg_CE_loss = 0
                avg_col_loss = 0
                relative_distance = {f"{idx}": [] for idx in maskidx}  # NAD
                print("evaluating...")
                # torch.cuda.empty_cache()
                val_iterator = iter(val_dataloader)
                with torch.inference_mode():
                    for k in tqdm(range(eval_batch_size)):
                        try:
                            val_data = next(val_iterator)
                        except StopIteration:
                            val_iterator = iter(val_dataloader)
                            val_data = next(val_iterator)
                        val_labels = val_data["labels"].to(device)
                        val_attention_mask = val_data["attention_mask"].to(device)
                        val_input_ids = val_data["input_ids"].to(device)
                        val_cam_scene = val_data["pixel_values"]
                        # val_num_sample += val_labels.shape[0]
                        val_labels = mask_labels(action_tokenizer, val_labels, maskidx)
                        val_cam_scene_t, val_cam_infer_t, val_vla_scene = (
                            apply_perturbation(prj_adv, val_cam_scene, mean, std, root)
                        )
                        val_output: CausalLMOutputWithPast = vla(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask,
                            pixel_values=val_vla_scene.to(torch.bfloat16).to(device),
                            labels=val_labels,
                            use_cache=False,  # save gpu mem
                        )
                        val_MSE_Distance, val_UAD = weighted_loss(
                            val_output.logits, val_labels, maskidx
                        )
                        val_caml2 = (
                            torch.norm(val_cam_scene_t - val_cam_infer_t, dim=1)
                            .mean(1)
                            .mean(1)
                        )
                        val_caml2 = val_caml2.mean()

                        # TODO: add color fidelity validation

                        val_action_logits = val_output.logits[
                            :,
                            vla.vision_backbone.featurizer.patch_embed.num_patches : -1,
                        ]
                        val_action_preds = val_action_logits.argmax(dim=2)
                        val_action_gt = val_labels[:, 1:].to(val_action_preds.device)
                        val_mask = (
                            val_action_gt > action_tokenizer.action_token_begin_idx
                        )
                        continuous_actions_pred = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(
                                val_action_preds[val_mask].cpu().numpy()
                            )
                        )
                        continuous_actions_gt = torch.tensor(
                            action_tokenizer.decode_token_ids_to_actions(
                                val_action_gt[val_mask].cpu().numpy()
                            )
                        )
                        relative_distance = calculate_relative_distance(
                            continuous_actions_pred,
                            continuous_actions_gt,
                            maskidx,
                            relative_distance,
                        )
                        avg_MSE_Distance += val_MSE_Distance.item()
                        avg_UAD += val_UAD.item()
                        avg_CE_loss += val_output.loss.item()
                        avg_col_loss += val_caml2.item()
                    torch.cuda.empty_cache()
                    avg_MSE_Distance /= eval_batch_size
                    avg_UAD /= eval_batch_size
                    avg_CE_loss /= eval_batch_size
                    avg_col_loss /= eval_batch_size
                    log_data = {
                        "val/MSE_distance": avg_MSE_Distance,
                        "val/UAD": avg_UAD,
                        "val/CE_loss": avg_CE_loss,
                        "val/col_loss": avg_col_loss,
                        "val/step": k,
                    }
                    for key, value in relative_distance.items():
                        property_name = f"val/NAD_{key}"
                        log_data[property_name] = sum(value) / len(value)
                    wandb.log(log_data)
                    stealthy = avg_col_loss * 255 < d_thr
                    successful = avg_UAD > 0.1
                    if stealthy and successful:
                        print(
                            f"Successful and stealthy attack at outer iter {i} inner iter {j} with UAD {avg_UAD}, col loss {avg_col_loss}"
                        )
                        if (
                            avg_MSE_Distance < MSE_Distance_best
                            and avg_col_loss < stealth_loss_best
                        ):
                            print(
                                f"New best [attack+stealthy] patch found at outer iter {i} inner iter {j} with UAD {avg_UAD}, col loss {avg_col_loss}"
                            )
                            MSE_Distance_best = avg_MSE_Distance
                            stealth_loss_best = avg_col_loss
                            best_adv = prj_adv.clone()
                        if avg_MSE_Distance < MSE_Distance_best:
                            MSE_Distance_best = avg_MSE_Distance
                            print(
                                f"Also a new best [attack] patch found at outer iter {i} inner iter {j} with UAD {avg_UAD}, col loss {col_loss}"
                            )
                            prj_adv_best = prj_adv.clone()
                        if avg_col_loss < stealth_loss_best:
                            stealth_loss_best = avg_col_loss
                            print(
                                f"Also a new best [stealthy] patch found at outer iter {i} inner iter {j} with UAD {avg_UAD}, col loss {avg_col_loss}"
                            )
                            stealth_adv_best = prj_adv.clone()
                        # val_CE_loss.append(float(avg_CE_loss))
                        # val_MSE_Distance.append(float(avg_MSE_Distance))
                        # val_UAD.append(float(avg_UAD))
                        # val_col_loss.append(float(avg_col_loss))
                        # TODO: log info
                        del (
                            val_labels,
                            val_attention_mask,
                            val_input_ids,
                            val_cam_scene,
                            val_cam_scene_t,
                            val_cam_infer_t,
                            val_vla_scene,
                            val_output,
                            val_action_logits,
                            val_action_preds,
                            val_action_gt,
                            val_mask,
                        )
                torch.cuda.empty_cache()

        # ------------ Outer Loop -------------------
        # save this iteration's prj_adv
        save_tensor_as_image(best_adv, f"{root}/best.png")
        save_tensor_as_image(prj_adv_best, f"{root}/adv_best.png")
        save_tensor_as_image(stealth_adv_best, f"{root}/stealth_best.png")

        if i % 100 == 0:
            tmp = prj_adv.clone()
            save_tensor_as_image(
                tmp, f"{root}/prj_adv_iter_{i}_success_{success_count}.png"
            )
        print("Success count in this outer loop:", success_count)
        torch.cuda.empty_cache()
    wandb.finish()


def mask_labels(action_tokenizer, labels, maskidx, dof=7):
    mask = labels > action_tokenizer.action_token_begin_idx
    masked_labels = labels[mask]
    masked_labels = masked_labels.view(masked_labels.shape[0] // dof, dof)
    template_labels = torch.ones_like(masked_labels, device=masked_labels.device) * -100
    for idx in maskidx:
        template_labels[:, idx] = masked_labels[:, idx]
    labels[labels > 2] = template_labels.view(-1)
    return labels


def save_tensor_as_image(tensor, path):
    Image = T.ToPILImage()(tensor.squeeze(0))
    Image.save(os.path.join(run_root, path))
    return Image


def calculate_relative_distance(pred, gt, maskidx, relative_distance):
    pred = pred.clone().view(pred.shape[0] // len(maskidx), len(maskidx))
    gt = gt.clone().view(gt.shape[0] // len(maskidx), len(maskidx))
    for idx1 in range(pred.shape[0]):
        for idx2 in range(pred.shape[1]):
            anchor = gt[idx1, idx2]
            input_point = pred[idx1, idx2]
            upper_bound = 1
            lower_bound = -1
            distance_to_upper = upper_bound - anchor
            distance_to_lower = anchor - lower_bound
            max_boundary_distance = max(distance_to_upper, distance_to_lower)
            distance_to_anchor = abs(input_point - anchor)
            temp_relative_distance = distance_to_anchor / max_boundary_distance
            relative_distance[f"{str(maskidx[idx2])}"].append(
                temp_relative_distance.item()
            )
    return relative_distance


def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def create_prj_adv(B=1, prj_brightness=0.5, prj_im_sz=(200, 200), device="cuda"):
    if prj_brightness == 0:
        im_gray = 0.2 * torch.randn(B, 3, *prj_im_sz).to(device)
    else:
        im_gray = prj_brightness * torch.ones(B, 3, *prj_im_sz).to(device)
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True
    prj_adv.retain_grad()
    # prj_adv = torch.nn.Parameter(prj_adv)
    # optimizer = torch.optim.Adam([prj_adv], lr=adv_lr) # for application with nn.Module
    return im_gray, prj_adv


""" overlay an stealth patch(prj_adv) to the entire image """


def apply_perturbation(prj_adv, images, mean, std, root):
    """
    Overlay the perturbation image
    return both images tensor, perturbed images tensor, VLA input tensor
    6 channel images for VLA input (224, 224)
    """
    perturbed_img_vla = []
    perturbed_img = torch.empty((len(images), 3, 224, 224)).to(device)
    images_tensor = torch.empty((len(images), 3, 224, 224)).to(device)
    for i, im in enumerate(images):  # (320, 240)
        im = transforms.ToTensor()(im)  # normalised?
        im = transforms.Compose(
            [
                transforms.CenterCrop((240, 240)),
                transforms.Resize((224, 224)),
            ]
        )(im).to(device)
        images_tensor[i] = im
        im = (im + prj_adv).clamp(0.0, 1.0)  # TODO: add eps = cap for change
        # from torchvision.utils import save_image
        perturbed_img[i] = im
        # save_image(im, f"{root}/perturbed_image.png")
        # print("perturbed image saved to", f"{root}/perturbed_image.png")

        im0 = normalize(im, mean[0].to(device), std[0].to(device))  # for simulation
        im1 = normalize(im, mean[1].to(device), std[1].to(device))  # for real world
        perturbed_img_vla.append(torch.cat([im0, im1], dim=1))
    return images_tensor, perturbed_img, torch.cat(perturbed_img_vla, dim=0)


""" using pcnet to project a stealth patch(prj_adv) to the center of the image """


def apply_projection(pcnet, prj_adv, images, mean, std):
    projected_images = []
    cam_im_sz = (240, 320)  # (h,W)
    vla_sz = (224, 224)
    bs = 2
    for im in images:
        im = transforms.ToTensor()(im)  # Converts to float32 in [0,1] and (C,H,W)
        print("original im size:", im.shape)  # [3, 240, 320]
        im = im.expand(bs, -1, -1, -1).to(device)
        prj_adv = torch.clamp(expand_4d(prj_adv), 0, 1)  # size ([1, 3, 256, 256])
        im = pcnet(prj_adv, im).to(device)  # size [1, 3, 240, 320]

        save_image(im[0], "output.png")

        im = transforms.Compose(
            [
                transforms.Resize(256),  # resize short side to slightly longer than 224
                transforms.CenterCrop(vla_sz),
            ]
        )(im)
        im0 = normalize(im, mean[0].to(device), std[0].to(device))  # for simulation
        im1 = normalize(im, mean[1].to(device), std[1].to(device))  # for real world
        # im0 shape = im1 shape = [1, 3, 224, 224]
        # concatenate im0 and im1 to get a 6 channel image for openvla
        projected_images.append(torch.cat([im0, im1], dim=1))
    return torch.cat(projected_images, dim=0)


import matplotlib.pyplot as plt


def show_vla_tensor(x):
    x = x[0].cpu()  # (6,H,W)
    img1, img2 = x[:3].permute(1, 2, 0), x[3:6].permute(1, 2, 0)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[0].axis("off")
    axs[1].imshow(img2)
    axs[1].axis("off")
    plt.show()


def weighted_loss(logits, labels, maskid):
    temp_label = labels[:, 1:].to(labels.device)  # (bs,seq_len) remove bos token
    action_mask = temp_label > 2
    temp_logits = logits[
        :, :, 31744:32000
    ]  # (bs,seq_len,256) only consider the 256 classes for the target class
    action_logits = temp_logits[
        :, -temp_label.shape[-1] - 1 : -1, :
    ]  # shift logits see modeling_llama.py line 1233
    action_logits = action_logits[action_mask]
    reweigh = torch.arange(1, 257).to(logits.device) / 256  # [1,,...256]
    temp_prob = F.softmax(action_logits, dim=-1)  # [bs,action_length,256]
    reweighted_prob = (temp_prob * reweigh).sum(dim=-1)  # [bs, action_length]
    hard_max_labels = temp_label[action_mask]
    hard_max_labels[hard_max_labels > 31872] = 31999
    hard_max_labels[hard_max_labels <= 31872] = 31744
    hard_max_labels[hard_max_labels == 31999] = 1 / 256
    hard_max_labels[hard_max_labels == 31744] = 1
    UAD = cal_UAD(action_logits.argmax(dim=-1) + 31744, temp_label[action_mask])
    distance_loss = F.mse_loss(
        5 * reweighted_prob.contiguous(), 5 * hard_max_labels.float().contiguous()
    )

    # targeted max distance CE loss
    # ce_action_logits = logits[:,-temp_label.shape[-1]-1:-1, :]
    # ce_action_logits = ce_action_logits[action_mask]
    # ce_hard_max_labels = temp_label[action_mask]
    # ce_hard_max_labels[ce_hard_max_labels > 31872]=31744
    # ce_hard_max_labels[ce_hard_max_labels <= 31872]=31999
    # ce_loss = F.cross_entropy(ce_action_logits, ce_hard_max_labels)
    # distance_loss = distance_loss + ce_loss
    return distance_loss, UAD


def filter_train(data):
    pixel_values = data["pixel_values"]
    labels = data["labels"].to(device)
    attention_mask = data["attention_mask"].to(device)
    input_ids = data["input_ids"].to(device)

    mask = labels > action_tokenizer.action_token_begin_idx
    masked_labels = labels[mask]
    masked_labels = masked_labels.view(masked_labels.shape[0] // 7, 7)
    one_index = []
    for idx in range(masked_labels.shape[0]):
        if masked_labels[idx, 6] == 31744:
            one_index.append(idx)
    if 1 < len(one_index) < 8:
        labels = labels[one_index, :]
        attention_mask = attention_mask[one_index, :]
        input_ids = input_ids[one_index, :]
        pixel_values = [pixel_values[i] for i in one_index]
    elif len(one_index) > 8:
        chosen = random.sample(one_index, k=8)
        labels = labels[chosen, :]
        attention_mask = attention_mask[chosen, :]
        input_ids = input_ids[chosen, :]
        pixel_values = [pixel_values[i] for i in chosen]
    elif one_index is None:
        chosen = random.sample(range(labels.shape[0]), k=8)
        labels = labels[chosen, :]
        attention_mask = attention_mask[chosen, :]
        input_ids = input_ids[chosen, :]
        pixel_values = [pixel_values[i] for i in chosen]
    return labels, attention_mask, input_ids, pixel_values


def cal_UAD(pred, gt):
    continuous_actions_gt = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(gt.clone().detach().cpu().numpy())
    )
    continuous_actions_pred = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(
            pred.clone().detach().cpu().numpy()
        )
    )
    max_distance = torch.where(
        continuous_actions_gt > 0,
        torch.abs(continuous_actions_gt - (-1)),
        torch.abs(continuous_actions_gt - 1),
    )
    distance = torch.abs(continuous_actions_pred - continuous_actions_gt)
    UAD = (distance / max_distance).mean()
    return UAD


# function that parse args
def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial attack")
    parser.add_argument(
        "--inner_loop",
        "-i",
        type=int,
        default=2000,
        help="number of inner loop iterations (default: 2000)",
    )
    parser.add_argument(
        "--outer_loop",
        "-o",
        type=int,
        default=50,
        help="number of outer loop iterations (default: 50)",
    )
    parser.add_argument(
        "--vla_model",
        "-v",
        type=str,
        default="libero_spatial",
    )
    parser.add_argument(
        "--p_thr",
        "-p",
        type=float,
        default=0.25,
        help="adversarial threshold (default: 0.25)",
    )
    parser.add_argument(
        "--d_thr",
        "-d",
        type=float,
        default=11.0,
        help="distance threshold (default: 11.0)",
    )
    return parser.parse_args()


# main function
if __name__ == "__main__":
    args = parse_args()
    inner_loop = args.inner_loop
    outer_loop = args.outer_loop
    dataset = args.vla_model
    p_thr = args.p_thr
    d_thr = args.d_thr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create openvla model
    print(
        f"Creating VLA model for {dataset}............................................."
    )
    if dataset == "libero_spatial":
        vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
    elif dataset == "libero_10":
        vla_path = "openvla/openvla-7b-finetuned-libero-10"
    elif dataset == "libero_object":
        vla_path = "openvla/openvla-7b-finetuned-libero-object"
    elif dataset == "libero_goal":
        vla_path = "openvla/openvla-7b-finetuned-libero-goal"
    else:
        raise ValueError("Dataset not supported")
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)  # added
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    for param in vla.parameters():
        param.requires_grad = False
    if hasattr(vla.config, "use_cache"):
        vla.config.use_cache = False
    print(
        f"{vla_path} model created and loaded............................................."
    )

    # create dataloader
    print(
        f"Creating dataloader for {dataset}............................................."
    )
    bs = 1
    server = "/data2/lsc/roboticAttack"
    train_dataloader, val_dataloader = get_dataloader(
        batch_size=bs, dataset=dataset, server=server, vla_path=vla_path
    )
    print(
        f"Dataloader created from {server}............................................."
    )

    root = "/data2/lsc/uada/pcnet"
    # print("Creating PCNet model.............................................")
    # data_root = abspath(join(root, "data"))
    # # model_name = ['PCNet_no_mask_no_rough_d']
    # model_name = ["PCNet"]
    # setup_list = ["coffee_mug"]
    # load_pretrained = True
    # pcnet_cfg = get_model_train_cfg(
    #     model_name, data_root, setup_list, load_pretrained=load_pretrained, plot_on=True
    # )
    # if load_pretrained:
    #     print("Pretrained config loaded: ", pcnet_cfg)
    # pcnet, model_ret, model_cfg = train_eval_pcnet(pcnet_cfg)
    # print("PCNet model created .............................................")

    # run attack

    print(
        f"Starting Attack with inner loop {inner_loop} and outer loop {outer_loop} and p_thr {p_thr} and d_thr {d_thr}..............."
    )
    results_root = abspath(
        join(
            root,
            "attack_run_standardized",
            f"attack_{dataset}_inner{inner_loop}_outer{outer_loop}_p{p_thr}_d{d_thr}",
        )
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = "attack_" + now
    run_root = abspath(join(results_root, run_name))
    os.makedirs(run_root, exist_ok=True)
    myAttack(
        dataset,
        None,
        vla,
        train_dataloader,
        val_dataloader,
        device,
        run_name,
        run_root,
        inner_loop=inner_loop,
        outer_loop=outer_loop,
        targeted=False,
        p_thr=p_thr,
        d_thr=d_thr,
    )
    print(
        f"Attack finished and log result saved at {run_root}........................."
    )

    # print("Saving the patch.............................................")
    # adv_img_path = "prj_adv.png"
    # T.ToPILImage()(prj_adv.squeeze(0)).save(os.path.join(run_root, adv_img_path))
    # print("Patch saved as", adv_img_path)
    print("Done.............................................")
