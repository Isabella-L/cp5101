import os
from os.path import join, abspath

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,6"  # set before torch import
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
from train_network import train_eval_pcnet, get_model_train_cfg
from perc_al.differential_color_functions import rgb2lab_diff, ciede2000_diff


def myAttack(pcnet, vla, train_dataloader, device, root, targeted=False):

    # learning rates
    adv_lr = 1 / 255
    col_lr = 1

    adv_w = 1  # weight for adversarial loss
    stealth_loss = ["caml2_camdE"]  # using both caml2 and camdE, prjl2 ignored
    prjl2_w = (
        0.1 if "prjl2" in stealth_loss else 0
    )  # weight for pixel difference between prj_adv and im_gray
    caml2_w = (
        1 if "caml2" in stealth_loss else 0
    )  # weight for pixel difference between cam_infer and cam_scene
    camdE_w = (
        1 if "camdE" in stealth_loss else 0
    )  # weight for color fidelity(visual realism) between cam_infer and cam_scene

    p_thresh = 0.9  # adversarial confidence threshold
    d_thresh = 5
    # d_threshes = [5, 7, 9, 11]

    # TODO: change back to more
    iters = 1
    inner_loop = 50
    B = 1

    # => creates a residual image of learnable perturbatiosn on the input images
    prj_brightness = 0.1
    prj_im_sz = (224, 224)
    im_gray, prj_adv, optimizer = create_prj_adv(
        adv_lr, B, prj_brightness, prj_im_sz, device
    )
    print("prj_adv initialised:", prj_adv.shape)
    save_image(prj_adv, f"{root}/initial.png")

    warmup = 20
    accumulate_steps = 1
    maskidx = "0"
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
    train_CE_loss = []
    train_MSE_distance_loss = []
    train_UAD = []
    mean = [
        torch.tensor([0.484375, 0.455078125, 0.40625]),  # imageNet normalization mean
        torch.tensor([0.5, 0.5, 0.5]),  # CLIP normalization mean
    ]
    std = [
        torch.tensor([0.228515625, 0.2236328125, 0.224609375]),
        torch.tensor([0.5, 0.5, 0.5]),
    ]

    for i in tqdm(range(0, iters)):
        data = next(iter(train_dataloader))

        labels = data["labels"].to(device)
        attention_mask = data["attention_mask"].to(device)
        input_ids = data["input_ids"].to(device)
        cam_scene = data["pixel_values"]

        for j in range(inner_loop):
            # cam_infer = apply_projection(pcnet, prj_adv, cam_scene, mean, std)
            cam_infer = apply_perturbation(prj_adv, cam_scene, mean, std, root)
            print("cam_infer size:", cam_infer.shape)

            # ----------------y_pred ← F(T(x+δ)) -------------
            output: CausalLMOutputWithPast = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=cam_infer.to(torch.bfloat16).to(device),
                labels=labels,
            )
            # print("Getting VLA Output.............................................")
            # # print(output)
            # if hasattr(output, "loss") and output.loss is not None:
            #     print(f"🧮 loss: {output.loss.item():.6f}")
            # if hasattr(output, "logits"):
            #     print(f"📊 logits: shape = {tuple(output.logits.shape)}")

            # --------------compute loss & back propagation-----------------
            # adv loss
            celoss = output.loss
            MSE_Distance, UAD = weighted_loss(output.logits, labels, maskidx)
            MSE_Distance = MSE_Distance + 1 / celoss
            MSE_Distance.backward()

            train_CE_loss.append(celoss.item())
            train_MSE_distance_loss.append(MSE_Distance.item())
            train_UAD.append(UAD.item())
            log_patch_grad = prj_adv.grad.detach().mean().item()

            # stealth loss
            prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)
            col_loss_batch = prjl2_w * prjl2
            # stealthiness loss: cam-captured image should look like cam_scene (L2 loss)
            caml2 = (
                torch.norm(cam_scene - cam_infer, dim=1).mean(1).mean(1)
            )  # mean L2 norm, consistent with Zhao_CVPR_20
            col_loss_batch += caml2_w * caml2

            # stealthiness loss: cam-captured image should look like cam_scene (CIE deltaE 2000 loss)
            camdE = (
                ciede2000_diff(
                    rgb2lab_diff(cam_infer, device),
                    rgb2lab_diff(cam_scene, device),
                    device,
                )
                .mean(1)
                .mean(1)
            )
            col_loss_batch += camdE_w * camdE

            # average stealthiness (color) losses
            col_loss = col_loss_batch.mean()

            # ---------------update------------------
            optimizer.step()
            prj_adv.data = prj_adv.data.clamp(0, 1)
            optimizer.zero_grad()
            vla.zero_grad()
        torch.cuda.empty_cache()

    torch.save(prj_adv.detach().cpu(), "prj_adv.pt")
    return prj_adv.detach().cpu()


def normalize(images, mean, std):
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def create_prj_adv(
    adv_lr=1e-2, B=1, prj_brightness=0.5, prj_im_sz=(200, 200), device="cuda"
):
    if prj_brightness == 0:
        im_gray = torch.zeros(B, 3, *prj_im_sz).to(device)
    else:
        im_gray = prj_brightness * torch.ones(B, 3, *prj_im_sz).to(device)
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True
    prj_adv.retain_grad()
    optimizer = transformers.AdamW([prj_adv], lr=adv_lr)
    # prj_adv = torch.nn.Parameter(prj_adv)
    # optimizer = torch.optim.Adam([prj_adv], lr=adv_lr) # for application with nn.Module
    return im_gray, prj_adv, optimizer


""" overlay an stealth patch(prj_adv) to the entire image """


def apply_perturbation(prj_adv, images, mean, std, root):
    perturbed_img = []
    # make sure prj_adv is same size as image 224x224
    for im in images:  # (320, 240)
        im = transforms.ToTensor()(im)  # normalised?
        # resize to
        im = transforms.Compose(
            [
                transforms.CenterCrop((240, 240)),
                transforms.Resize((224, 224)),
            ]
        )(im).to(device)
        im = (im + prj_adv).clamp(0.0, 1.0)  # TODO: add eps = cap for change
        from torchvision.utils import save_image

        # save_image(im, f"{root}/perturbed_image.png")
        # print("perturbed image saved to", f"{root}/perturbed_image.png")

        im0 = normalize(im, mean[0].to(device), std[0].to(device))  # for simulation
        im1 = normalize(im, mean[1].to(device), std[1].to(device))  # for real world
        perturbed_img.append(torch.cat([im0, im1], dim=1))
    return torch.cat(perturbed_img, dim=0)


""" using pcnet to project a stealth patch(prj_adv) to the center of the image """


def apply_projection(pcnet, prj_adv, images, mean, std):
    projected_images = []
    cam_im_sz = (240, 320)  # (h,W)
    vla_sz = (224, 224)
    bs = 1
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


# main function
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create openvla model
    print("Creating VLA model.............................................")
    vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
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
    print(
        "{vla_path} model created and loaded............................................."
    )

    # create dataloader
    print("Creating dataloader.............................................")
    bs = 1
    dataset = "libero_spatial"
    server = "/data2/lsc/roboticAttack"
    train_dataloader, val_dataloader = get_dataloader(
        batch_size=bs, dataset=dataset, server=server, vla_path=vla_path
    )
    print("{dataset} dataloader created.............................................")

    # create pcnet model
    print("Creating PCNet model.............................................")
    root = "/data2/lsc/uada/pcnet"
    data_root = abspath(join(root, "data"))
    # model_name = ['PCNet_no_mask_no_rough_d']
    model_name = ["PCNet"]
    setup_list = ["coffee_mug"]
    load_pretrained = True
    pcnet_cfg = get_model_train_cfg(
        model_name, data_root, setup_list, load_pretrained=load_pretrained, plot_on=True
    )
    if load_pretrained:
        print("Pretrained config loaded: ", pcnet_cfg)
    pcnet, model_ret, model_cfg = train_eval_pcnet(pcnet_cfg)
    print("PCNet model created .............................................")

    # run attack
    print("Starting Attack.............................................")
    results_root = abspath(join(root, "attack_run"))
    # name each run with date and time
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = "attack_" + now
    run_root = abspath(join(results_root, run_name))
    os.makedirs(run_root, exist_ok=True)

    prj_adv = myAttack(pcnet, vla, train_dataloader, device, run_root, targeted=False)
    print("prj_adv size:", prj_adv.shape)
    print("Attack finished.............................................")

    print("Saving the patch.............................................")

    adv_img_path = "prj_adv.png"
    T.ToPILImage()(prj_adv.squeeze(0)).save(os.path.join(run_root, adv_img_path))
    print("Patch saved as", adv_img_path)
    print("Done.............................................")
