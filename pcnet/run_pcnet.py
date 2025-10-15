import os
from os.path import join, abspath
import torch
from img_proc import expand_4d, center_crop as cc
import utils as ut

# set which GPU(s) to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import print_sys_info, set_torch_reproducibility
from train_network import train_eval_pcnet, get_model_train_cfg

print_sys_info()

# [reproducibility] did not see significantly differences when set to True, but True is significantly slower.
set_torch_reproducibility(False)

# training configs
data_root = abspath(join(os.getcwd(), "data"))
# model_name = ['PCNet_no_mask_no_rough_d']
model_name = ["PCNet"]
setup_list = ["coffee_mug"]

pcnet_cfg = get_model_train_cfg(
    model_name, data_root, setup_list, load_pretrained=True, plot_on=True
)
print(
    "-------------------------------------- PCNET config -----------------------------------"
)
print(pcnet_cfg)
print(
    "-------------------------------------- PCNET config -----------------------------------"
)

pcnet, model_ret, model_cfg = train_eval_pcnet(pcnet_cfg)

# test run
device = "cuda" if torch.cuda.is_available() else "cpu"
B = 1
cam_im_sz = (320, 240)
cam_scene = cc(ut.torch_imread("/data2/lsc/uada/cam_scene.png"), cam_im_sz[::-1])
cam_scene_batch = cam_scene.expand(B, -1, -1, -1).to(device)

prj_brightness = 0.5
prj_im_sz = (256, 256)
im_gray = prj_brightness * torch.ones(B, 3, *prj_im_sz).to(device)
prj_adv = im_gray.clone()
prj_adv.requires_grad = True

cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)

print("Infered image size: ", cam_infer.shape)

from torchvision.utils import save_image

save_image(cam_infer[0].cpu(), "cam_infer.png")
save_image(prj_adv[0].cpu(), "prj_adv.png")
