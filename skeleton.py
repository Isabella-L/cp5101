import os
from os.path import join
import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
import argparse
import numpy as np
import uuid
import random
from tqdm import tqdm
import cv2 as cv


def main():
    set_seed(42)
    pwd = os.getcwd()
    run_id = str(uuid.uuid4())
    out_dir = os.path.join(os.getcwd(), "run", "ProjectedUADA", run_id)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VLA model
    # default bridge_org dataset
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        force_download=True
    ).to(device)
    for param in vla.parameters():     # freeze the model
        param.requires_grad = False
        
    print("Openvla model openvla-7b loaded successfully.")
    
    # ---- dummy input for attack ----
    setup_info = {
        "classifier_crop_sz": (240.240),
        "prj_im_sz": (256, 256),   # (H, W)
        "prj_brightness": 0.50,
        "cam_im_sz": (320, 240),
    }   
    
    cam_scene = cc(torch_imread(join(pwd, 'cam_scene.png')), setup_info['cam_im_sz'][::-1]) # cam-captured scene (Is), ref/img_0002

    attack(pcnet = None,  # Placeholder for PCNet model
           vla = vla,
           target_idx = [],
           targeted = False,
           cam_scene = cam_scene,  # Placeholder for camera scene input)
           stealth_loss= 'caml2',  
           device = device,
            setup_info = setup_info,  
            out_loop = 1,
            inner_loop = 50)
    
    print("Attack done!")
    
# ---- attack function ----
def attack( pcnet, vla, 
            target_idx, targeted,
            cam_scene, stealth_loss, 
            device, setup_info, 
            out_loop = 1,
            inner_loop=50):
    
    device = torch.device(device)
    num_target = len(target_idx)
    
    # camera-captured scene image used for attacks
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1).to(device)
    
    # projector input image
    im_gray = setup_info['prj_brightness'] * torch.ones(num_target, 3, *setup_info['prj_im_sz']).to(device)  # TODO: cam_train.mean() may be better?
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True
    
    # [TODO] only untargeted attack for now
    v = 7 if targeted else 0
    
    # learning rates
    adv_lr = 2  # SPAA Algorithm 1's \beta_1: step size in minimizing adversarial loss
    col_lr = 1  # SPAA Algorithm 1's \beta_2: step size in minimizing stealthiness loss
    adv_w   = 1                                # adversarial loss weights
    prjl2_w = 0.1 if 'prjl2' in stealth_loss else 0    # projector input image l2 loss weights, SPAA paper prjl2_w=0
    caml2_w = 1   if 'caml2' in stealth_loss else 0    # camera captured image l2 loss weights
    camdE_w = 1   if 'camdE' in stealth_loss else 0    # camera captured image deltaE loss weights
    
    # adversarial confidence threshold
    p_thresh = 0.9
    
    # stealthness threshold 
    d_thr = 5 # cfg_default.d_threshes       = [5, 7, 9, 11]
    
    
    for i in tqdm(range(out_loop)):
        # outer loop to generalise over all scenes
        for inner_loop in range(inner_loop): 
            # TODO: add in PCNet
            # cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)
            # cam_infer = cam_scene     
            
            output: CausalLMOutputWithPast = vla(
                
                
            
            

# ---- helper functions ----    
def expand_4d(x):
    # expand a 1D,2D,3D tensor to 4D tensor (BxCxHxW)
    for i in range(4 - x.ndim):
        x = x[None]
    return x

def cc(x, size): 
    # center crop an image by size
    h, w = x.shape[-2:]
    th, tw = size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return x[..., i:i + th, j:j + tw]

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# read a single image to float tensor CxHxW
def torch_imread(filename):
    return torch.Tensor(cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255

            
if __name__ == "__main__":
    main()