import os
from os.path import join, abspath
from openvla_dataloader import get_dataloader
import numpy as np
import torch
import random
from PIL import Image 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


batch_size = 2
root = abspath("/data2/lsc/roboticAttack")
dataset = "bridge_orig"
vla_path = "openvla/openvla-7b"
# dataset = "liberao_spatial"
# vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
train_dataloader, val_dataloader = get_dataloader(batch_size=batch_size,
                                                  dataset=dataset,
                                                  server=root, 
                                                  vla_path=vla_path)
set_seed(42)

train_iterator = iter(train_dataloader)
val_iterator = iter(val_dataloader)
print(" Visualising data from dataloader...")
for i in range (10):
    data = next(train_iterator)
    pixel_values = data["pixel_values"]
    pixel_values[1].show()

    labels = data["labels"].to(device)
    attention_mask = data["attention_mask"].to(device)
    input_ids = data["input_ids"].to(device)
    
    print(f"batch {i}: pixel_values: {pixel_values}, labels: {labels}, attention_mask : {attention_mask}, input_ids: {input_ids}")
    
    
    