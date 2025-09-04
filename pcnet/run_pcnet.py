import os
from os.path import join, abspath

# set which GPU(s) to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import print_sys_info, set_torch_reproducibility
from train_network import train_eval_pcnet, get_model_train_cfg

print_sys_info()

# [reproducibility] did not see significantly differences when set to True, but True is significantly slower.
set_torch_reproducibility(False)

# training configs
data_root = abspath(join(os.getcwd(), 'data'))
# model_name = ['PCNet_no_mask_no_rough_d']
model_name = ['PCNet']
setup_list = ['coffee_mug']

pcnet_cfg = get_model_train_cfg(model_name, data_root, setup_list, load_pretrained=False, plot_on=True)
print('-------------------------------------- PCNET config -----------------------------------')
print(pcnet_cfg)
print('-------------------------------------- PCNET config -----------------------------------')

_, pcnet_ret, _ = train_eval_pcnet(pcnet_cfg)
    
