import os
from os.path import join, abspath
from projector_based_attack import run_projector_based_attack, get_attacker_cfg, summarize_single_attacker


# %% (5.1) [server] Train PCNet and perform SPAA (you may want to transfer data to the server and train/attack from that machine)
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
device_ids = [0]
data_root = abspath(join(os.getcwd(), 'data'))
setup_name = 'coffee_mug'  # debug
setup_path = join(data_root, 'setups', setup_name)

# get the attack configs
spaa_cfg = get_attacker_cfg(attacker_name='SPAA', data_root=data_root, setup_list=[setup_name], device_ids=device_ids, load_pretrained=True, plot_on=True) # using pre-trained pcnet

# start attack
spaa_cfg = run_projector_based_attack(spaa_cfg)
print(f'Finish SPAA attack, you may want to transfer the data in {join(setup_path, "prj/adv")} to the local machine for real projector-based attacks')


# # %% (5.2) [local] Project and capture SPAA generated adversarial projections, then summarize the attack results
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device_ids = [0]
# data_root = abspath(join(os.getcwd(), '../../data'))
# setup_name = 'test_setup'  # debug
# setup_path = join(data_root, 'setups', setup_name)

# # get the attack configs
# spaa_cfg = get_attacker_cfg(attacker_name='SPAA', data_root=data_root, setup_list=[setup_name], device_ids=device_ids, load_pretrained=False, plot_on=True)

# project_capture_real_attack(spaa_cfg)

# # summarize the attack
# spaa_ret = summarize_single_attacker(attacker_name=spaa_cfg.attacker_name, data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
#                                      device=spaa_cfg.device, device_ids=spaa_cfg.device_ids)
