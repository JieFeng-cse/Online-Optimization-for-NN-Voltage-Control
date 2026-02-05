import numpy as np
import torch
import os

import datetime

from utils import test_online_performance_NN_w_topo_change_lasso


import argparse

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="DDPG Experiment Parameters")
    parser.add_argument("--traj-num", type=int, default=200, help="Number of trajecotries")

    parser.add_argument("--topo-name", type=str, default='none', help="Change topology name")
    parser.add_argument("--env-name", type=str, default='13bus', help="env name: 13bus or 56bus")
    parser.add_argument("--sens-method", type=str, default='naive', help="sensitivity estimation method")
    parser.add_argument("--sens-after-topo", action="store_true", help="online regression to update sensitivity after topology estimation")
    
    parser.add_argument("--use-sensitivity", action="store_true", help="If true, use estimated sensitivity for online update, else use topology change detection")

    parser.add_argument("--online-update", action="store_true", help="Online update the policy")
    parser.add_argument("--no-update", action="store_true", help="No online update at all time")
    return parser.parse_args()

def print_result_output(args, ep_cost_list):
    """
    Generates and prints a descriptive string based on the flags:
    - wrong_model
    - wrong_update
    - correct_update
    - use_XP
    - online_update
    - no_update
    """

    no_update = args.no_update
    use_sense = args.use_sensitivity
    env_name = args.env_name
    
    sens_method = args.sens_method
    

    result_str = f"{env_name}"
    if use_sense:
        result_str += f" using sensitivity for online optimization, using {sens_method}"
        
    elif not no_update:
        result_str += " using topology change detection for online optimization,"
    else:
        result_str += "no update."

    # Generate timestamped filename
    result_str += f'final average cost: {np.mean(ep_cost_list)}, std {np.std(ep_cost_list)}, '
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    filename = f"results/final_result/result_{timestamp}.txt"

    # Write to file
    with open(filename, "w") as file:
        file.write(result_str)

    print(result_str)
    print(np.mean(ep_cost_list))
    print(np.std(ep_cost_list))
    
# Parse arguments
args = get_args()
traj_num = args.traj_num
online_update_flag = args.online_update
no_update = args.no_update
sens_method = args.sens_method

sens_after_topo = args.sens_after_topo

if no_update:
    flag_grad = 0
else:
    flag_grad = 1

update_topo_flag = False



use_sense = args.use_sensitivity #use sensitivity or topology change detection.
env_name = args.env_name
if env_name == '13bus':
    base_v = 4.16
    injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    control_bus = np.array([2, 7, 9])

    probing_nodes_id = [1,3,5,6,7,8,9,10,11]
    probing_nodes = [2,4,6,7,8,9,10,11,12]

    n_control = len(control_bus)  
    n_bus = len(injection_bus)
    Q_limit = np.asarray([[-1.5,1.5],[-1.4,1.4],[-1.0,0.6]])
    C = np.asarray([1.2,0.3,0.6])*0.1

    coef_c = 0.02
    # Controled buses
    controller_id = [1,6,8]

    Q = np.eye(n_bus)  # State cost matrix
    R = np.eye(n_control)
    for i in range(n_control):
        R[i,i]=C[i]*coef_c
elif env_name == '56bus':
    base_v = 12
    injection_bus = np.array([18, 21, 30, 45, 53]) - 1 
    control_bus = injection_bus
    controller_id = control_bus.copy() - 1
    
    probing_nodes_id = (injection_bus -1).tolist()
    probing_nodes = injection_bus
    
    n_control = len(control_bus)  
    n_bus = 55

    C = np.asarray([1.2, 0.3, 0.6, 0.5, 0.5])*0.5

    coef_c = 0.01
    Q = np.eye(n_bus)  # State cost matrix
    R = np.eye(n_control)
    for i in range(n_control):
        R[i,i]=C[i]*coef_c
else:
    assert False, 'wrong env name'



Phi_1 = torch.zeros((n_bus,n_control)).to(device)
for i, id in enumerate(controller_id):
    Phi_1[id,i] = 1
Phi_2 = torch.zeros((n_control, n_bus)).to(device)
for i, id in enumerate(controller_id):
    Phi_2[i,id] = 1
    

H_s = None

type_name = 'single-phase'
path_base = f'checkpoints/{type_name}/{env_name}/NN_control'
topo_name = 'original_topo'
ep_cost_list, ep_after_topo_cost_list, sens_time_list, cost_traj_list, sens_err_list, succ_rate = test_online_performance_NN_w_topo_change_lasso(traj_num, controller_id, Q, R, path_base, topo_name, probing_nodes= probing_nodes, 
                                                        probing_nodes_id=probing_nodes_id,
                               num_steps=1000, use_sense=use_sense, sens_method = sens_method,
                            online_update_flag=online_update_flag, update_topo_flag=update_topo_flag,
                            flag_grad=flag_grad, H_s = H_s, n_control=n_control,sens_after_topo=sens_after_topo,\
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), env_name=env_name, base_v=base_v)
print_result_output(args, ep_cost_list)
print('***********episodic cost after topology change**************')
print(np.mean(ep_after_topo_cost_list))
print(np.std(ep_after_topo_cost_list))
print('****************************')
if use_sense:
    print('sensitivity estimation time')
    print(np.mean(sens_time_list))
    print(np.std(sens_time_list))
if use_sense:
    print('use sensitivity estimation')
else:
    print(np.mean(sens_time_list))
    print(np.std(sens_time_list))
    print('use topology change detection for update.')
    
cost_traj_list = np.array(cost_traj_list)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('results', exist_ok=True)
if use_sense:
    method = 'sense' + sens_method
else:
    method = 'topo_detect'
filename = f"results/cost_traj/cost_{timestamp}_{method}.npy"
np.save(filename, cost_traj_list)

valid_vals = [x for x in sens_err_list if x is not None]
mean_val = np.mean(valid_vals) if valid_vals else float("nan")

print("Mean error of sensitivity estimation=", mean_val)
