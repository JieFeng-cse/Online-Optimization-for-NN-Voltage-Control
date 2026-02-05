from numpy import linalg as LA
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
from tqdm import tqdm
from topo_est import build_sensitivity, topology_recovery, rls_update, topology_detection_lasso
from online_opt import online_update_Xp_NN
from env import create_13bus, IEEE13bus, create_56bus, VoltageCtrl_nonlinear
from models import SafePolicyNetwork_SetPoint
from .network_utils import create_RX_from_net

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

slop_list_pre = torch.tensor([[1.],[1.],[1.]]).to(device)
pp_list_pre = torch.tensor([[-0.028],[0.049],[0.019]]).to(device)

def build_init_env():
    """
    Build the initial environment for 13-bus system.

    Returns:
        _type_: Simulation environment
    """
    pp_net = create_13bus()
    _, X = create_RX_from_net(pp_net, noise=0)
    X_gen = X/np.square(4.16)
    injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    control_bus = np.array([2, 7, 9])

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

    # all the probing buses can be controlled
    env = IEEE13bus(pp_net, injection_bus, injection_bus, \
                    Q = Q, R = R)
    return env

def build_init_56bus():
    """
    Build the initial environment for 56-bus system.

    Returns:
        _type_: Simulation environment
    """
    pp_net = create_56bus()
    _, X = create_RX_from_net(pp_net, noise=0)
    X_gen = X/np.square(12)

    injection_bus = np.array([18, 21, 30, 45, 53]) - 1 
    control_bus = injection_bus
    controller_id = control_bus.copy() - 1

    n_control = len(control_bus)  
    n_bus = 55

    C = np.asarray([1.2, 0.3, 0.6, 0.5, 0.5])*0.5

    coef_c = 0.01
    Q = np.eye(n_bus)  # State cost matrix
    R = np.eye(n_control)
    for i in range(n_control):
        R[i,i]=C[i]*coef_c

    # all the probing buses can be controlled
    env = VoltageCtrl_nonlinear(pp_net, injection_bus, control_bus, \
                    Q = Q, R = R)
    return env
    

def get_true_topo(env, probing_nodes, probing_nodes_id, base_v = 4.16, use_linear=False, N=12):
    _, X = create_RX_from_net(env.network, noise=0)  
    X_gen = X/np.square(base_v)
    X_full = X_gen
    X_sens = X_full[:, probing_nodes_id]
    _, lines = topology_recovery(X_sens, probing_nodes, linear_system=use_linear, N = N) 
    return X_full, lines
    

def get_init_topo(env, policy_list, probing_nodes, probing_nodes_id, n_bus, controller_id):
    """
    This function collects data using initial policies to estimate the initial topology.
    This is not used in the current version of the code, but can serve as a reference for future work.
    """
    #Note, there is mismatch of 56 bus simulation and 13 bus simulation, especially in state observation
    state, _ = env.reset(seed = 13)
    dis_list = []
    v_diff_list = []
    num_steps = 1000
    update_topo_flag = True
    find = False
    n_control = len(controller_id)  

    done= False
    disturbance_level = 0.01
    n_prob = len(probing_nodes_id)
    last_action = torch.zeros((n_bus,1)).to(device)

    for i in range(num_steps):
        action = torch.zeros((n_bus,1)).to(device)
        if update_topo_flag and (not find):
            dis_q = torch.randint(-1, 2, (n_prob, 1)).to(device) * disturbance_level
        else:
            dis_q = torch.zeros((n_prob,1)).to(device)
            
        for id in range(n_control):
            state_torch = torch.square(torch.FloatTensor(np.asarray(state[controller_id[id]])).unsqueeze(0).to(device))
            tmp_action = policy_list[id](state_torch).squeeze()
            id_in_prob = probing_nodes_id.index(controller_id[id])
            dis_q[id_in_prob] -= tmp_action
        
        action[probing_nodes_id] = last_action[probing_nodes_id] + dis_q
        action_np = action.clone().detach().cpu().numpy()

        next_state, reward, reward_sep, done = env.step_Preward(action_np, action_np[controller_id])
        if update_topo_flag and done:
            v_diff = np.expand_dims(np.square(next_state), axis=1)-np.expand_dims(np.square(state), axis=1)
            u_prob = dis_q.clone().detach().cpu().numpy()
            
            dis_list.append(u_prob)
            v_diff_list.append(v_diff)
            
            X_sens, find = build_sensitivity(probing_nodes, dis_list, v_diff_list)
            if find:
                X_full, lines = topology_recovery(X_sens, probing_nodes) 
                flat_id = {id for line in lines for id in line}
                if (len(lines)) == n_bus and (flat_id == set(range(n_bus+1))):
                    print(lines)
                    break
            
        last_action = action.clone()
        state = np.copy(next_state)
    return X_full, lines


def load_policy(n_control, path_base, topo_name,  env,\
                 action_dim=1, hidden_dim=64, obs_dim=1, type_name='single-phase'):
    policy_list = []
    for i in range(n_control):
        policy_net = SafePolicyNetwork_SetPoint(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        pth_policy = osp.join(path_base, f'{topo_name}_policy_net_checkpoint_a{i}.pth')
        policynet_dict = torch.load(pth_policy)
        policy_net.load_state_dict(policynet_dict)
        policy_list.append(policy_net)
    return policy_list


def sample_random_topo_change():
    potential_changes = {
                     's1': [5, 3, 9],
                     's2': [5, 3, 12],
                     's3': [5, 2, 7],
                     's4': [5, 1, 9],
                     's5': [6, 2, 7],
                     's6': [6, 4, 7],
                     's7': [[5,2,7],[6,3,9]],
                     's8': [[6,4,7],[10,3,9]]
                     }
    solution = {
        's1': [(0,4), (2,8)],
        's2': [(0,4), (2,11)],
        's3': [(0,4), (1,6)],
        's4': [(0,4), (0,8)],
        's5': [(4,6), (1,6)],
        's6': [(4,6), (3,6)],
        's7': [(0,4), (1,6), (4,6), (2,8)],
        's8': [(2,8), (3,6), (4,6), (4,8)]
    }
    id = np.random.choice(len(potential_changes))
    key = f's{id+1}'
    return potential_changes[key], solution[key], key

def sample_random_topo_change_56bus():
    potential_changes = {
                     's1': [39, 1, 40],
                     's2': [47, 17, 48],
                     's3': [13, 33, 14],
                     's4': [40, 47, 45],
                     's5': [47, 10, 48],
                     's6': [[39,1,40],[47,10,48]],
                     's7': [[39,1,40],[13,33,14]],
                     's8': [[32,19,33],[40,47,41]]
                     }
    solution = {
        's1': [(0,39), (32,39)],
        's2': [(16,47), (45,47)],
        's3': [(13,32), (11,13)],
        's4': [(44,46), (39,40)],
        's5': [(9,47), (45,47)],
        's6': [(0,39), (32,39), (9,47), (45,47)],
        's7': [(0,39), (32,39), (13,32), (11,13)],
        's8': [(18,32), (39,40), (40,46), (30,32)]
    }
    id = np.random.choice(len(potential_changes))
    key = f's{id+1}'
    return potential_changes[key], solution[key], key

def collect_one_step(state, u, X):
    next_state = np.sqrt(X@u + np.expand_dims(state**2,1)) 
    v_diff = next_state.squeeze()**2 - state**2
    return v_diff, next_state.squeeze()

def change_topo(env, topo_change):
    topo_change = np.asarray(topo_change)
    if len(topo_change.shape)==2:
        for i in range(topo_change.shape[0]):
            change_topo_inner(env, topo_change[i,0], topo_change[i,1], topo_change[i,2], 0.3)
    else:
        change_topo_inner(env, topo_change[0], topo_change[1], topo_change[2], 0.3)

def change_topo_inner(env, line_id, from_bus, to_bus, x_ohm_per_km):
    env.network.line.loc[line_id,'from_bus'] = from_bus
    env.network.line.loc[line_id,'to_bus'] = to_bus
    env.network.line.loc[line_id,'x_ohm_per_km'] = x_ohm_per_km
    
def err_cal(u, v, X):
    u = u.reshape(-1,1)
    v = v.reshape(-1,1)
    e = v - X @ u
    return np.linalg.norm(e)

def run_traj_NN_w_topo_change_sens_lasso(controller_id, Q, R, env, state, all_state, seed, X_full, lines, probing_nodes_id,\
             y_q_theta_dict, G_t_theta_dict, name_list, policy_list, state_traj, \
             pp_traj, action_traj, cost_traj, episodic_cost, after_topo_cost, last_action, H_s, Phi_1, Phi_2,\
             n_control, lr=0.1, lambda_y = 1.0, num_steps = 5000, disturbance_level=0.01, sens_method = 'naive',\
             flag_grad = 1, dis_flag=False, online_update_flag=False, use_XP=False, update_topo_flag=True, \
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), bus56=False, topo_change_step=50):
    n_bus = len(lines)
    n_prob = len(probing_nodes_id)
    X_sens = X_full[:, probing_nodes_id]
    detector = topology_detection_lasso(n_rows=n_bus, lines=lines.copy(), window=10, min_hits= None, eps=None, jump_ratio = None, tol=None, mask_gamma = 10)
    detector.update_X_est(X_full)
    invX_est = np.linalg.inv(X_full)
    sens_err = None
    
    delta = 1e2
    X_rls = X_full[:, controller_id].copy()
    P_rls = np.eye(n_control) * delta 
    
    if dis_flag:
        state, all_state = env.reset(seed)
    dis_list = []
    v_list = []
    
    sense_converge_time = num_steps
    sense_converge_flag = False
    update_topo_flag = update_topo_flag
    find = False
    done= False
    X_P = None
    err_rls_history = []    
    scenario_key = None
    for i in range(num_steps):
        action = torch.zeros((n_bus,1)).to(device)
        dis_q = torch.zeros((n_prob,1)).to(device)
        pp_list = []

        for id in range(n_control):
            pp_list.append((1+0.08*F.tanh(policy_list[id].theta2)).clone().detach().cpu().numpy())
            if bus56:
                state_torch = torch.square(torch.FloatTensor(np.asarray(all_state[controller_id[id]])).unsqueeze(0).to(device))
            else:
                state_torch = torch.square(torch.FloatTensor(np.asarray(state[controller_id[id]])).unsqueeze(0).to(device))
            tmp_action = policy_list[id](state_torch).squeeze()
            id_in_prob = probing_nodes_id.index(controller_id[id])
            dis_q[id_in_prob] = -tmp_action
        action[probing_nodes_id] = last_action[probing_nodes_id] + dis_q

        pp_traj.append(pp_list)
        action_np = action.clone().detach().cpu().numpy()
        if bus56:
            next_state, reward, next_state_all, done = env.step_Preward(action_np[controller_id], action_np[controller_id])
        else:
            next_state, reward, reward_sep, done = env.step_Preward(action_np, action_np[controller_id])
        
        if not bus56:
            v_diff = np.expand_dims(np.square(next_state), axis=1)-np.expand_dims(np.square(state), axis=1)
        else:
            v_diff = np.expand_dims(np.square(next_state_all), axis=1)-np.expand_dims(np.square(all_state), axis=1)
        u_prob = dis_q.clone().detach().cpu().numpy()
        
        if not bus56:
            v_diff_pre, _ = collect_one_step(state, u_prob, X_sens)
        else:
            v_diff_pre, _ = collect_one_step(all_state, u_prob, X_sens)
        v_diff_new = np.expand_dims(v_diff_pre, axis=1) - v_diff
        R_obs = invX_est@v_diff_new
        
        err = err_cal(u_prob, v_diff, X_sens)

        if i % 200 == 0:
            env.step_disturbance()
        if i == topo_change_step:
            if bus56:
                topo_change, solution, scenario_key = sample_random_topo_change_56bus()
            else:
                topo_change, solution, scenario_key = sample_random_topo_change()
            print('topology changed: ', topo_change)
            change_topo(env, topo_change)

        event = detector.monitor(err)
        if event is not None:
            t_jump, tag = event
            if 'topo' in tag and (not detector.topo_change_detected):
                detector.reset_hist()
                detector.topo_change_detected = True
                detector.current_topo_available = False
                find = False
                dis_list = []
                v_list = []
                print('topo change detected', i)
                
        
        if detector.topo_change_detected:
            if not bus56:
                dis_list.append(dis_q[[0,3,5]].clone().detach().cpu().numpy())
                u_rls = dis_q[[0,3,5]].clone().detach().cpu().numpy()
                v_list.append(np.expand_dims(np.square(next_state), axis=1)-np.expand_dims(np.square(state), axis=1))
            else:
                dis_list.append(dis_q.clone().detach().cpu().numpy())
                u_rls = dis_q.clone().detach().cpu().numpy()
                v_list.append(np.expand_dims(np.square(next_state_all), axis=1)-np.expand_dims(np.square(all_state), axis=1))
            if sens_method == 'naive':
                if find:
                    err_rls = err_cal(u_rls, v_diff, X_P)
                    err_rls_history.append(err_rls)
                    find = False
                    if len(err_rls_history) >= 2:
                    # Calculate the mean of the last two errors
                        mean_err = np.mean(err_rls_history[-2:])
                        if mean_err < 0.0001:
                            find = True
                        else:
                            find = False
            elif sens_method == 'rls':
                X_rls, P_rls = rls_update(u_rls, v_diff, X_rls, P_rls)
                err_rls = err_cal(u_rls, v_diff, X_rls)
                err_rls_history.append(err_rls)
                if len(err_rls_history) >= 2:
                # Calculate the mean of the last two errors
                    mean_err = np.mean(err_rls_history[-2:])
                    if mean_err < 0.0001:
                        find = True
            else:
                assert False, 'undefined sensitivity estimation method'
            if not sense_converge_flag and find:
                sense_converge_flag = True
                sense_converge_time = i
                print(i)
                    
        if not bus56:       
            state_hat = torch.square(torch.FloatTensor(np.asarray(next_state)).unsqueeze(0).to(device)).T 
        else:
            state_hat = torch.square(torch.FloatTensor(np.asarray(next_state_all)).unsqueeze(0).to(device)).T 
        # Calculate the cost 
        ht = ((state_hat-1).T @ Q @ (state_hat-1) + action[controller_id].T @ R @ action[controller_id]).squeeze()
            
        # cumulative cost
        episodic_cost += ht.detach().cpu().numpy()
        cost_traj.append(episodic_cost.copy())
        if i > 300 and i < 400:
            after_topo_cost += ht.detach().cpu().numpy()
        
        if detector.current_topo_available: 
            H_s = torch.from_numpy(detector.X_est[:, controller_id]).float().to(device)
            y_q_theta_dict, G_t_theta_dict, policy_list = online_update_Xp_NN(policy_list, y_q_theta_dict, G_t_theta_dict, state_hat, \
                                                                        action[controller_id], name_list, Q, R, controller_id,\
                                                                        H_s, n_control,Phi_2, lr, lambda_y, flag_grad)
        elif find:
            if sens_method == 'naive':
                H_s = torch.from_numpy(X_P).float().to(device)
            elif sens_method == 'rls':
                H_s = torch.from_numpy(X_rls).float().to(device)
            else:
                assert False, 'undefined sensitivity estimation method'
                
            y_q_theta_dict, G_t_theta_dict, policy_list = online_update_Xp_NN(policy_list, y_q_theta_dict, G_t_theta_dict, state_hat, \
                                                                        action[controller_id], name_list, Q, R, controller_id,\
                                                                        H_s, n_control,Phi_2, lr, lambda_y, flag_grad)

        last_action = action.clone()
        if not bus56:
            state = np.copy(next_state)
            state_traj.append(state)
        else:
            all_state = np.copy(next_state_all)
            state_traj.append(all_state)
        action_traj.append(last_action[controller_id].clone().detach().cpu().numpy())
    R_c, X_c = create_RX_from_net(env.network, noise=0)
    if not bus56:
        X_c = X_c/np.square(4.16)
    else:
        X_c = X_c/np.square(12)
    if sens_method == 'naive' and find:
        sens_err = np.linalg.norm(X_c[:,controller_id] - X_P)
    else:
        sens_err = np.linalg.norm(X_c[:,controller_id] - X_rls)
    cost_traj = np.array(cost_traj)
    return policy_list, state_traj, pp_traj, action_traj, last_action, episodic_cost, after_topo_cost, H_s, y_q_theta_dict, G_t_theta_dict, X_P, online_update_flag, sense_converge_time, cost_traj, sens_err, scenario_key

def no_change(detector):
    detector.topo_change_detected = False
    detector.previous_topo_available = True
    detector.current_topo_available = True
    detector.reset_hist()
    
def run_traj_NN_w_topo_change_lasso(controller_id, Q, R, env, state, all_state, seed, X_full, lines, probing_nodes_id,\
             y_q_theta_dict, G_t_theta_dict, name_list, policy_list, state_traj, \
             pp_traj, action_traj, cost_traj, episodic_cost, after_topo_cost, last_action, H_s, Phi_1, Phi_2,\
             n_control, lr=0.1, lambda_y = 1.0, num_steps = 5000, disturbance_level=0.01,\
             flag_grad = 1, dis_flag=False, online_update_flag=False, update_topo_flag=True, sens_after_topo=False,\
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), bus56=False, topo_change_step=50):
    n_bus = len(lines)
    n_prob = len(probing_nodes_id)
    X_sens = X_full[:, probing_nodes_id]
    sense_converge_time = num_steps
    find = False
    if bus56:
        delta = 1e2
        window = 15
        mask_gamma = 70
    else:
        delta = 1e2
        window = 10
        mask_gamma = 15
    
    detector = topology_detection_lasso(n_rows=n_bus, lines=lines.copy(), window= window, min_hits= None, eps=None, jump_ratio = None, tol=None, mask_gamma = mask_gamma)
    detector.update_X_est(X_full)
    invX_est = np.linalg.inv(X_full)
    sens_err = None
    
    cooldown = 0
    line_succ = 0
    node_succ = 0
    detect_succ = 0
    solution = None
    scenario_key = None
    
    if bus56:
        X_rls = X_sens.copy()
    else:
        X_rls = X_full[:, controller_id]
    P_rls = np.eye(n_control) * delta 
    rls_update_flag = False
    
    if dis_flag:
        state, all_state = env.reset(seed)
    dis_list = []
    v_list = []
    update_topo_flag = update_topo_flag
    X_P = None
    line_info = None
    line_succ_final = 0
    line_info_final = None
    err_rls_history = []   
        
    for i in range(num_steps):
        action = torch.zeros((n_bus,1)).to(device)
        dis_q = torch.zeros((n_prob,1)).to(device)
        pp_list = []

        for id in range(n_control):
            pp_list.append((1+0.08*F.tanh(policy_list[id].theta2)).clone().detach().cpu().numpy())
            if bus56:
                state_torch = torch.square(torch.FloatTensor(np.asarray(all_state[controller_id[id]])).unsqueeze(0).to(device))
            else:
                state_torch = torch.square(torch.FloatTensor(np.asarray(state[controller_id[id]])).unsqueeze(0).to(device))
            tmp_action = policy_list[id](state_torch).squeeze()
            id_in_prob = probing_nodes_id.index(controller_id[id])
            dis_q[id_in_prob] = -tmp_action
        action[probing_nodes_id] = last_action[probing_nodes_id] + dis_q

        pp_traj.append(pp_list)
        action_np = action.clone().detach().cpu().numpy()
        
        if bus56:
            next_state, reward, next_state_all, done = env.step_Preward(action_np[controller_id], action_np[controller_id])
        else:
            next_state, reward, reward_sep, done = env.step_Preward(action_np, action_np[controller_id])
        
        if not bus56:
            v_diff = np.expand_dims(np.square(next_state), axis=1)-np.expand_dims(np.square(state), axis=1)
        else:
            v_diff = np.expand_dims(np.square(next_state_all), axis=1)-np.expand_dims(np.square(all_state), axis=1)
        u_prob = dis_q.clone().detach().cpu().numpy()
        
        if not bus56:
            v_diff_pre, _ = collect_one_step(state, u_prob, X_sens)
        else:
            v_diff_pre, _ = collect_one_step(all_state, u_prob, X_sens)
        
        v_diff_new = np.expand_dims(v_diff_pre, axis=1) - v_diff
        R_obs = invX_est@v_diff_new
        
        v_list.append(state)
        dis_list.append(u_prob)
        
        err = err_cal(u_prob, v_diff, X_sens)

        if i % 200 == 0:
            env.step_disturbance()
        if i == topo_change_step:
            if bus56:
                topo_change, solution, scenario_key = sample_random_topo_change_56bus()
            else:
                topo_change, solution, scenario_key = sample_random_topo_change()
            print('topology changed: ', topo_change)
            change_topo(env, topo_change)

        # if np.linalg.norm(v_diff)> 1e-4:
        if cooldown == 0:
            event = detector.monitor(err)
            if event is not None:
                t_jump, tag = event
                if 'topo' in tag and (not detector.topo_change_detected):
                    detector.reset_hist()
                    detector.topo_change_detected = True
                    detect_succ = 1
                    detector.current_topo_available = False
                    print('topo change detected', i)
            
            if detector.topo_change_detected:
                active, stable_ids = detector.update(R_obs, v_diff)
                if detector.hist_count > detector.window:
                    if solution is not None:
                        if all(num in stable_ids for tup in solution for num in tup):
                            node_succ = 1
                    if len(stable_ids) > 2:
                        line_change_flag, detect_failed, changed_lines, potential_line_list = detector.est_line_change(stable_ids)
                        if not line_change_flag:
                            no_change(detector)
                            rls_update_flag = True
                        if line_change_flag and not detect_failed:
                            X_full_new, line_info, line_succ = detector.recover_full_topo(potential_line_list, solution, detect_failed)
                            if line_info['fault']:
                                line_change_flag = False
                                rls_update_flag = True
                                no_change(detector)
                            
                        if line_change_flag:
                            line_info_final = line_info
                            if sense_converge_time == num_steps:
                                sense_converge_time = i
                            detector.update_X_est(X_full_new)
                            invX_est = np.linalg.inv(X_full_new)
                            X_sens = X_full_new[:, probing_nodes_id]
                            if sens_after_topo:
                                if bus56:
                                    X_rls = X_sens.copy()
                                else:
                                    X_rls = X_full_new[:, controller_id]
                            line_change_flag = False
                            rls_update_flag = True
                            
                            print(f'good, topo change detected and updated, line candidates: {(line_info["line"], i)}') 
                            cooldown = 50
                    else:
                        no_change(detector)
        else:
            cooldown -= 1
                    
        if sens_after_topo and rls_update_flag:
            if not bus56:
                u_rls = dis_q[[0,3,5]].clone().detach().cpu().numpy()
            else:
                u_rls = dis_q.clone().detach().cpu().numpy()
            X_rls, P_rls = rls_update(u_rls, v_diff, X_rls, P_rls, lam=0.999)
            err_rls = err_cal(u_rls, v_diff, X_rls)
            err_rls_history.append(err_rls)
            if len(err_rls_history) >= 2:
            # Calculate the mean of the last two errors
                mean_err = np.mean(err_rls_history[-2:])
                if mean_err < 0.0001:
                    find = True
                    if sense_converge_time == num_steps:
                        sense_converge_time = i
        if not bus56:
            state_hat = torch.square(torch.FloatTensor(np.asarray(next_state)).unsqueeze(0).to(device)).T 
        else:
            state_hat = torch.square(torch.FloatTensor(np.asarray(next_state_all)).unsqueeze(0).to(device)).T 
        # Calculate the cost 
        ht = ((state_hat-1).T @ Q @ (state_hat-1) + action[controller_id].T @ R @ action[controller_id]).squeeze()
            
        # cumulative cost
        episodic_cost += ht.detach().item()
        cost_traj.append(episodic_cost)
        if i > 300 and i < 400:
            after_topo_cost += ht.detach().cpu().numpy()
        
        if detector.current_topo_available or find: 
            if sens_after_topo and rls_update_flag:
                H_s = torch.from_numpy(X_rls).float().to(device)
            else:
                H_s = torch.from_numpy(detector.X_est[:, controller_id]).float().to(device)
                
            y_q_theta_dict, G_t_theta_dict, policy_list = online_update_Xp_NN(policy_list, y_q_theta_dict, G_t_theta_dict, state_hat, \
                                                                        action[controller_id], name_list, Q, R, controller_id,\
                                                                        H_s, n_control,Phi_2, lr, lambda_y, flag_grad)

        last_action = action.clone()
        if bus56:
            all_state = np.copy(next_state_all)
            state_traj.append(all_state)
        else:
            state = np.copy(next_state)
            state_traj.append(state)
        action_traj.append(last_action[controller_id].clone().detach().cpu().numpy())
        line_succ_final = max(line_succ, line_succ_final)
    R_c, X_c = create_RX_from_net(env.network, noise=0)
    if not bus56:
        X_c = X_c/np.square(4.16)
    else:
        X_c = X_c/np.square(12)

    if sense_converge_time < num_steps:
        sens_err = np.linalg.norm(detector.X_est[:, controller_id]-X_c[:, controller_id])
    else:
        sens_err = np.linalg.norm(X_full[:, controller_id]-X_c[:, controller_id])
    cost_traj = np.array(cost_traj)
    final_success = 1
    if line_info_final is None or line_info_final.get('fault'):
        final_success = 0
    else:
        expected_lines = line_info_final.get('line', [])
        final_success = (set(solution) == set(expected_lines))
    print(sense_converge_time)
    return policy_list, state_traj, pp_traj, action_traj, last_action, episodic_cost, after_topo_cost, H_s, y_q_theta_dict, G_t_theta_dict, X_P, \
        online_update_flag, cost_traj, sens_err, sense_converge_time, line_succ_final, node_succ, detect_succ, final_success, scenario_key

def test_online_performance_NN_w_topo_change_lasso(traj_num, controller_id, Q, R, path_base, topo_name, probing_nodes, probing_nodes_id, 
                               num_steps=1000, env_name = '13bus',
                            online_update_flag=False, update_topo_flag=True, use_sense=False, sens_method ='naive',  use_true_X_at_start=True,
                            flag_grad=1, H_s = None, n_control=3, sens_after_topo=False, base_v=4.16,\
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    bus56 = False
    cost_traj_list = []
    sens_error_list = []
    if env_name == '13bus':
        injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])#1,2,5,3,4,7,8,9,6,10,11,12 //2,7,9
        control_bus = np.array([2, 7, 9])
        n_bus = len(injection_bus)
        env = build_init_env()
        lr = 0.1
    elif env_name == '56bus':
        injection_bus = np.array([18, 21, 30, 45, 53]) - 1 
        control_bus = injection_bus
        n_bus = 55
        env = build_init_56bus()
        bus56 = True
        lr = 0.1
    else:
        assert False, 'wrong env name'
    ep_cost_list = []
    ep_after_topo_cost_list = []
    sense_est_time_list = []
    n_control = len(control_bus)  

    
    policy_list = load_policy(n_control, path_base, topo_name,  env)
    if use_true_X_at_start:
        if bus56:
            all_nodes = list(range(1,56))
            all_nodes_id = list(range(0,55))
            X_full,lines = get_true_topo(env, probing_nodes=all_nodes, probing_nodes_id = all_nodes_id,\
        base_v=12.0, use_linear=True, N=n_bus)
        else:
            X_full, lines = get_true_topo(env, probing_nodes, probing_nodes_id, base_v=base_v, use_linear=True, N=n_bus)
    else:
        X_full, lines = get_init_topo(env, policy_list, probing_nodes, probing_nodes_id, n_bus, controller_id)
    succ_good_sens = 0 #number of trials that get a good sensitivity estimation
    line_succ_sum = 0
    node_succ_sum = 0
    detect_succ_sum = 0
    scenario_keys = [f"s{i}" for i in range(1, 9)]
    scenario_stats = {
        key: {"count": 0, "detect": 0, "node": 0, "line": 0, "success": 0}
        for key in scenario_keys
    }
    
    for run_id in tqdm(range(traj_num)):
        if env_name == '13bus':
            net = create_13bus()
            
            env = IEEE13bus(net, injection_bus, injection_bus, \
                    Q = Q, R = R)
        elif env_name == '56bus':
            pp_net = create_56bus()
            env = VoltageCtrl_nonlinear(pp_net, injection_bus, control_bus, \
                Q = Q, R = R)
            
        state, all_state = env.reset(seed = run_id)#original seed=13 #vector of R^12 
        torch.manual_seed(seed = run_id)
        '''
        Create the policy
        '''
        policy_list = load_policy(n_control, path_base, topo_name,  env)
        
        # Build the auxilary variables
        name_list = []
        for name, param in policy_list[0].named_parameters():
            name_list.append(name)
        if env_name == '13bus':
            state_hat = torch.square(torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(device)).T 
        else:
            state_hat = torch.square(torch.FloatTensor(np.asarray(all_state)).unsqueeze(0).to(device)).T 
        q_t_plus_wrt_params_list = [policy_list[id].get_param_grad(state_hat[controller_id[id]]) for id in range(n_control)]

        ############ Record for sizes ##########
        """
        Consider hidden_size as h:
        gradient of each theta2 has size [1]
        gradient of each b&c have size [h]
        gradient of each q&z have size [1,h]
        """
        y_q_theta_dict = {}
        G_t_theta_dict = {}
        for name, param in policy_list[0].named_parameters():
            name_list.append(name)
            if len(param.data.shape) == 1:
                y_q_theta_dict[name] = torch.zeros((n_control, n_control*param.data.shape[0])).to(device)
                G_t_theta_dict[name] = torch.zeros((1, n_control*param.data.shape[0])).to(device)
            elif len(param.data.shape) == 2:
                y_q_theta_dict[name] = torch.zeros((n_control, n_control*param.data.shape[1])).to(device)
                G_t_theta_dict[name] = torch.zeros((1, n_control*param.data.shape[1])).to(device)
            else:
                assert False, "Error: the parameter has more than 2 dimension!"
        '''
        Data log
        '''    
        state_traj = []
        pp_traj = []
        action_traj = []
        cost_traj = []
        if env_name == '13bus':
            state_traj.append(state)
        else:
            state_traj.append(all_state)

        last_action = torch.zeros((n_bus,1)).to(device)
        action_traj.append(last_action[controller_id].clone().detach().cpu().numpy())

        episodic_cost = 0
        after_topo_cost = 0

        seed = run_id

        # helper matrix (transfer n*n to n*m or m*n)
        Phi_1 = torch.zeros((n_bus,n_control)).to(device)
        for i, id in enumerate(controller_id):
            Phi_1[id,i] = 1
        Phi_2 = torch.zeros((n_control, n_bus)).to(device)
        for i, id in enumerate(controller_id):
            Phi_2[i,id] = 1

        Q_torch = torch.tensor(Q, dtype=torch.float32).to(device)
        R_torch = torch.tensor(R, dtype=torch.float32).to(device)
        line_succ = 0
        node_succ = 0
        detect_succ = 0
        # try:
        if use_sense:
            policy_list, state_traj, pp_traj, action_traj, last_action, episodic_cost, after_topo_cost, H_s, _, _, X_P, _, sense_converge_time, cost_traj, sens_err, scenario_key = \
            run_traj_NN_w_topo_change_sens_lasso(controller_id, Q_torch, R_torch, env, state, all_state, seed, X_full, lines, probing_nodes_id, \
                y_q_theta_dict, G_t_theta_dict, name_list, policy_list, state_traj, \
                pp_traj, action_traj, cost_traj, episodic_cost, after_topo_cost, last_action, H_s, Phi_1, Phi_2,\
                n_control, lr=lr, lambda_y = 1.0, num_steps = num_steps, disturbance_level=0.01, sens_method=sens_method,\
                flag_grad = flag_grad, dis_flag=False, online_update_flag=online_update_flag, update_topo_flag=update_topo_flag, \
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), bus56=bus56)
            ep_cost_list.append(episodic_cost)
            ep_after_topo_cost_list.append(after_topo_cost)
            sense_est_time_list.append(sense_converge_time)
        else:
            policy_list, state_traj, pp_traj, action_traj, last_action, episodic_cost, after_topo_cost, H_s, _, _, X_P, _, cost_traj, sens_err, sense_converge_time, line_succ, node_succ, detect_succ, final_success, scenario_key = \
            run_traj_NN_w_topo_change_lasso(controller_id, Q_torch, R_torch, env, state, all_state, seed, X_full, lines, probing_nodes_id, \
                    y_q_theta_dict, G_t_theta_dict, name_list, policy_list, state_traj, \
                    pp_traj, action_traj, cost_traj, episodic_cost, after_topo_cost, last_action, H_s, Phi_1, Phi_2,\
                    n_control, lr=lr, lambda_y = 1.0, num_steps = num_steps, disturbance_level=0.01, sens_after_topo=sens_after_topo,\
                    flag_grad = flag_grad, dis_flag=False, online_update_flag=online_update_flag, update_topo_flag=update_topo_flag, \
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), bus56=bus56)
            ep_cost_list.append(episodic_cost)
            ep_after_topo_cost_list.append(after_topo_cost)
            sense_est_time_list.append(sense_converge_time)
        # print(episodic_cost)
        cost_traj_list.append(cost_traj)
        sens_error_list.append(sens_err)
        line_succ_sum += line_succ
        node_succ_sum += node_succ
        detect_succ_sum += detect_succ
        if use_sense:
            if sense_converge_time < num_steps:
                succ_good_sens += 1
        else:
            if final_success:
                succ_good_sens += 1
        if scenario_key in scenario_stats:
            stats = scenario_stats[scenario_key]
            stats["count"] += 1
            if use_sense:
                stats["success"] += int(sense_converge_time < num_steps)
            else:
                stats["detect"] += int(detect_succ)
                stats["node"] += int(node_succ)
                stats["line"] += int(line_succ)
                stats["success"] += int(final_success)
    succ_rate = succ_good_sens/traj_num
    line_succ_rate = line_succ_sum/traj_num
    print(f'successful rate for successfully detect the event {detect_succ_sum/traj_num}, containing the right nodes is: {node_succ_sum/traj_num}, containing the right line is: {line_succ_rate}, success rate {succ_rate}')
    for key in scenario_keys:
        stats = scenario_stats[key]
        if stats["count"] == 0:
            print(f'scenario {key}: no trials')
            continue
        if use_sense:
            print(f'scenario {key}: success rate {stats["success"]/stats["count"]} ({stats["success"]}/{stats["count"]})')
        else:
            print(
                f'scenario {key}: detect {stats["detect"]/stats["count"]}, '
                f'node {stats["node"]/stats["count"]}, '
                f'line {stats["line"]/stats["count"]}, '
                f'success {stats["success"]/stats["count"]} '
                f'({stats["success"]}/{stats["count"]})'
            )
    return ep_cost_list, ep_after_topo_cost_list, sense_est_time_list, cost_traj_list, sens_error_list, succ_rate

def traj_rollout_no_update(controller_id, env, state, all_state, X_full, lines, probing_nodes_id,\
             policy_list, last_action,n_control, num_steps = 5000,\
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), bus56=False, topo_change_step=50):
    n_bus = len(lines)
    n_prob = len(probing_nodes_id)
    X_sens = X_full[:, probing_nodes_id]
    
    invX_est = np.linalg.inv(X_full)
    
    dis_list = []
    v_diff_list = []
    r_list = []
        
    for i in range(num_steps):
        action = torch.zeros((n_bus,1)).to(device)
        dis_q = torch.zeros((n_prob,1)).to(device)

        for id in range(n_control):
            if bus56:
                state_torch = torch.square(torch.FloatTensor(np.asarray(all_state[controller_id[id]])).unsqueeze(0).to(device))
            else:
                state_torch = torch.square(torch.FloatTensor(np.asarray(state[controller_id[id]])).unsqueeze(0).to(device))
            tmp_action = policy_list[id](state_torch).squeeze()
            id_in_prob = probing_nodes_id.index(controller_id[id])
            dis_q[id_in_prob] = -tmp_action
        action[probing_nodes_id] = last_action[probing_nodes_id] + dis_q

        action_np = action.clone().detach().cpu().numpy()
        
        if bus56:
            next_state, reward, next_state_all, done = env.step_Preward(action_np[controller_id], action_np[controller_id])
        else:
            next_state, reward, reward_sep, done = env.step_Preward(action_np, action_np[controller_id])
        
        if not bus56:
            v_diff = np.expand_dims(np.square(next_state), axis=1)-np.expand_dims(np.square(state), axis=1)
        else:
            v_diff = np.expand_dims(np.square(next_state_all), axis=1)-np.expand_dims(np.square(all_state), axis=1)
        u_prob = dis_q.clone().detach().cpu().numpy()
        
        if not bus56:
            v_diff_pre, _ = collect_one_step(state, u_prob, X_sens)
        else:
            v_diff_pre, _ = collect_one_step(all_state, u_prob, X_sens)
        
        v_diff_new = np.expand_dims(v_diff_pre, axis=1) - v_diff
        R_obs = invX_est@v_diff_new
        
        if i > topo_change_step+1:
            v_diff_list.append(v_diff)
            dis_list.append(u_prob)
            r_list.append(R_obs)

        if i % 200 == 0:
            env.step_disturbance()
        if i == topo_change_step:
            if bus56:
                topo_change = [39, 1, 40]
            else:
                topo_change =[5, 2, 7]
            change_topo(env, topo_change)
                    
        last_action = action.clone()
        if bus56:
            all_state = np.copy(next_state_all)
        else:
            state = np.copy(next_state)
    
    return v_diff_list, dis_list, r_list

def data_collect(traj_num, controller_id, Q, R, path_base, topo_name, probing_nodes, probing_nodes_id, \
                               num_steps=1000, env_name = '13bus', use_true_X_at_start=True,\
                            n_control=3, base_v=4.16,\
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    bus56 = False
    if env_name == '13bus':
        injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        control_bus = np.array([2, 7, 9])
        n_bus = len(injection_bus)
        env = build_init_env()
    elif env_name == '56bus':
        injection_bus = np.array([18, 21, 30, 45, 53]) - 1 
        control_bus = injection_bus
        n_bus = 55
        env = build_init_56bus()
        bus56 = True
    else:
        assert False, 'wrong env name'
    n_control = len(control_bus)  
    v_diff_traj_list = []
    R_traj_list = []
    u_traj_list = []

    
    policy_list = load_policy(n_control, path_base, topo_name,  env)
    if use_true_X_at_start:
        if bus56:
            all_nodes = list(range(1,56))
            all_nodes_id = list(range(0,55))
            X_full,lines = get_true_topo(env, probing_nodes=all_nodes, probing_nodes_id = all_nodes_id,\
        base_v=12.0, use_linear=True, N=n_bus)
        else:
            X_full, lines = get_true_topo(env, probing_nodes, probing_nodes_id, base_v=base_v, use_linear=True, N=n_bus)
    else:
        X_full, lines = get_init_topo(env, policy_list, probing_nodes, probing_nodes_id, n_bus, controller_id)
    
    for run_id in tqdm(range(traj_num)):
        if env_name == '13bus':
            net = create_13bus()
            
            env = IEEE13bus(net, injection_bus, injection_bus, \
                    Q = Q, R = R)
        elif env_name == '56bus':
            pp_net = create_56bus()
            env = VoltageCtrl_nonlinear(pp_net, injection_bus, control_bus, \
                Q = Q, R = R)
            
        state, all_state = env.reset(seed = run_id)#original seed=13 #vector of R^12 
        torch.manual_seed(seed = run_id)
        '''
        Create the policy
        '''
        policy_list = load_policy(n_control, path_base, topo_name,  env)
        
        # Build the auxilary variables
        name_list = []
        for name, param in policy_list[0].named_parameters():
            name_list.append(name)

        last_action = torch.zeros((n_bus,1)).to(device)
        
        v_diff_list, dis_list, r_list = traj_rollout_no_update(controller_id, env, state, all_state, X_full, lines, probing_nodes_id,\
             policy_list, last_action,n_control,num_steps=num_steps, device=device, bus56=bus56)
        v_diff_traj_list.append(v_diff_list)
        u_traj_list.append(dis_list)
        R_traj_list.append(r_list)
    return v_diff_traj_list, u_traj_list, R_traj_list
