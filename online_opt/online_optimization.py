from numpy import linalg as LA
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def online_update(policy_list, y_q_theta_1, y_q_theta_2, state, state_hat, action, controller_id, Q, R, H_s, n_control,\
                  Phi_1, Phi_2, lr=0.1, lambda_y=1.0, flag_grad=1,\
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    h_q_grad = 2 * action.T @ R + 2 * (state_hat-1).T @ Q @ H_s
    
    # recursively update y_qs
    q_t_plus_wrt_q_t = torch.eye(state.shape[-1]).to(device) - Phi_1@torch.diag(torch.tensor([policy_list[id].slope.data for id in range(n_control)])).to(device)@Phi_2@H_s
    y_q_theta_1 = lambda_y*q_t_plus_wrt_q_t@y_q_theta_1 + Phi_1@torch.diag(torch.tensor([1+0.08*F.tanh(policy_list[id].theta3.data) - state_hat[controller_id[id]] for id in range(n_control)])).to(device) 
    # 1.04*1.04 = 1.0816 is the upper bound of the setpoint adjustment
    
    y_q_theta_2 = lambda_y*q_t_plus_wrt_q_t@y_q_theta_2 + Phi_1@torch.diag(torch.tensor([policy_list[id].slope.data for id in range(n_control)])).to(device)*0.08@\
        torch.diag(torch.tensor([1-0.08*torch.square(F.tanh(policy_list[id].theta3.data)) for id in range(n_control)])).to(device)

    # calculate the gradite
    G_t_theta_1 = h_q_grad @ y_q_theta_1
    G_t_theta_2 = h_q_grad @ y_q_theta_2

    for id in range(n_control):
        for name, param in policy_list[id].named_parameters():
            if param.requires_grad:
                if name == 'theta3':
                    param.data = param.data - lr*G_t_theta_2[0,id]*flag_grad
                if name == 'slope':
                    param.data = param.data - lr*G_t_theta_1[0,id]*flag_grad
                    param.data = torch.clamp(param.data, 0, 7)
    return y_q_theta_1, y_q_theta_2, policy_list

def online_update_Xp(policy_list, y_q_theta_1, y_q_theta_2, state, state_hat, action, controller_id, Q, R, X_P, n_control,\
                  Phi_1, Phi_2, lr=0.1, lambda_y=1.0, flag_grad=1,\
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    #m bus with probing, n bus in total has observation (or estimation)
    H_s = torch.FloatTensor(X_P).to(device)
    C = np.asarray([1.2,0.3,0.6])*0.1
    coef_c = 0.02
    R = torch.Tensor(np.diag(C)*coef_c).to(device)
    h_q_grad = 2 * action.T @ R + 2 * (state_hat-1).T @ Q @ H_s # shape: 1*m, R: m*m, action:m*1
    
    # recursively update y_qs
    q_t_plus_wrt_q_t = torch.eye(action.shape[0]).to(device) - torch.diag(torch.tensor([policy_list[id].slope.data for id in range(n_control)])).to(device)@Phi_2@H_s # shape: m*m
    y_q_theta_1 = lambda_y*q_t_plus_wrt_q_t@y_q_theta_1 + torch.diag(torch.tensor([1+0.08*F.tanh(policy_list[id].theta3.data) - state_hat[controller_id[id]] for id in range(n_control)])).to(device)
    
    y_q_theta_2 = lambda_y*q_t_plus_wrt_q_t@y_q_theta_2 + torch.diag(torch.tensor([policy_list[id].slope.data for id in range(n_control)])).to(device)*0.08@\
        torch.diag(torch.tensor([1-0.08*torch.square(F.tanh(policy_list[id].theta3.data)) for id in range(n_control)])).to(device)

    # calculate the gradite
    G_t_theta_1 = h_q_grad @ y_q_theta_1
    G_t_theta_2 = h_q_grad @ y_q_theta_2

    for id in range(n_control):
        for name, param in policy_list[id].named_parameters():
            if param.requires_grad:
                if name == 'theta3':
                    param.data = param.data - lr*G_t_theta_2[0,id]*flag_grad
                    # param.data = torch.clamp(param.data, 0.95*0.95, 1.05*1.05)
                if name == 'slope':
                    param.data = param.data - lr*G_t_theta_1[0,id]*flag_grad
                    param.data = torch.clamp(param.data, 0, 7)
    return y_q_theta_1, y_q_theta_2, policy_list

def online_update_Xp_NN(policy_list, y_q_theta_dict, G_t_theta_dict, state_hat, action, name_list, Q, R,\
                  controller_id, H_s, n_control,Phi_2, lr=0.1, lambda_y=1.0, flag_grad=1,\
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """_summary_

    Args:
        policy_list (_type_): _description_
        y_q_theta_dict (_type_): dict storing the auxilary variable for all parameters, key is the parameter name, for each parameter, 
            the size is m*ms, s is the size of the parameters. Note, it is a concatenation of m local controllers.
        state_hat (_type_): _description_
        action (_type_): _description_
        controller_id (_type_): _description_
        Q (_type_): _description_
        R (_type_): _description_
        X_P (_type_): _description_
        n_control (_type_): _description_
        Phi_2 (_type_): _description_
        lr (float, optional): _description_. Defaults to 0.1.
        lambda_y (float, optional): _description_. Defaults to 1.0.
        flag_grad (int, optional): _description_. Defaults to 1.
    """
    h_q_grad = 2 * action.T @ R + 2 * (state_hat-1).T @ Q @ H_s # shape: 1*m, R: m*m, action:m*1
    
    diag_jacob = torch.diag(torch.tensor([policy_list[id].get_jocobian_z(state_hat[controller_id[id]].unsqueeze(0)) for id in range(n_control)])).to(device)
    q_t_plus_wrt_q_t = torch.eye(action.shape[0]).to(device) - diag_jacob @ Phi_2 @ H_s # m*m This has eigenvalues smaller than 1
    q_t_plus_wrt_params_list = [policy_list[id].get_param_grad(state_hat[controller_id[id]].unsqueeze(0)) for id in range(n_control)]
    
    q_t_plus_wrt_params_dict = {}
    for name in name_list:
        q_t_plus_wrt_params_name_list =[]
        for id in range(n_control):
            tmp = -q_t_plus_wrt_params_list[id][name]
            if len(tmp.shape) == 1:
                tmp = tmp.unsqueeze(0)
            tmp_padded = torch.zeros((tmp.shape[0],tmp.shape[1]*n_control)).to(device)
            tmp_padded[:,id*tmp.shape[1]:(id+1)*tmp.shape[1]] = tmp
            q_t_plus_wrt_params_name_list.append(tmp_padded)
        q_t_plus_wrt_params_dict[name] = torch.cat(q_t_plus_wrt_params_name_list, dim = 0)
        
        y_q_theta_dict[name] = lambda_y*q_t_plus_wrt_q_t@y_q_theta_dict[name] + q_t_plus_wrt_params_dict[name]
        # print(q_t_plus_wrt_params_dict['theta2'])
        G_t_theta_dict[name] = h_q_grad @ y_q_theta_dict[name]
            
    
    for id in range(n_control):
        for name, param in policy_list[id].named_parameters():
            if param.requires_grad:
                if len(param.shape) == 1:
                    param.data = param.data - lr*G_t_theta_dict[name][0,id*param.shape[0]:(id+1)*param.shape[0]]*flag_grad 
                elif len(param.shape) == 2:
                    assert (param.data).shape == (G_t_theta_dict[name][0,id*param.shape[1]:(id+1)*param.shape[1]].unsqueeze(0)).shape
                    param.data = param.data - lr*G_t_theta_dict[name][0,id*param.shape[1]:(id+1)*param.shape[1]].unsqueeze(0)*flag_grad 
                else:
                    assert False, "Error: The shape of the calculated gradient does not match the parameters." 
    return y_q_theta_dict, G_t_theta_dict, policy_list
                