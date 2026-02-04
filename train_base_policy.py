import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import torch
import argparse
import os
import os.path as osp

from env.env_single_phase_13bus import IEEE13bus, create_13bus

from models import ValueNetwork, DDPG, ReplayBufferPI, SafePolicyNetwork_SetPoint

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

seed = 10
torch.manual_seed(seed)

"""
Create Agent list and replay buffer
"""
def get_args():
    parser = argparse.ArgumentParser(description="DDPG Experiment Parameters")
    parser.add_argument("--vlr", type=float, default=1e-4, help="Value network learning rate")
    parser.add_argument("--plr", type=float, default=1e-4, help="Policy network learning rate")
    parser.add_argument("--ph-num", type=int, default=1, help="Number of phases (ph_num)")
    parser.add_argument("--max-ac", type=float, default=0.5, help="Maximum action value") #for original topo, use a smaller noise can achieve better performance.
    parser.add_argument("--change-topo", action="store_true", help="Enable changing topology")
    parser.add_argument("--topo-name", type=str, default='topo1', help="Change topology name")
    return parser.parse_args()

# Parse arguments
args = get_args()

# Assign parsed arguments to variables
vlr = args.vlr
plr = args.plr
ph_num = args.ph_num
max_ac = args.max_ac
change_topo = args.change_topo

print('change topology: ', change_topo)
topo_name = args.topo_name
if change_topo == True and topo_name == 'topo1':
    print(topo_name, ', change two line.')
    pp_net = create_13bus()
    pp_net.line.loc[7, 'from_bus'] = 4
    pp_net.line.loc[8, 'from_bus'] = 4
elif change_topo == True and topo_name == 'topo2':
    print(topo_name, ', change three line.')
    pp_net = create_13bus()
    pp_net.line.loc[7, 'from_bus'] = 4
    pp_net.line.loc[8, 'from_bus'] = 4
    pp_net.line.loc[2, 'from_bus'] = 9 # line (1->3) change to line (9->3)
else:
    pp_net = create_13bus()
    topo_name = 'original_topo'
injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
control_bus = np.array([2, 7, 9])
control_id = control_bus-1

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

env = IEEE13bus(pp_net, injection_bus, control_bus, \
                Q = Q, R = R)


obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 64

type_name = 'single-phase'

agent_list = []
replay_buffer_list = []

for i in range(n_control):
    policy_net = SafePolicyNetwork_SetPoint(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    target_policy_net = SafePolicyNetwork_SetPoint(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    value_net = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    agent = DDPG(policy_net=policy_net, value_net=value_net,
                 target_policy_net=target_policy_net, target_value_net=target_value_net, value_lr=vlr, policy_lr=plr)
    
    replay_buffer = ReplayBufferPI(capacity=1000000)
    
    agent_list.append(agent)
    replay_buffer_list.append(replay_buffer)

FLAG = 1
# FLAG = 0 #load model, not implemented.
if (FLAG ==1):
    num_episodes = 1000 #1500

    # trajetory length each episode
    num_steps = 30  
    batch_size = 256

    rewards = []
    avg_reward_list = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed = episode)
        episode_reward = 0
        episode_reward_list = []
        last_action = np.zeros((n_control, ph_num)) 

        for step in range(num_steps):
            action = []
            action_p = []
            for i in range(n_control):
                # sample action according to the current policy and exploration noise
                action_agent = agent_list[i].policy_net.get_action(np.square(np.asarray([state[control_id[i]]]))) \
                    + np.random.normal(0, max_ac)/(episode+1) ** 0.3
                action_p.append(action_agent)                    
                action.append(action_agent)
            
            action = last_action - np.asarray(action)

            # execute action a_t and observe reward r_t and observe next state s_{t+1}
            next_state, reward, reward_sep, done = env.step_Preward(action, action_p)
            
            if(np.min(next_state)<0.75): #if voltage violation > 25%, episode ends.
                break
            else:
                for i in range(n_control): 
                    state_buffer = (np.square(np.asarray([state[control_id[i]]]))).reshape(ph_num,) 
                    action_buffer = action[i].reshape(ph_num,)
                    last_action_buffer = last_action[i].reshape(ph_num,)
                    next_state_buffer = (np.square(np.asarray([next_state[control_id[i]]]))).reshape(ph_num, )
                    
                    # store transition (s_t, a_t, r_t, s_{t+1}) in R
                    replay_buffer_list[i].push(state_buffer, action_buffer, last_action_buffer,
                                               reward, next_state_buffer, done) #_sep[i]

                    
                    if len(replay_buffer_list[i]) > batch_size:
                        agent_list[i].train_step(replay_buffer=replay_buffer_list[i], 
                                                batch_size=batch_size)

                state = np.copy(next_state)
                episode_reward += reward   

            last_action = np.copy(action)

        # rewards.append(episode_reward)
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-40:])
        if(episode%50==0):
            print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
        avg_reward_list.append(avg_reward)
    current_value_lr = agent_list[0].value_optimizer.param_groups[0]['lr']
    current_policy_lr = agent_list[0].policy_optimizer.param_groups[0]['lr']
    print(current_value_lr)
    print("########## Saving Checkpoints ##########")
    for i in range(n_control):
        path_base = f'checkpoints/{type_name}/13bus/NN_control'
        os.makedirs(path_base, exist_ok=True)
        if change_topo == True:
            pth_value = osp.join(path_base, f'{topo_name}_value_net_checkpoint_a{i}.pth')
            pth_policy = osp.join(path_base, f'{topo_name}_policy_net_checkpoint_a{i}.pth')
        else:
            pth_value = osp.join(path_base, f'original_topo_value_net_checkpoint_a{i}.pth')
            pth_policy = osp.join(path_base, f'original_topo_policy_net_checkpoint_a{i}.pth')
        torch.save(agent_list[i].value_net.state_dict(), pth_value)
        torch.save(agent_list[i].policy_net.state_dict(), pth_policy)
    print("########## Checkpoints Saved ##########")

else:
    raise ValueError("Model loading optition does not exist!")

fig, axs = plt.subplots(1, n_control, figsize=(15,3))
for i in range(n_control):
    # plot policy
    N = 50
    s_array = np.zeros(N,)
    
    a_array_baseline = np.zeros(N,)
    a_array = np.zeros((N,ph_num))
    
    for j in range(N):
        if ph_num ==1:
            state = np.array([0.8+0.01*j])
            s_array[j] = state
            action_baseline = (np.maximum(state-1.05, 0)-np.maximum(0.95-state, 0)).reshape((1,))
        else:
            state = np.resize(np.array([0.8+0.01*j]),(3))
            s_array[j] = state[0]        
            action_baseline = (np.maximum(state[0]-1.03, 0)-np.maximum(0.97-state[0], 0)).reshape((1,))
        
        action = agent_list[i].policy_net.get_action(np.square(np.asarray([state]))) 
        a_array_baseline[j] = -action_baseline[0]
        a_array[j] = -action
    axs[i].plot(s_array, 2*a_array_baseline, '-.', label = 'Linear')
    for k in range(ph_num):        
        axs[i].plot(s_array, a_array[:,k])
        axs[i].legend(loc='lower left')
os.makedirs('figures', exist_ok=True)  # Ensure directory exists

plt.tight_layout()  # Adjust layout to fit elements
plt.savefig(f'figures/controllers_{topo_name}.png', dpi=300, bbox_inches='tight')
plt.show()

## test policy
state, _ = env.reset()
episode_reward = 0
last_action = np.zeros((n_control,1))
action_list=[]
state_list =[]
reward_list = []
state_list.append(state)
for step in range(100):
    action = []
    for i in range(n_control):
        # sample action according to the current policy and exploration noise
        action_agent = agent_list[i].policy_net.get_action(np.square(np.asarray([state[controller_id[i]]])))
        action.append(action_agent)

    # PI policy    
    action = last_action - np.asarray(action)

    # execute action a_t and observe reward r_t and observe next state s_{t+1}
    next_state, reward, reward_sep, done = env.step_Preward(action, (action-last_action))
    reward_list.append(reward)
    action_list.append(action-last_action)
    state_list.append(next_state)
    last_action = np.copy(action)
    state = next_state
fig, axs = plt.subplots(1, n_control+1, figsize=(15,3))
for i in range(n_control):
    axs[i].plot(range(len(state_list)), np.array(state_list)[:,controller_id[i]], '-.', label = 'states')
    axs[i].legend(loc='lower left')
fig1, axs1 = plt.subplots(1, n_control+1, figsize=(15,3))
for i in range(n_control):
    axs1[i].plot(range(len(action_list)), np.array(action_list)[:,i], '-.', label = 'actions')
    axs1[i].legend(loc='lower left')
axs[n_control].plot(range(len(reward_list)),reward_list)

plt.savefig('figures/performance.png', dpi=300, bbox_inches='tight')
plt.show()
