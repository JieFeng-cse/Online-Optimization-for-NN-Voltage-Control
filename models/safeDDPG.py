import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
from torch.autograd.functional import jacobian

# DPPG class
class DDPG:
    def __init__(self, policy_net, value_net,
                 target_policy_net, target_value_net,
                 value_lr=2e-4,
                 policy_lr=1e-4,
                 gamma_scheduler=0.95,
                 step_size=500):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy_net = policy_net
        self.value_net = value_net
        self.target_policy_net = target_policy_net
        self.target_value_net = target_value_net
        
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        
        self.value_optimizer = optim.Adam(value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()

        # LR schedulers
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer,
                                                         step_size=step_size,
                                                         gamma=gamma_scheduler)
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer,
                                                          step_size=step_size,
                                                          gamma=gamma_scheduler)

    def train_step(self, replay_buffer, batch_size,
                   gamma=0.99,
                   soft_tau=1e-2):

        state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        next_action = action-self.target_policy_net(next_state)    
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + gamma*(1.0-done)*target_value
        
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
 
        self.value_optimizer.step()
        
        policy_loss = self.value_net(state, last_action-self.policy_net(state)) 
        policy_loss =  -policy_loss.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_scheduler.step()
        self.policy_scheduler.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data*soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    
# monotone policy network with set point
class SafePolicyNetwork_SetPoint(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, scale = 0.15, init_w=3e-3):
        super(SafePolicyNetwork_SetPoint, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        self.hidden_dim = hidden_dim
        self.scale = scale
        
        #define weight and bias recover matrix
        self.w_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.w_recover = -torch.triu(self.w_recover, diagonal=0)\
        +torch.triu(self.w_recover, diagonal=2)+2*torch.eye(self.hidden_dim)
        self.w_recover=self.w_recover.to(self.device)
        
        self.b_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.b_recover = torch.triu(self.b_recover, diagonal=0)-torch.eye(self.hidden_dim)
        self.b_recover = self.b_recover.to(self.device)
        
        self.select_w = torch.ones(1, self.hidden_dim).to(self.device)
        self.select_wneg = -torch.ones(1, self.hidden_dim).to(self.device)

        # voltage setpoint
        theta2 = torch.zeros(1, requires_grad=True).to(self.device)
        self.theta2 = torch.nn.Parameter(theta2)
        
        # initialization
        self.b = torch.rand(self.hidden_dim)
        self.b = (self.b/torch.sum(self.b))*scale
        self.b = torch.nn.Parameter(self.b, requires_grad=True)
        
        self.c = torch.rand(self.hidden_dim)
        self.c = (self.c/torch.sum(self.c))*scale
        self.c = torch.nn.Parameter(self.c, requires_grad=True)
        
        self.q = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        self.z = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        
    def forward(self, state):
        q_sq = torch.square(self.q)
        z_sq = torch.square(self.z)

        q_sq = torch.clamp(q_sq, min=1e-3)
        z_sq = torch.clamp(z_sq, min=1e-3)
        self.w_plus=torch.matmul(q_sq, self.w_recover)
        
        self.w_minus=torch.matmul(-z_sq, self.w_recover)
        
        b = self.b.data
        b = b.clamp(min=0)
        b = self.scale*b/torch.norm(b, 1)
        self.b.data = b
        
        c = self.c.data
        c = c.clamp(min=0)
        c = self.scale*c/torch.norm(c, 1)
        self.c.data = c

        set_point = 1+0.08*F.tanh(self.theta2)
        
        self.b_plus=torch.matmul(-self.b, self.b_recover) - set_point
        self.b_minus=torch.matmul(-self.c, self.b_recover) + set_point
        
        self.nonlinear_plus = torch.matmul(F.relu(torch.matmul(state, self.select_w)
                                                  + self.b_plus.view(1, self.hidden_dim)),
                                           torch.transpose(self.w_plus, 0, 1))
        
        self.nonlinear_minus = torch.matmul(F.relu(torch.matmul(state, self.select_wneg)
                                                   + self.b_minus.view(1, self.hidden_dim)),
                                            torch.transpose(self.w_minus, 0, 1))
        
        x = (self.nonlinear_plus+self.nonlinear_minus) 
        
        return x
    
    def get_jocobian_z(self, state):
        """
        consider the controller as phi(v-b), here we calculate the jocobian of phi
        """
        state = (state).requires_grad_(True)
        jac = jacobian(self.forward, state).squeeze()
        return jac

    def get_param_grad(self, state):
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state).sum()
        action.backward()
        grads = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
            else:
                grads[name] = None
        self.zero_grad(set_to_none=True)
        return grads

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0] 


# value network
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ReplayBufferPI:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, last_action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, last_action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

