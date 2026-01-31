import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math

class IEEE13bus(gym.Env):
    def __init__(self, pp_net, injection_bus, control_bus, v0=1, vmax=1.05, vmin=0.95, all_bus=False,\
                  Q = np.eye(12), R = np.eye(12)):
        self.network =  pp_net
        self.obs_dim = 1
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.control_bus = control_bus
        self.agentnum = len(self.control_bus)
        if self.agentnum == 12:
            all_bus=True
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        self.all_bus = all_bus
        
        self.state = np.ones(self.agentnum, )
        self.init_state = np.ones(self.agentnum, )
        self.Q = Q
        self.R = R
    
    def step_Preward(self, action, p_action): 
        # state-transition with initial voltage deviation
        done = False 
        
        # The controlled PV nodes are simply modeled by static generators with Q control capability. Here bus 2, 7, 9 are controlled.
        reward_sep = np.zeros(self.agentnum, )
        if self.agentnum==3:
            self.network.sgen.at[1, 'q_mvar'] = action[0]
            self.network.sgen.at[6, 'q_mvar'] = action[1]
            self.network.sgen.at[8, 'q_mvar'] = action[2]
        else:
            for i in range(len(self.injection_bus)):
                self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        all_bus = [1,2,3,4,5,6,7,8,9,10,11,12]
        state_all = self.network.res_bus.iloc[all_bus].vm_pu.to_numpy()
        
        state_cost = np.expand_dims(np.square(state_all)-1,0)@self.Q@np.expand_dims(np.square(state_all)-1,1) 
        if self.agentnum> 3:
            action_cost = p_action.T @ self.R @ p_action
        else:
            action_cost = action.T@self.R@action
        reward =( -(state_cost + action_cost)).squeeze()
        
        if(np.min(self.state) > 0.96 and np.max(self.state)< 1.04):
            done = True
        return self.state, reward, reward_sep, done
    
    def step_load(self, action, p_action, load_p, load_q, pv_p): #state-transition with specific load
        """
        Step load function deploys the true load and generation profile in the real world for the simulation.

        Args:
            action (_type_): Reactive power injection, q_t
            p_action (_type_): Reactive power change, u_t
            load_p (_type_): Real world active power load
            load_q (_type_): Real world reactive power load
            pv_p (_type_): Real world PV generation power

        Returns:
            _type_: _description_
        """
        done = False 
        # Scaled to match the load level in the original 13-bus system
        self.network.load.at[0, 'p_mw'] = load_p*0.02
        self.network.load.at[0, 'q_mvar'] = load_q*0.02
        self.network.load.at[1, 'p_mw'] = load_p*0.03
        self.network.load.at[1, 'q_mvar'] = load_q*0.02
        self.network.load.at[2, 'p_mw'] = load_p*0.03
        self.network.load.at[2, 'q_mvar'] = load_q*0.02
        self.network.load.at[3, 'p_mw'] = load_p*0.02
        self.network.load.at[3, 'q_mvar'] = load_q*0.01
        self.network.load.at[4, 'p_mw'] = load_p*0.03
        self.network.load.at[4, 'q_mvar'] = load_q*0.02
        self.network.load.at[5, 'p_mw'] = load_p*0.02
        self.network.load.at[5, 'q_mvar'] = load_q*0.01
        self.network.load.at[6, 'p_mw'] = load_p*0.02
        self.network.load.at[6, 'q_mvar'] = load_q*0.01
        self.network.load.at[7, 'p_mw'] = load_p*0.03
        self.network.load.at[7, 'q_mvar'] = load_q*0.02
        self.network.load.at[8, 'p_mw'] = load_p*0.01
        self.network.load.at[8, 'q_mvar'] = load_q*0.01        
        

        # self.network.sgen.at[0, 'p_mw'] = pv_p*0.1
        self.network.sgen.at[1, 'p_mw'] = pv_p*0.15
        self.network.sgen.at[2, 'p_mw'] = pv_p*0.1
        self.network.sgen.at[3, 'p_mw'] = pv_p*0.01
        self.network.sgen.at[4, 'p_mw'] = 0.05*pv_p
        self.network.sgen.at[5, 'p_mw'] = 0.01*pv_p
        self.network.sgen.at[6, 'p_mw'] = pv_p*0.1
        self.network.sgen.at[7, 'p_mw'] = pv_p*0.02
        self.network.sgen.at[8, 'p_mw'] = pv_p*0.2
        self.network.sgen.at[9, 'p_mw'] = pv_p*0.01
        self.network.sgen.at[10, 'p_mw'] = pv_p*0.01
        self.network.sgen.at[11, 'p_mw'] = pv_p*0.01
        
        # The controlled PV nodes are simply modeled by static generators with Q control capability. Here bus 2, 7, 9 are controlled.
        if self.agentnum==3:
            self.network.sgen.at[1, 'q_mvar'] = action[0]
            self.network.sgen.at[6, 'q_mvar'] = action[1]
            self.network.sgen.at[8, 'q_mvar'] = action[2]
        else:
            for i in range(len(self.injection_bus)):
                self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        all_bus = [1,2,3,4,5,6,7,8,9,10,11,12]
        state_all = self.network.res_bus.iloc[all_bus].vm_pu.to_numpy()
        
        state_cost = np.expand_dims(np.square(state_all)-1,0)@self.Q@np.expand_dims(np.square(state_all)-1,1) 
        # get the cost h_t
        if self.agentnum> 3:
            action_cost = p_action.T @ self.R @ p_action
        else:
            action_cost = action.T@self.R@action
        reward =( -(state_cost + action_cost)).squeeze()
        
        if(np.min(self.state) > 0.96 and np.max(self.state)< 1.04):
            done = True
        
        return self.state, state_all, reward, done
    
    def step_disturbance(self):
        # Apply a step disturbance to a random bus
        random_bus_id = np.random.choice(np.arange(12))
        self.network.sgen.at[random_bus_id, 'p_mw'] = self.initial_p[random_bus_id]+np.random.uniform(-1, 1)*0.1
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return random_bus_id, self.state
    
    def reset(self, seed=1, scale=1): 
        """
        Reset the voltage by random load/generation profile.
        """
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = -0.5*np.random.uniform(1, 7)*scale
            self.network.sgen.at[1, 'p_mw'] = -0.8*np.random.uniform(1, 4)*scale
            self.network.sgen.at[4, 'p_mw'] = -0.3*np.random.uniform(1, 5)*scale
            if self.all_bus:
                for i in range(len(self.injection_bus)):
                    self.network.sgen.at[i, 'p_mw'] = -0.3*np.random.uniform(1, 2.5)*scale
        elif(senario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = np.random.uniform(0.5, 4)*scale
            self.network.sgen.at[1, 'p_mw'] = np.random.uniform(1, 4.51)*scale
            self.network.sgen.at[4, 'p_mw'] = np.random.uniform(0, 5)*scale

            # self.network.sgen.at[3, 'q_mvar'] = 0.3*np.random.uniform(0, 0.2)
            self.network.sgen.at[2, 'p_mw'] = 0.6*np.random.uniform(0, 2)*scale
            self.network.sgen.at[3, 'p_mw'] = 0.6*np.random.uniform(2, 3)*scale
            # self.network.sgen.at[5, 'q_mvar'] = 0.4*np.random.uniform(0, 10)
            self.network.sgen.at[6, 'p_mw'] = 0.7*np.random.uniform(0, 2)*scale
            
            self.network.sgen.at[10, 'p_mw'] = np.random.uniform(0.2, 3)*scale
            self.network.sgen.at[11, 'p_mw'] = np.random.uniform(0, 1.5)*scale
            #for all buses scheme
            if self.all_bus:
                self.network.sgen.at[7, 'p_mw'] = 0.5*np.random.uniform(1, 2)*scale
                self.network.sgen.at[8, 'p_mw'] = 0.2*np.random.uniform(1, 3)*scale
                self.network.sgen.at[5, 'p_mw'] = 0.2*np.random.uniform(2, 3)*scale
                self.network.sgen.at[9, 'p_mw'] = np.random.uniform(0.1, 0.5)*scale
        
        else: #mixture (this is used only during testing)
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = -2*np.random.uniform(2, 3)*scale
            self.network.sgen.at[1, 'p_mw'] = np.random.uniform(15, 35)*scale
            self.network.sgen.at[1, 'q_mvar'] = 0.1*self.network.sgen.at[1, 'p_mw']*scale
            self.network.sgen.at[2, 'p_mw'] = 0.2*np.random.uniform(2, 12)*scale
            
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        self.init_state = np.copy(self.state)
        all_bus = [1,2,3,4,5,6,7,8,9,10,11,12]
        self.initial_p = []
        for i in range(12):
            self.initial_p.append(self.network.sgen.at[i, 'p_mw'])
        return self.state, self.network.res_bus.iloc[all_bus].vm_pu.to_numpy()
    
    def reset0(self, seed=1): #reset voltage to nominal value
        
        self.network.load['p_mw'] = 0*self.load0_p
        self.network.load['q_mvar'] = 0*self.load0_q

        self.network.sgen['p_mw'] = 0*self.gen0_p
        self.network.sgen['q_mvar'] = 0*self.gen0_q
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        self.init_state = np.copy(self.state)
        all_bus = [1,2,3,4,5,6,7,8,9,10,11,12]
        self.initial_p = []
        for i in range(12):
            self.initial_p.append(self.network.sgen.at[i, 'p_mw'])
        return self.state, self.network.res_bus.iloc[all_bus].vm_pu.to_numpy()

def create_13bus(pp_model_pth='models/case_13.mat'):
    pp_net = pp.converter.from_mpc(pp_model_pth, casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 1, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 2, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 4, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 5, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 6, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 8, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 9, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 10, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 11, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 12, p_mw = 0, q_mvar=0)
    
    # # In the original IEEE 13 bus system, there is no load in bus 3, 7, 8. 
    # # Add the load to corresponding node for dimension alignment in RL training
    pp.create_load(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 8, p_mw = 0, q_mvar=0)

    return pp_net

if __name__ == "__main__":
    # run this file to test the environment
    # this code will result in a figure 'state_load.png' saved in the current folder, which presents the voltage trajectory under load variation without control action.
    # it may take around 1-2 minutes to run this code.
    pp_net = create_13bus()
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
    
    state, _ = env.reset0(seed = 18)
    
    from scipy.io import loadmat
    p_real = loadmat('models/aggr_p.mat')
    p_real = p_real['p']

    q_real = loadmat('models/aggr_q.mat')
    q_real = q_real['q']

    pv_p = loadmat('models/PV.mat')
    pv_p = np.resize(pv_p['actual_PV_profile'],(q_real.shape[0],1))
    state_traj = []
    
    for i in range(pv_p.shape[0]): #pv_p.shape[0]
        action = np.zeros((n_bus, 1))
        next_state, _, reward, done = env.step_load(action, action[controller_id], p_real[i], q_real[i], pv_p[i])
        state_traj.append(next_state)
    plt.plot(state_traj)
    plt.savefig('state_load.png')


