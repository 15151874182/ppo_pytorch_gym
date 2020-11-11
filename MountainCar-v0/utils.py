import numpy as np
import torch
# gym.envs.registry.all

def simple_normalization(state):
    state=np.where(state>1000,state/10000,state)
    state=np.where(state>100,state/1000,state)
    state=np.where(state>10,state/100,state)
    state=np.where(state>=1,state/10,state)
    state=np.where((state>0) & (state<0.1),0,state)
    state=np.where(state<-100,state/1000,state)
    state=np.where(state<-10,state/100,state)
    state=np.where(state<=-1,state/10,state) 
    state=np.where((state<0) & (state<-0.1),0,state)
    # np.clip(state,-1,1)
    return state

def check_illegal(obs,action):
    obs_simulate, reward_simulate, done_simulate, info = obs.simulate(action)

    if info["is_illegal"] or info["is_ambiguous"] or info["is_illegal_reco"]:
        print("illegal!!!")
        return True
    else:
        return False
    