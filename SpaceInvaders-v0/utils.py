import numpy as np
from torch.nn.functional import interpolate

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
    
    
def rgb2gray(array):####(210,160,3)
    from PIL import Image
    img=Image.fromarray(array)
    img = img.convert('L') 
    img=np.array(img)
    return img########(210,,160)
def down_sampling(array):
    import torch
    tensor=torch.Tensor(array).unsqueeze(0).unsqueeze(0)
    tensor= interpolate(tensor,size=[110,84])
    array=tensor.squeeze(0).squeeze(0).numpy()
    array=array[-84:,:]
    return array

def atari_state_preprocess(state):##########np.array(210,160,3)
    state=rgb2gray(state)#####1.RGB转GRAY
    state=down_sampling(state)####2.down_sampling
    return state#######np.array(84,84)
    