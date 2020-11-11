####################################################
import os
import torch.autograd as autograd
from config import hp
from PPO_agent import PPO_agent
import gym

####################################################
if hp['USE_CUDA']:
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()
else:
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
####################################################    

if __name__ == '__main__':
     
    env = gym.make('Acrobot-v1').unwrappe
                                             
    agent=PPO_agent(env)
    agent.train_agent()

