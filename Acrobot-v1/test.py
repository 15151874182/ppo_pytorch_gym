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
     
    env = gym.make('Acrobot-v1').unwrapped#####unwrapped使done的步数不受限制
    agent=PPO_agent(env)
    print('obs_size:',agent.observation_size)
    print('action_size:',agent.action_size)    
    agent.test_agent(render=True,i_epoch=100)

