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
     
    env = gym.make('MountainCar-v0').unwrapped####在这个游戏中，小车开始很难爬上高峰，所以‘存活’步数很高
                                              ####，后期通过学习，很快就完成目标，所以‘存活’数变少，这是好现象！
    agent=PPO_agent(env)
    agent.train_agent()

