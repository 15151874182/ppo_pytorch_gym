import os,time
from config import hp

import numpy as np
import itertools

import torch.optim as optim
from collections import namedtuple
from tensorboardX import SummaryWriter
from itertools import count
from PPO_NN import Actor,Critic
import torch
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

class PPO_agent():
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """
    def __init__(self, env):
        """Initialize a new agent."""
        self.env=env
        self.observation_size=env.observation_space.shape[0]
        self.action_size=env.action_space.n
        
        self.Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
        self.actor_net = Actor(self.observation_size,self.action_size)
        self.critic_net = Critic(self.observation_size)
        self.buffer = []   
        self.hp=hp#########hyper-parameters-dict
        self.writer = SummaryWriter('./logs')
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)   
        if not os.path.exists('./model'):
            os.makedirs('./model')
            

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self,i_epoch):
        torch.save(self.actor_net.state_dict(), './model/actor_net_{}.pt'.format(i_epoch))
        torch.save(self.critic_net.state_dict(), './model/critic_net_{}.pt'.format(i_epoch))

    def load_param(self,i_epoch,actor_model,critic_model):
        actor_model.load_state_dict(torch.load('./model/actor_net_{}.pt'.format(i_epoch)))
        critic_model.load_state_dict(torch.load('./model/critic_net_{}.pt'.format(i_epoch)))

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.hp['counter'] += 1


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.hp['gamma'] * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.hp['ppo_update_time']):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.hp['batch_size'], False):
                # if self.training_step % 1000 ==0:
                #     print('I_ep {} ，train {} times'.format(i_ep,self.hp['training_step']))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)#####torch.Size([32, 1])
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.hp['clip_param'], 1 + self.hp['clip_param']) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.hp['training_step'])
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.hp['max_grad_norm'])
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.hp['training_step'])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.hp['max_grad_norm'])
                self.critic_net_optimizer.step()
                self.hp['training_step'] += 1

        del self.buffer[:] # clear experience
        
    def train_agent(self):        
        for i_epoch in range(self.hp['step']):
            print(i_epoch)
            state=self.env.reset()
            for t in count():
                a, action_prob = self.select_action(state)         
                next_state, reward, done, info = self.env.step(a)
                trans = self.Transition(state, a, action_prob, reward, next_state)
                self.store_transition(trans)
                state = next_state
    
                if done or t>=200:
                    if len(self.buffer) >= self.hp['batch_size']:
                        self.update(i_epoch)
                    print('#################t:',t)
                    self.writer.add_scalar('livestep', t, global_step=i_epoch)
                    break
            if i_epoch%100==0:
                self.save_param(i_epoch)
                print('save model!')
    def test_agent(self):
        self.load_param(i_epoch=900,actor_model=self.actor_net,critic_model=self.critic_net)
        for i_epoch in range(self.hp['test_step']):
            print(i_epoch)
            state=self.env.reset()
            for t in count():
                a, action_prob = self.select_action(state) 
                # print(a,action_prob)
                next_state, reward, done, info = self.env.step(a)
                state = next_state
                if done or t>=200:
                    print('#################t:',t)
                    break        




