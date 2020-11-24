import os,time
from config import hp

import numpy as np
import itertools

import torch.optim as optim
from collections import namedtuple
from tensorboardX import SummaryWriter
from itertools import count
from PPO_NN import Actor,Critic,CNN_for_atari
import torch
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from utils import atari_state_preprocess

class PPO_agent():
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a subclass of this
    grid2op.Agent.BaseAgent.
    """   
    def __init__(self, env):
        """Initialize a new agent."""
        self.env=env     
        self.observation_size=256
        self.action_size=env.action_space.n   
        print(self.observation_size,self.action_size)
               
        self.actor_net = Actor(self.observation_size,self.action_size)
        self.critic_net = Critic(self.observation_size)
        self.cnn_net=CNN_for_atari()

        self.Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])        
        self.buffer = []   
        self.hp=hp#########hyper-parameters-dict
        self.writer = SummaryWriter('./logs')
        self.optimizer = optim.Adam(itertools.chain(self.actor_net.parameters(),self.cnn_net.parameters(),self.critic_net.parameters()), 1e-3)

        if not os.path.exists('./model'):
            os.makedirs('./model')

    def rawstate_to_state(self,rawstate):####np.array(210,160,3)
         state=atari_state_preprocess(rawstate)###(84,84)
         state=torch.Tensor(state).unsqueeze(0).unsqueeze(0)
         state=self.cnn_net(state)      
         return state.detach().numpy()#######np.array(256)
     
    def select_action(self, state):########(x,0),array
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


                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.hp['training_step'])
                             
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.hp['training_step'])
               
                total_loss=action_loss+value_loss
                self.writer.add_scalar('loss/total_loss', total_loss, global_step=self.hp['training_step'])
                
                #update actor,critic,cnn network in the same time
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.hp['max_grad_norm'])
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.hp['max_grad_norm'])                
                self.optimizer.step()
                self.hp['training_step'] += 1

        del self.buffer[:] # clear experience
        
    def train_agent(self):        
        for i_epoch in range(self.hp['step']):
            print(i_epoch)
            rawstate=self.env.reset()#####np.array(210,160,3) 
            state=self.rawstate_to_state(rawstate)####$(s),torch.Tensor(256)
            
            for t in count():
                
                a, action_prob = self.select_action(state)         
                next_rawstate, reward, done, info = self.env.step(a)
                
                next_state=self.rawstate_to_state(next_rawstate)####$(s),torch.Tensor(256)
                
                trans = self.Transition(state, a, action_prob, reward, next_state)
                self.store_transition(trans)
                state = next_state
    
                if done or t>=9999:
                    if len(self.buffer) >= self.hp['batch_size']:
                        self.update(i_epoch)
                    print('#################t:',t)
                    self.writer.add_scalar('livestep', t, global_step=i_epoch)
                    break
            if i_epoch%50==0:
                self.save_param(i_epoch)
                print('save model!')
                
    def test_agent(self,render,i_epoch):
        self.load_param(i_epoch,actor_model=self.actor_net,critic_model=self.critic_net)
        for i_epoch in range(self.hp['test_step']):
            print(i_epoch)
            rawstate=self.env.reset()
            
            state=self.rawstate_to_state(rawstate)####$(s),torch.Tensor(256)
            
            for t in count():
                a, action_prob = self.select_action(state) 
                # print(a,action_prob)
                next_rawstate, reward, done, info = self.env.step(a)
                
                next_state=self.rawstate_to_state(next_rawstate)####$(s),torch.Tensor(256)                
                
                if render==True:
                    self.env.render()
                state = next_state
                if done or t>=9999:
                    print('#################t:',t)
                    break        




