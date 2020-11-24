import torch
import torch.nn as nn
import torch.nn.functional as F
from config import hp

torch.manual_seed(hp['seed'])

class CNN_for_atari(torch.nn.Module):
    def __init__(self):
        super(CNN_for_atari,self).__init__()#####input=(batch,1,84,84)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=8,stride=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),#########(batch, 16, 20, 20) 
            
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            )########(batch, 32, 9, 9)
            
    def forward(self, x):
        x = self.net(x)
        x=x.flatten()
        x=nn.Linear(2592,256)(x)
        x = F.relu(x)
        return x  ##output=(256,)
        
class Actor(nn.Module):
    def __init__(self,num_state,num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self,num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        # self.fc2 = nn.Linear(512, 256)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x=F.relu(self.fc2(x))
        value = self.state_value(x)
        return value