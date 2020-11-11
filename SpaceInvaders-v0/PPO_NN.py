import torch
import torch.nn as nn
import torch.nn.functional as F
from config import hp

torch.manual_seed(hp['seed'])

class CNN_for_atari(torch.nn.Module):
    def __init__(self):
        super(CNN_for_atari,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),             
        )
    def forward(self, x):
        x = self.net(x)
        return x.reshape(-1)    
        
class Actor(nn.Module):
    def __init__(self,num_state,num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        # self.fc2 = nn.Linear(512, 256)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x=F.relu(self.fc2(x))
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