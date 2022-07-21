from pickle import NONE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorNetwork(nn.Module):

    def __init__(self, input_size ,action_space, lr=0.00003):
        super(ActorNetwork, self).__init__()

        self.action_space = action_space

        self.dense1 = nn.Linear(input_size, 64)
        nn.init.orthogonal_(self.dense1.weight)

        self.dense2 = nn.Linear(64, 64)
        nn.init.orthogonal_(self.dense2.weight)

        self.pi_mean = nn.Linear(64, self.action_space)
        nn.init.orthogonal_(self.pi_mean.weight)

        self.pi_sigma = nn.Linear(64, self.action_space)
        nn.init.orthogonal_(self.pi_sigma.weight)

        self.lr=lr
        
        #self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        #self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        
        x = F.tanh(self.dense1(x))
        x = F.tanh(self.dense2(x))

        mean = F.tanh(self.pi_mean(x)) * 2
        stdev = F.softplus(self.pi_sigma(x)) + 0.3
        
        return mean, stdev

    def sample_action(self, states):

        mean, sigma = self(states)
        dist = Normal(loc=mean, scale=sigma)
        sampled_action = dist.sample()

        return sampled_action.detach().numpy().reshape(-1, 1)

class CriticNetwork(nn.Module):

    def __init__(self,input_size, lr=0.0001):

        super(CriticNetwork, self).__init__()

        self.dense1 = nn.Linear(input_size, 64)
        self.dense2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        
        self.lr=lr

    def forward(self, x):

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        out = self.out(x)

        return out

class CriticNetwork2(nn.Module):

    def __init__(self,input_size, lr=0.0001):

        super(CriticNetwork2, self).__init__()

        self.dense1 = nn.Linear(input_size + 1, 64)
        self.dense2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        
        self.lr=lr

    def forward(self, x, a):

        x = torch.cat([x, a], dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        out = self.out(x)

        return out

class ActorNetwork2(nn.Module):

    ACTION_RANGE = 2

    def __init__(self, input_size ,action_space, lr=0.00003):
        super(ActorNetwork2, self).__init__()

        self.action_space = action_space

        self.dense1 = nn.Linear(input_size, 64)
        nn.init.orthogonal_(self.dense1.weight)

        self.dense2 = nn.Linear(64, 64)
        nn.init.orthogonal_(self.dense2.weight)
        
        self.out = nn.Linear(64, 1)
        nn.init.orthogonal_(self.out.weight)

        self.lr=lr
        
        #self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        #self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        
        x = F.tanh(self.dense1(x))
        x = F.tanh(self.dense2(x))
        action = self.out(x)

        action = action * self.ACTION_RANGE #action range

        return action

    def sample_action(self, states, noise=None):

        action = self(states).detach().numpy().reshape(-1,1)

        if noise:
            action += np.random.normal(0, noise*self.ACTION_RANGE, size=self.action_space)
            action = np.clip(action, -self.ACTION_RANGE, self.ACTION_RANGE)

        return action

if __name__ == "__main__":

    policy = ActorNetwork(action_space=1)
    s = np.array([[1, 2, 3, 4]])
    out = policy(s)
    a = policy.sample_action(s)
    print(a)