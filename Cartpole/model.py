from torch import nn
import random
import numpy as np

ACTION = 2

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.action_space = ACTION

        self.fc1 = nn.Linear(4, 64)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 64)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(64, self.action_space)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def get_action(self, state, epsilon):
        
        self.eval()
        if random.random() > epsilon:
            qvalue = self(state)
            action = qvalue.max(1)[1].view(1, 1).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
