from torch import nn
import random
import numpy as np

FRAME_WIDTH = 160
FRAME_HEIGHT = 210
INPUT_WIDTH = 84
INPUT_HEIGHT= 84
ACTION = 4
STATE_LENGTH = 4

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.action_space = ACTION

        self.conv1 = nn.Conv2d(STATE_LENGTH, 32, 8, stride=(4, 4))
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=(2, 2))
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=(1, 1))
        nn.init.kaiming_normal_(self.conv3.weight)
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2_adv = nn.Linear(512, ACTION)
        nn.init.kaiming_normal_(self.fc2_adv.weight)
        self.fc2_val = nn.Linear(512, 1)  # 価値V側
        nn.init.kaiming_normal_(self.fc2_val.weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        adv = self.fc2_adv(x)
        val = self.fc2_val(x).expand(-1, adv.size(1))

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output


    def get_action(self, state, epsilon):
        
        self.eval()
        if random.random() > epsilon:
            qvalue = self(state)
            action = qvalue.max(1)[1].view(1, 1).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action