import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

class ImitationLearningModel(nn.Module):
    def __init__(self):
        super(ImitationLearningModel, self).__init__()
        self.fc1 = nn.Linear(3, 256)  
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 7)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x