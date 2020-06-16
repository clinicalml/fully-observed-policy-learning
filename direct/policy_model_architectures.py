import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear model
class PolicyNetLinear(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PolicyNetLinear, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_outputs)


    def forward(self, x):
        x = self.fc1(x)
        return torch.softmax(x, dim=-1)


# MLP with single hidden layer + ReLU activation
class PolicyNetSingleLayer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PolicyNetSingleLayer, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, num_outputs)

    def forward(self, x, avoid_last=False):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
