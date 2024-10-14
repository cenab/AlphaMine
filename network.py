# network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AlphaZeroNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * input_shape[1] * input_shape[2], 256)
        self.bn_fc = nn.BatchNorm1d(256)

        # Policy head
        self.policy_head = nn.Linear(256, num_actions)

        # Value head
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn_fc(self.fc1(x)))

        # Policy and value outputs
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value
