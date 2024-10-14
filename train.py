# train.py

import torch
import torch.nn.functional as F
import torch.optim as optim

def train_network(network, training_data):
    network.train()
    optimizer = optim.Adam(network.parameters(), lr=1e-4)

    for state, policy_target, value_target in training_data:
        optimizer.zero_grad()

        state_tensor = torch.tensor(state).unsqueeze(0).float()
        policy_target_tensor = torch.tensor(policy_target).unsqueeze(0)
        value_target_tensor = torch.tensor([[value_target]]).float()

        policy_pred, value_pred = network(state_tensor)

        # Compute losses
        value_loss = F.mse_loss(value_pred, value_target_tensor)
        policy_loss = -torch.sum(policy_target_tensor * policy_pred)
        loss = value_loss + policy_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
