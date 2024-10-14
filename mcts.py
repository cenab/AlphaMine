# mcts.py

import math
import numpy as np
import torch
from collections import defaultdict
from utils import simulate_action

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

def mcts_search(root, network, num_simulations, env):
    for _ in range(num_simulations):
        node = root
        path = []

        # Selection
        while node.children:
            total_visits = sum(child.visits for child in node.children.values())
            max_ucb = max(
                node.children.items(),
                key=lambda item: ucb_score(item[1], total_visits)
            )
            node = max_ucb[1]
            path.append(node)

        # Expansion
        state_tensor = torch.tensor(node.state).unsqueeze(0).float()
        policy_logits, value = network(state_tensor)
        policy = policy_logits.exp().detach().numpy()[0]

        # Normalize the policy
        policy = policy / np.sum(policy)

        # Add children nodes
        for action, prob in enumerate(policy):
            next_state = simulate_action(env, node.state, action)
            if next_state is not None:
                child_node = MCTSNode(next_state, parent=node)
                child_node.prior = prob
                node.children[action] = child_node

        # Backpropagation
        for node in reversed(path):
            node.visits += 1
            node.value += value.item()

    # Choose the action with the highest visit count
    best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_action

def ucb_score(node, total_visits, c_puct=1.0):
    prior_score = c_puct * node.prior * math.sqrt(total_visits) / (1 + node.visits)
    value_score = node.value / (1 + node.visits)
    return value_score + prior_score
