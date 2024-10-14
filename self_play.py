# self_play.py

import numpy as np
from mcts import MCTSNode, mcts_search
from utils import process_state

def self_play(env, network):
    state = env.reset()
    done = False
    trajectory = []

    while not done:
        # Process the state
        processed_state = process_state(state)

        root = MCTSNode(processed_state)
        action = mcts_search(root, network, num_simulations=50, env=env)

        # Take the action in the environment
        next_state, reward, done, info = env.step(env.action_space.noop())
        state = next_state

        # Store the transition
        trajectory.append((processed_state, action, reward))

    # Compute value targets from the trajectory
    total_reward = sum([x[2] for x in trajectory])

    training_data = []
    for state, action, _ in trajectory:
        policy_target = np.zeros(env.action_space.n)
        policy_target[action] = 1  # The action taken
        value_target = total_reward  # Simplified value target
        training_data.append((state, policy_target, value_target))

    return training_data
