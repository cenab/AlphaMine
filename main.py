# main.py

import torch
from network import AlphaZeroNet
from self_play import self_play
from train import train_network
import minerl
import gym

def main():
    # Initialize the MineRL environment
    env = gym.make('MineRLTreechop-v0')

    # Define input shape and number of actions
    input_shape = (3, 64, 64)  # Example input shape (channels, height, width)
    num_actions = env.action_space.n

    # Initialize the neural network
    network = AlphaZeroNet(input_shape, num_actions)

    # Training iterations
    for iteration in range(1000):
        print(f"Starting iteration {iteration}")

        # Self-play to generate training data
        training_data = []
        for _ in range(10):  # Number of self-play games per iteration
            game_data = self_play(env, network)
            training_data.extend(game_data)

        # Train the network
        train_network(network, training_data)

        # Optionally, save the model
        torch.save(network.state_dict(), f"model_iteration_{iteration}.pt")

        # Evaluate the network's performance
        print(f"Iteration {iteration} completed.")

if __name__ == "__main__":
    main()
