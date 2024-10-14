# utils.py

import numpy as np

def process_state(state):
    """
    Process the raw state from the environment into the format expected by the neural network.
    """
    # Example: Extract RGB observation and resize
    rgb = state['pov']  # Assuming 'pov' key contains the image
    rgb = rgb.transpose((2, 0, 1))  # Convert to (C, H, W)
    rgb = rgb / 255.0  # Normalize pixel values
    return rgb

def simulate_action(env, state, action):
    """
    Simulate taking an action from a given state.
    This is a placeholder function; in practice, you need a way to predict the next state.
    """
    # Since we can't step the environment without changing its state,
    # you might need to create a copy or use a model to predict the next state.
    # For simplicity, we'll return None here.
    return None
