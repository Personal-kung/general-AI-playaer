import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaNet(nn.Module):
    """
    A unified Policy-Value Network (AlphaZero style) that adapts 
    to any board size (rows x cols) and action space.
    """
    def __init__(self, input_shape, action_size):
        super(AlphaNet, self).__init__()
        # input_shape: (channels, rows, cols)
        _, self.rows, self.cols = input_shape
        
        # Shared Feature Extractor (Convolutional Backbone)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Flattened size for fully connected layers
        self.flatten_size = 64 * self.rows * self.cols

        # Policy Head: Returns Log-Probabilities over all legal moves
        self.policy_head = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.LogSoftmax(dim=1)
        )

        # Value Head: Returns scalar in range [-1, 1] (win/loss probability)
        self.value_head = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            # Connect to a single scalar then Tanh for [-1, 1]
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """Processes board state to return (policy_log_probs, state_value)."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten 4D tensor to 2D
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value