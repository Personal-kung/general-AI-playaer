import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)


class AlphaNet(nn.Module):
    def __init__(self, input_shape=(1, 6, 7), action_size=42, num_res_blocks=4):
        super().__init__()
        # Initial Convolution
        self.start_block = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Backbone: Residual Towers
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(128) for _ in range(num_res_blocks)]
        )

        # Policy Head (Probability of moves)
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * input_shape[1] * input_shape[2], action_size),
            nn.LogSoftmax(dim=1),
        )

        # Value Head (Who is winning?)
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * input_shape[1] * input_shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.start_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)
