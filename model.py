import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaNet(nn.Module):
    def __init__(self, input_shape, action_size, num_res_blocks=4):
        super().__init__()
        c, h, w = input_shape

        # Initial Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        # Residual Blocks
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                )
                for _ in range(num_res_blocks)
            ]
        )

        # Heads
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)

        # Dynamic calculation of linear input features
        self.flat_features_policy = 2 * h * w
        self.flat_features_value = 1 * h * w

        self.policy_fc = nn.Linear(self.flat_features_policy, action_size)
        self.value_fc = nn.Sequential(
            nn.Linear(self.flat_features_value, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        for block in self.res_blocks:
            x = F.relu(x + block(x))

        p = self.policy_conv(x).view(-1, self.flat_features_policy)
        v = self.value_conv(x).view(-1, self.flat_features_value)

        return F.log_softmax(self.policy_fc(p), dim=1), self.value_fc(v)
