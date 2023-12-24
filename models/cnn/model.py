import torch.nn as nn


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),  # 220
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 110

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 106
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 53

            nn.Flatten(),
            nn.Linear(53 * 53 * 16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        return self.network(x)
