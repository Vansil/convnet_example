import torch.nn as nn

class CorrelationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 20, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 20 x 16 x 16
            nn.Conv2d(20, 20, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 20 x 8 x 8
            nn.Conv2d(20, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 10 x 4 x 4
            nn.Flatten(),
            nn.Linear(10*4*4, 64),
            nn.ReLU(),
            # 64
            nn.Linear(64, 16),
            nn.ReLU(),
            # 16
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)