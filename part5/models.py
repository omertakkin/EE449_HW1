import torch
import torch.nn.functional as F
    

class CNN_5(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)   # Conv-3x3x8
    self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)  # Conv-3x3x16
    self.conv3 = torch.nn.Conv2d(16, 8, 3, padding=1)  # Conv-3x3x8
    self.conv4 = torch.nn.Conv2d(8, 16, 3, padding=1)  # Conv-3x3x16
    self.conv5 = torch.nn.Conv2d(16, 16, 3, padding=1)  # Conv-3x3x16
    self.conv6 = torch.nn.Conv2d(16, 8, 3, padding=1)  # Conv-3x3x8
    self.pool = torch.nn.MaxPool2d(2, 2)
    self.fc1 = torch.nn.Linear(8 * 8 * 8, 64)
    self.fc2 = torch.nn.Linear(64, 10)

  def forward(self, x):
    # Input: N, 3, 32, 32
    x = F.relu(self.conv1(x))   # -> N, 8, 32, 32
    x = F.relu(self.conv2(x))   # -> N, 16, 32, 32
    x = F.relu(self.conv3(x))   # -> N, 8, 32, 32
    x = F.relu(self.conv4(x))   # -> N, 16, 32, 32
    x = self.pool(x)            # -> N, 16, 16, 16
    x = F.relu(self.conv5(x))   # -> N, 16, 16, 16
    x = F.relu(self.conv6(x))   # -> N, 8, 16, 16
    x = self.pool(x)            # -> N, 8, 8, 8

    x = torch.flatten(x, 1)     # -> N, 8 * 8 * 8 = 512
    x = F.relu(self.fc1(x))     # -> N, 64
    x = self.fc2(x)             # -> N, 10
    return x