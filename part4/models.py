import torch
import torch.nn.functional as F


class MLP_1_ReLU(torch.nn.Module):
  def __init__(self):
    super(MLP_1_ReLU, self).__init__()
    self.fc1 = torch.nn.Linear(3 * 32 * 32, 32)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(32, 10)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out
  
class MLP_1_Sigmoid(torch.nn.Module):
  def __init__(self):
    super(MLP_1_Sigmoid, self).__init__()
    self.fc1 = torch.nn.Linear(3 * 32 * 32, 32)
    self.sigmoid = torch.nn.Sigmoid()
    self.fc2 = torch.nn.Linear(32, 10)

  def forward(self, x):
    out = self.fc1(x)
    out = self.sigmoid(out)
    out = self.fc2(out)
    return out
  

class MLP_2_ReLU(torch.nn.Module):
  def __init__(self):
    super(MLP_2_ReLU, self).__init__()
    self.fc1 = torch.nn.Linear(3 * 32 * 32, 32)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(32, 64, bias=False)
    self.fc3 = torch.nn.Linear(64, 10)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    return out
  
class MLP_2_Sigmoid(torch.nn.Module):
  def __init__(self):
    super(MLP_2_Sigmoid, self).__init__()
    self.fc1 = torch.nn.Linear(3 * 32 * 32, 32)
    self.sigmoid = torch.nn.Sigmoid()
    self.fc2 = torch.nn.Linear(32, 64, bias=False)
    self.fc3 = torch.nn.Linear(64, 10)

  def forward(self, x):
    out = self.fc1(x)
    out = self.sigmoid(out)
    out = self.fc2(out)
    out = self.sigmoid(out)
    out = self.fc3(out)
    return out
  

class CNN_3_ReLU(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)  # Conv-3x3x16
    self.conv2 = torch.nn.Conv2d(16, 8, 5, padding=2)  # Conv-5x5x8
    self.conv3 = torch.nn.Conv2d(8, 16, 7, padding=3)  # Conv-7x7x16
    self.pool = torch.nn.MaxPool2d(2, 2)
    self.fc1 = torch.nn.Linear(16 * 8 * 8, 64)
    self.fc2 = torch.nn.Linear(64, 10)

  def forward(self, x):
    # Input: N, 3, 32, 32
    x = F.relu(self.conv1(x))   # -> N, 16, 32, 32
    x = F.relu(self.conv2(x))   # -> N, 8, 32, 32
    x = self.pool(x)            # -> N, 8, 16, 16
    x = F.relu(self.conv3(x))   # -> N, 16, 16, 16
    x = self.pool(x)            # -> N, 16, 8, 8
    x = torch.flatten(x, 1)     # -> N, 16 * 8 * 8 = 1024
    x = F.relu(self.fc1(x))     # -> N, 64
    x = self.fc2(x)             # -> N, 10
    return x
  
class CNN_3_Sigmoid(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)  # Conv-3x3x16
    self.conv2 = torch.nn.Conv2d(16, 8, 5, padding=2)  # Conv-5x5x8
    self.conv3 = torch.nn.Conv2d(8, 16, 7, padding=3)  # Conv-7x7x16
    self.pool = torch.nn.MaxPool2d(2, 2)
    self.fc1 = torch.nn.Linear(16 * 8 * 8, 64)
    self.fc2 = torch.nn.Linear(64, 10)

  def forward(self, x):
    # Input: N, 3, 32, 32
    x = F.sigmoid(self.conv1(x))   # -> N, 16, 32, 32
    x = F.sigmoid(self.conv2(x))   # -> N, 8, 32, 32
    x = self.pool(x)            # -> N, 8, 16, 16
    x = F.sigmoid(self.conv3(x))   # -> N, 16, 16, 16
    x = self.pool(x)            # -> N, 16, 8, 8
    x = torch.flatten(x, 1)     # -> N, 16 * 8 * 8 = 1024
    x = F.sigmoid(self.fc1(x))     # -> N, 64
    x = self.fc2(x)             # -> N, 10
    return x
    
class CNN_4_ReLU(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   # Conv-3x3x16
    self.conv2 = torch.nn.Conv2d(16, 8, 3, padding=1)  # Conv-3x3x8
    self.conv3 = torch.nn.Conv2d(8, 16, 5, padding=2)  # Conv-5x5x16
    self.conv4 = torch.nn.Conv2d(16, 16, 5, padding=2)  # Conv-5x5x16
    self.pool = torch.nn.MaxPool2d(2, 2)
    self.fc1 = torch.nn.Linear(16 * 8 * 8, 64)
    self.fc2 = torch.nn.Linear(64, 10)

  def forward(self, x):
    # Input: N, 3, 32, 32
    x = F.relu(self.conv1(x))   # -> N, 16, 32, 32
    x = F.relu(self.conv2(x))   # -> N, 8, 32, 32
    x = F.relu(self.conv3(x))   # -> N, 16, 32, 32
    x = self.pool(x)            # -> N, 16, 16, 16
    x = F.relu(self.conv4(x))   # -> N, 16, 16, 16
    x = self.pool(x)           # -> N, 16, 8, 8

    x = torch.flatten(x, 1)     # -> N, 16 * 8 * 8 = 1024
    x = F.relu(self.fc1(x))     # -> N, 64
    x = self.fc2(x)             # -> N, 10
    return x
    
class CNN_4_Sigmoid(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   # Conv-3x3x16
    self.conv2 = torch.nn.Conv2d(16, 8, 3, padding=1)  # Conv-3x3x8
    self.conv3 = torch.nn.Conv2d(8, 16, 5, padding=2)  # Conv-5x5x16
    self.conv4 = torch.nn.Conv2d(16, 16, 5, padding=2)  # Conv-5x5x16
    self.pool = torch.nn.MaxPool2d(2, 2)
    self.fc1 = torch.nn.Linear(16 * 8 * 8, 64)
    self.fc2 = torch.nn.Linear(64, 10)

  def forward(self, x):
    # Input: N, 3, 32, 32
    x = F.sigmoid(self.conv1(x))   # -> N, 16, 32, 32
    x = F.sigmoid(self.conv2(x))   # -> N, 8, 32, 32
    x = F.sigmoid(self.conv3(x))   # -> N, 16, 32, 32
    x = self.pool(x)            # -> N, 16, 16, 16
    x = F.sigmoid(self.conv4(x))   # -> N, 16, 16, 16
    x = self.pool(x)           # -> N, 16, 8, 8

    x = torch.flatten(x, 1)     # -> N, 16 * 8 * 8 = 1024
    x = F.sigmoid(self.fc1(x))     # -> N, 64
    x = self.fc2(x)             # -> N, 10
    return x

class CNN_5_ReLU(torch.nn.Module):
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
  
class CNN_5_Sigmoid(torch.nn.Module):
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
    x = F.sigmoid(self.conv1(x))   # -> N, 8, 32, 32
    x = F.sigmoid(self.conv2(x))   # -> N, 16, 32, 32
    x = F.sigmoid(self.conv3(x))   # -> N, 8, 32, 32
    x = F.sigmoid(self.conv4(x))   # -> N, 16, 32, 32
    x = self.pool(x)            # -> N, 16, 16, 16
    x = F.sigmoid(self.conv5(x))   # -> N, 16, 16, 16
    x = F.sigmoid(self.conv6(x))   # -> N, 8, 16, 16
    x = self.pool(x)            # -> N, 8, 8, 8

    x = torch.flatten(x, 1)     # -> N, 8 * 8 * 8 = 512
    x = F.sigmoid(self.fc1(x))     # -> N, 64
    x = self.fc2(x)             # -> N, 10
    return x