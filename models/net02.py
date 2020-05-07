import torch.nn as nn
import torch.nn.functional as F
from .common import CLASSES_COUNT

model_name = "net02"

class Net02(nn.Module):
    def __init__(self):
        super(Net02, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(128, 128, 5)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.fc1 = nn.Linear(128 * 12 * 12, 2048)
        self.fc2 = nn.Linear(2048, CLASSES_COUNT)

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = self.bn3(self.pool(F.relu(self.conv3(x))))
        x = self.bn4(self.pool(F.relu(self.conv4(x))))
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x