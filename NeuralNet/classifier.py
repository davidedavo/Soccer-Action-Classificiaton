import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_1d(nn.Module):
    def __init__(self, num_classes):
        super(CNN_1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.fc1 = nn.Linear(21728, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.pool = nn.AvgPool1d(4, 4)
    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        print(x.shape)
        bs, _, _ = x.shape
        x = self.pool(x).reshape(bs, -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        #RICORDARSI TORCH.NO_GRAD NELL'EXTRACTION