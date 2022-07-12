import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in')

    
    def sub_forward(self, x):
        
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu(F.max_pool2d(self.conv3(out), 2))
        out = F.relu(self.conv4(out))

        out = out.view(out.shape[0], -1)
        out = F.sigmoid(self.fc1(out))
        return out

    def forward(self, x1, x2):
        
        h1 = self.sub_forward(x1)
        h2 = self.sub_forward(x2)

        diff = torch.abs(h1 - h2)

        scores = self.fc2(diff)

        return scores