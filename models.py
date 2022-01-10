import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, image_size, image_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, 3, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.fc = nn.Linear((image_size // 16) ** 2 * 512, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
