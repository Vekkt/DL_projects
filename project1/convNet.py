import torch
from torch import nn
import torch.functional as F

class DigiNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 2, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.feature_classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU()
        )

    def forward(self, input):
        x = self.feature_extractor(input)
        x = x.flatten(start_dim=1)
        x = self.feature_classifier(x)
        return x

class PairNet(nn.Module):
    def __init__(self, aux_loss=False, weight_sharing=False):
        super().__init__()
        self.aux_loss = aux_loss
        self.weight_sharing = weight_sharing

        self.net1 = DigiNet()
        # Network-level weight sharing
        self.net2 = self.net1 if weight_sharing else DigiNet()
        self.feature_classifier = nn.Linear(100, 1)

    def forward(self, input):
        image1, image2 = input[:100].split(1, dim=1)
        digit1 = self.net1(image1)
        digit2 = self.net1(image2)

        # I used another feature classifier here
        # We could (but it my be bad) to predict the digit
        # by directly taking the argmax of digit1 and digit2
        # (i.e. index of the maximum value, between 0 et 9)
        x = torch.bmm(digit1[:, :, None], digit2[:, None, :])
        x = x.flatten(start_dim=1)
        res = self.feature_classifier(x)

        if self.aux_loss:
            return res, digit1, digit2
        else:
            return res


