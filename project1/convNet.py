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
            nn.ReLU(),
            nn.Flatten()
        )

        self.net = nn.Sequential(
            self.feature_extractor,
            nn.Flatten(),
            self.feature_classifier
        )

    def forward(self, input):
        return self.net(input)

class PairNet(nn.Module):
    def __init__(self, aux_loss=False, weight_sharing=False):
        super().__init__()
        self.aux_loss = aux_loss
        self.weight_sharing = weight_sharing

        self.net1 = DigiNet()
        # Network-level weight sharing
        self.net2 = self.net1 if weight_sharing else DigiNet()
        self.feature_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 1)
        )

    def forward(self, input):
        image1, image2 = input[:100].split(1, dim=1)
        output1 = self.net1(image1)
        output2 = self.net2(image2)

        # I used another feature classifier here
        # We could (but it my be bad) predict the digit
        # by directly taking the argmax of digit1 and digit2
        # (i.e. index of the maximum value, between 0 et 9).
        # We could also use a more complex classifier.
        x = torch.bmm(output1.unsqueeze(2), output2.unsqueeze(1))
        res = self.feature_classifier(x).flatten()

        if self.aux_loss:
            return res.float(), torch.cat((output1.unsqueeze(1), output2.unsqueeze(1)), 1).float()
        else:
            return res.float()


