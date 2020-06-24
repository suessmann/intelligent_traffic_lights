import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DQNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.features1 = nn.Sequential(
                nn.Conv2d(1, 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 2, stride=1),
                nn.ReLU()
                )
        self.features2 = nn.Sequential(
                nn.Conv2d(1, 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 2, stride=1),
                nn.ReLU()
                )
        # self.features3 = nn.Linear(4, 4)

        self.linear_relu1 = nn.Sequential(
                nn.Linear(32*7*7 + 32*7*7 + 4, 128),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                # nn.Dropout(0.2)
                )

        self.classifier = nn.Sequential(
                nn.Linear(32, 4),
                # nn.Softmax(dim=-1)
                # nn.ReLU()
                )


    def forward(self, state):
        x1, x2, x3 = state

        x1 = self.features1(x1.view(x1.size(0), 1, -1, 16))
        x2 = self.features2(x2.view(x1.size(0), 1, -1, 16))
        # x3 = self.features3(x3)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.linear_relu1(x)
        x = self.classifier(x)

        return x

    def predict(self, state, eps):
        prob = random.random()
        if prob < eps:
            return random.randint(0, 3)
        else:
            act = self.forward(state)
            return act.argmax().item()


class FCQNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.features1 = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU()
                )
        self.features2 = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU()
                )
        self.linear_q = nn.Linear(128 + 128 + 4, 4)

    def forward(self, state):
        x1, x2, x3 = state

        x1 = self.features1(x1.view(x1.size(0), 1, 1, -1))
        x2 = self.features2(x2.view(x1.size(0), 1, 1, -1))
        # x3 = self.features3(x3)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.linear_q(x)

        return x

    def predict(self, state, eps):
        prob = random.random()
        if prob < eps:
            return random.randint(0, 3)
        else:
            act = self.forward(state)
            return act.argmax().item()
