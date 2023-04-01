import torch
import torch.nn as nn

class p1fp(nn.Module):

    def __init__(self, numClass):
        super(p1fp,self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=12, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, padding=5),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=12, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, padding=5),
            #fc1
            nn.Flatten(),
            nn.Linear(6528, 1024),
            nn.Tanh(),
            nn.Dropout(p=0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, numClass)
        )
        #self.final = nn.Softmax(dim = 0)

    def forward(self, x):
        x = self.feature_layers(x)
        self.feature = x
        x = self.fc(x)
        return x