import torch
import torch.nn as nn

channel_out = [32, 64, 128, 256]
kernel_size = [8, 8, 8, 8]
conv_stride_size = [1, 1, 1, 1]
pool_stride_size = [4, 4, 4, 4]
pool_size = [8, 8, 8, 8]

class DFNet(nn.Module):

    def __init__(self, numClass):
        super(DFNet,self).__init__()
        self.feature_layers = nn.Sequential(
            #conv1
            nn.Conv1d(
                in_channels=1, out_channels=channel_out[0], kernel_size=kernel_size[0], stride=conv_stride_size[0], padding='same'),
            nn.BatchNorm1d(channel_out[0]),
            nn.ELU(alpha=1.0),
            nn.Conv1d(in_channels=channel_out[0], out_channels=channel_out[0],
                      kernel_size=kernel_size[0], stride=conv_stride_size[0], padding='same'),
            nn.BatchNorm1d(channel_out[0]),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=pool_size[0], stride=pool_stride_size[0]),
            nn.Dropout(p=0.1),
            #conv2
            nn.Conv1d(
                in_channels=channel_out[0], out_channels=channel_out[1], 
                kernel_size=kernel_size[1], stride=conv_stride_size[1], padding='same'),
            nn.BatchNorm1d(channel_out[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=channel_out[1], out_channels=channel_out[1],
                      kernel_size=kernel_size[1], stride=conv_stride_size[1], padding='same'),
            nn.BatchNorm1d(channel_out[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1]),
            nn.Dropout(p=0.1),
            #conv3
            nn.Conv1d(
                in_channels=channel_out[1], out_channels=channel_out[2], 
                kernel_size=kernel_size[2], stride=conv_stride_size[2], padding='same'),
            nn.BatchNorm1d(channel_out[2]),
            nn.ReLU(),
            nn.Conv1d(in_channels=channel_out[2], out_channels=channel_out[2],
                      kernel_size=kernel_size[2], stride=conv_stride_size[2], padding='same'),
            nn.BatchNorm1d(channel_out[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2]),
            nn.Dropout(p=0.1),
            nn.Conv1d(
                in_channels=channel_out[2], out_channels=channel_out[3], 
                kernel_size=kernel_size[3], stride=conv_stride_size[3], padding='same'),
            nn.BatchNorm1d(channel_out[3]),
            nn.ReLU(),
            nn.Conv1d(in_channels=channel_out[3], out_channels=channel_out[3],
                      kernel_size=kernel_size[3], stride=conv_stride_size[3], padding='same'),
            nn.BatchNorm1d(channel_out[3]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3]),
            nn.Dropout(p=0.1),
            #fc1
            nn.Flatten(),
            nn.Linear(4608, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.7)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, numClass)
        )

    def forward(self, x):
        x = self.feature_layers(x)
        self.feature = x
        x = self.fc(x)
        return x
