# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:51:05 2021

@author: gdx
"""

import torch
from torch import nn
#from cnn_atten_rnn import CNN

class CNN(nn.Module):
    """
    input_shape: batchsize * 1 * 640
    output_shape: batchsize * num_labels
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2)
        )

    def forward(self, x):
        x=x.float()
        x = self.stage1(x)
        return x


class ML_CNN(nn.Module):
    """
    input_shape: batchsize * 1 * 640
    output_shape: batchsize * num_labels
    tips : include sigmoid
    """
    def __init__(self):
        super(ML_CNN, self).__init__()
        self.cnn = CNN()
        self.gap = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(128, 8)
            )

    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x)
        x = x.view(-1,128)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x



