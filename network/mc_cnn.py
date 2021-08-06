# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:43:45 2021

@author: gdx
"""
import torch
from torch import nn
import torch.nn.functional as F

### cnn extract features
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
        x = x.float()
        x = self.stage1(x)
        return x
    
class MC_CNN(nn.Module):
    """
    input_shape: batchsize * 1 * 640
    output_shape: batchsize * num_labels
    """
    def __init__(self):
        super(MC_CNN, self).__init__()
        self.cnn = CNN()
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.liner1 = nn.Linear(128 , 128)
        self.drop = nn.Dropout(p=0.25)
        self.liner2 = nn.Linear(128 , 48)

    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.liner1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.liner2(x)
        out = F.softmax(x,dim=1)
        
        if self.training is True: #no activation in training
          return x
        else:
          return out

#reproduce the paper deep cnn from Applied Energy
class deep_cnn(nn.Module):
    """
    input_shape: batchsize * 1 * 640
    output_shape: batchsize * num_labels
    """
    def __init__(self):
        super(deep_cnn, self).__init__()        
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 3, 1, 1),
            nn.ReLU(True),         
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.ReLU(True),         
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.ReLU(True),         
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveMaxPool1d(1),
            nn.BatchNorm1d(256)
        )
        
        self.liner1 = nn.Sequential(nn.Linear(256 , 128), nn.ReLU(True), nn.BatchNorm1d(128))
        self.liner2 = nn.Linear(128 , 48)

    def forward(self, x):
        x=x.float()
        x = self.stage1(x)
        x = x.view(-1, 256)
        x = self.liner1(x)
        x = F.relu(self.liner2(x))
        out = F.softmax(x,dim=1)
        
        if self.training is True: #no activation in training
          return x
        else:
          return out


#reproduce the paper cnn-LSTM from 2017-Deep Power:
class CNN_LSTM(nn.Module):
    """
    input_shape: batchsize * 1 * 640
    output_shape: batchsize * num_labels
    """
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1,padding=1),
                    nn.ReLU(True),
                    nn.MaxPool1d(2, 2),
                    
                    nn.Conv1d(64, 128, 3, 1, 1),
                    nn.ReLU(True),         
                    nn.MaxPool1d(2, 2))
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=50, batch_first=True) #batch, time_step, input_size
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.liner = nn.Linear(50, 48)

    def forward(self, x):
        x = x.float()
        x = self.cnn(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = torch.transpose(x, 1, 2)
        x = self.gap(x)
        x = x.view(-1,50)
        x = self.liner(x)
        out = F.softmax(x,dim=1)

        if self.training is True: #no activation in training
          return x
        else:
          return out

# dumy_input = torch.randn(2,1,640)
# net = CNN_LSTM()
# out = net(dumy_input)
# print(out.shape)
