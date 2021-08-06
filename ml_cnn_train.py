# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 22:13:18 2021

@author: gdx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import matplotlib.pyplot as plt

from utils.dataread import Ml_dataread
from utils.metrics import Ml_metrics
from network.ml_cnn import ML_CNN

# dummy_input = torch.randn(1, 1, 640)
# net = ML_CNN()
# torch.onnx.export(net, dummy_input, "./model_struct/ml_cnn.onnx")

### model_train
use_gpu=True
if use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print('use_gpu')
else:
    device = torch.device('cpu')
    print('use_cpu')

 ### para settings
Epoch = 200
Batch = 288
model_save_path = './model/ml_cnn35.pth'

 ### load data
file_path='./datasets/train_dataset/'
data_name='train_ml_300_rand'
train_data = Ml_dataread(file_path, data_name, label_num=7)

train_loader = DataLoader(dataset=train_data,
                          batch_size=Batch,
                          shuffle=True,
                          pin_memory=False,
                          drop_last=True)

 ### val setting
file_path='./datasets/train_dataset/'
data_name='train_ml_100_rand'
val_data = Ml_dataread(file_path, data_name, label_num=7)
val_loader = DataLoader(dataset=val_data,
                        batch_size=2400,
                        shuffle=True,
                        pin_memory=False,
                        drop_last=True)

 ### creat net, loss and adam
net = ML_CNN().to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_acc = []
train_loss = []
val_acc = []
val_loss = []

metric = Ml_metrics()
plt.ion()
fig = plt.figure(figsize=(10, 4))

start = time.process_time()
 ### the loop
best_acc = 0.
best_epoch = 0
for epoch in range(Epoch):
    running_loss=0.
    running_acc=0.
    net.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        outputs = net(inputs)
        
        optimizer.zero_grad()
        loss_value = criterion(outputs,labels.long().float())
        loss_value.backward()
        optimizer.step()
        
        running_loss += loss_value.item()
        running_acc += metric.multi_acc(outputs,labels)
    
    acc1 = running_acc/(i+1)
    loss1 = running_loss/(i+1)
    train_acc.append(acc1)
    train_loss.append(loss1)

    net.eval()
    with torch.no_grad():
        running_loss=0.
        running_acc=0.
        for i, data in enumerate(val_loader):
            input_val, label_val = data
            input_val, label_val = input_val.to(device), label_val.to(device)
            val_output = net(input_val)
            running_loss += criterion(val_output, label_val.long().float()).item()
            running_acc += metric.multi_acc(val_output,label_val)
        acc2 = running_acc/(i+1)
        loss2 = running_loss/(i+1)
        val_acc.append(acc2)
        val_loss.append(loss2)

    if acc2 >= best_acc:
        best_acc = acc2
        best_epoch = epoch
        if epoch > 0.5*Epoch: #Save the model in the second half of training
            torch.save(net.state_dict(), model_save_path)

    print('epoch: %d | acc: %0.5f | loss: %.5f | val_acc: %0.5f | val_loss: %.5f' % (epoch + 1, acc1,loss1,acc2,loss2))
    plt.clf()
    ax1, ax2 = fig.subplots(1, 2)
    plt.suptitle("train_history (epoch is %d, best acc is %.5f in epoch %d)"%(epoch+1, best_acc, best_epoch+1),
                 fontsize=15, y=0.99)
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.grid(True)
    ax1.set_title('Train and Validation accuracy')
    ax1.plot(train_acc, 'r', label='train acc')
    ax1.plot(val_acc, 'b', label='val acc')
    ax1.legend()
    
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.grid(True)
    ax2.set_title('Train and Validation loss')
    ax2.plot(train_loss, 'g', label='train loss')
    ax2.plot(val_loss, 'k', label='val loss')
    plt.legend()
    plt.pause(0.1)
plt.ioff()

end = time.process_time() 
print('Running time is %s s'%(end-start))
plt.show()

