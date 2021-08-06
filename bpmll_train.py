# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 21:50:31 2021

@author: gdx
"""
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import optimizers

from utils.mylosses import bp_mll_loss
from utils.dataread import Ml_dataread
import matplotlib.pyplot as plt

#save and show loss&acc
class LossHistory(Callback):
    def __init__(self,epoch_size,model_path):
        self.epoch_size=epoch_size
        self.model_path=model_path
        
    def on_train_begin(self, logs={}):       
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []
        
        self.highest = 0. #store the best accuracy
        self.highest_epoch = 0 #

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('multi_acc'))
        self.val_acc.append(logs.get('val_multi_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
        if self.val_acc[epoch] >= self.highest:
            self.highest = self.val_acc[epoch]
            self.highest_epoch = epoch
            if epoch>0.5*self.epoch_size:
                self.model.save(self.model_path)
        print('epoch: %d | acc: %0.5f | loss: %.5f | val_acc: %0.5f | val_loss: %.5f | best epoch is %3d with val_acc:%.5f' \
              %(epoch, self.acc[epoch], self.loss[epoch], self.val_acc[epoch], self.val_loss[epoch], self.highest_epoch,self.highest))


    def loss_plot(self):
        iters = range(len(self.loss))
        fig=plt.figure()
        # acc
        ax = fig.add_subplot(111)
        line1=ax.plot(iters, self.acc, 'r', label='train acc')
        # val_acc
        line2=ax.plot(iters, self.val_acc, 'b', label='val acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax1=ax.twinx()
        # loss
        line3=ax1.plot(iters, self.loss, 'g', label='train loss')
        # val_loss
        line4=ax1.plot(iters, self.val_loss, 'k', label='val loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        lins = line1+line2+line3+line4
        labs = [l.get_label() for l in lins]
        ax.legend(lins, labs)
        plt.grid(True)
        plt.title('Train and Validation loss & accuracy')
        plt.show()

def cnn_block(x, filters):
    x = Conv1D(filters, kernel_size = 3,padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv1D(filters, 3, padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    return x

### bpmll
def bpmll(input_dim=640, hidden_dim=128, output_dim=8):
    """
    model=creat_model(input_dim,hidden_dim=128)
    """
    input_tensor = Input(shape=(input_dim))
    x = Dense(hidden_dim, activation='relu',name="class_1")(input_tensor)
    x = Dropout(0.25)(x)
    output_tensor = Dense(output_dim, activation='sigmoid',name="class_2")(x)
    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    return model

### cnn_bpmll
def cnn_bpmll(input_dim=640, hidden_dim=128, output_dim=8):
    """
    model=creat_model(input_dim,hidden_dim=128)
    """
    input_tensor = Input(shape=(input_dim,1))
    x = cnn_block(input_tensor, 16)
    x = cnn_block(x, 32)
    x = cnn_block(x, 64)
    x = cnn_block(x, 128)
    x = GlobalMaxPooling1D()(x)
    x = Dense(hidden_dim, activation='relu',name="class_1")(x)
    x = Dropout(0.25)(x)
    output_tensor = Dense(output_dim, activation='sigmoid',name="class_2")(x)
    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    return model


def multi_acc(y_true, y_pred):
    pred = K.round(K.clip(y_pred, 0, 1))
    dpt = K.abs(pred - y_true)
    acc = K.clip(K.sum(dpt,axis=1), 0, 1)
    return 1.-K.mean(acc)

if __name__ == '__main__':  
    ### train model
    file_path='./datasets/train_dataset/'
    data_name='train_ml_300_rand'
    train_data = Ml_dataread(file_path, data_name, label_num=7)
    X_train = train_data.x.squeeze().numpy()
    Y_train = train_data.y.numpy()
    
    file_path='./datasets/train_dataset/'
    data_name='train_ml_100_rand'
    val_data = Ml_dataread(file_path, data_name, label_num=7)   
    X_val = val_data.x.squeeze().numpy()
    Y_val = val_data.y.numpy()
    
    Epoch = 200
    Batch_size = 288
    
    ## if model is cnn_bpmll
    model_save_path = './model/cnn_bpmll35.h5'
    dim_no = X_train.shape[1]    
    X_train = X_train.reshape(-1,dim_no,1)
    X_val = X_val.reshape(-1,dim_no,1)    
    model = cnn_bpmll(input_dim=640, hidden_dim=128, output_dim=8)
    
    ## if model is bpmll
    # model_save_path = './model/bpmll35.h5'
    # model = bpmll(input_dim=640, hidden_dim=128, output_dim=8)
    
    model.compile(loss=bp_mll_loss, 
                  optimizer=optimizers.Adam(lr=0.001), 
                  metrics=[multi_acc])
    model.summary()
    
    history = LossHistory(epoch_size=Epoch,model_path=model_save_path)

    # train a few epochs
    model.fit(X_train, Y_train,
              batch_size=Batch_size,
              epochs=Epoch,
              verbose=0,
              validation_data=(X_val, Y_val),
              callbacks=[history])
    
    history.loss_plot()

    
