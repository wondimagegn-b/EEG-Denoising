import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Sequential


def fcNN(datanum):
  model = tf.keras.Sequential()
  model.add(Input(shape=(datanum,)))
  model.add(layers.Dense(datanum, activation=tf.nn.relu ))
  model.add(layers.Dropout(0.3))


  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  
  return model


def RNN_lstm(datanum):
  model = tf.keras.Sequential()
  model.add(Input(shape=(datanum,1)))
  model.add(layers.LSTM(1,return_sequences = True ))

  model.add(layers.Flatten())

  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(datanum))
  model.summary()
  return model


def simple_CNN(datanum):
  model = tf.keras.Sequential()

  model.add(layers.Conv1D(64, 3, strides=1, padding='same',input_shape=[ datanum, 1]))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  #model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  #model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  #model.add(layers.Dropout(0.3))

  #num4
  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  #model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(datanum))

  model.build(input_shape=[ 1,datanum, 1] )
  model.summary()

  return model

# Resnet Basic Block module。
class Res_BasicBlock(layers.Layer):
  def __init__(self,kernelsize, stride=1):
    super(Res_BasicBlock, self).__init__()
    self.bblock = Sequential([layers.Conv1D(32,kernelsize,activation = 'relu',strides=stride,padding="same"),
                              layers.Conv1D(32,kernelsize,activation = 'relu',strides=stride,padding="same"),
                              layers.Dropout(0.3),
			      layers.Conv1D(16,kernelsize,activation = 'relu',strides=1,padding="same"),
                              layers.Conv1D(16,kernelsize,activation = 'relu',strides=1,padding="same"),
                              layers.Dropout(0.3),
                              layers.Conv1D(32,kernelsize,activation = 'relu',strides=1,padding="same"),
                              layers.Conv1D(32,kernelsize,activation = 'relu',strides=1,padding="same"),
                              layers.Dropout(0.3)])
                              
    self.jump_layer = lambda x:x


  def call(self, inputs, training=None):

    #Through the convolutional layer
    out = self.bblock(inputs)

    #skip
    identity = self.jump_layer(inputs)

    output = layers.add([out, identity])  #layers下面有一个add，把这2个层添加进来相加。
    
    return output


class BasicBlockall(layers.Layer):
  def __init__(self, stride=1):
    super(BasicBlockall, self).__init__()

    self.bblock3 = Sequential([Res_BasicBlock(3),
                              Res_BasicBlock(3)
                              ])                      
    
    self.bblock5 = Sequential([Res_BasicBlock(5),
                              Res_BasicBlock(5)
                              ])                      

    self.bblock7 = Sequential([Res_BasicBlock(7),
                              Res_BasicBlock(7)
                              ])
                              
    self.downsample = lambda x:x


  def call(self, inputs, training=None):
 
    out3 = self.bblock3(inputs)
    out5 = self.bblock5(inputs)
    out7 = self.bblock7(inputs)

    out = tf.concat( values = [out3,out5,out7] , axis = -1)

    return out


import torch.nn as nn
import torch



class TwobythreeR_CNN(nn.Module):
    def __init__(self, batch_size, datanum):
        super(TwobythreeR_CNN, self).__init__()
        self.batch_size = batch_size
        self.datanum = datanum
        self.net0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.3)
            )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3),
        )
        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.Dropout(0.3),
        )
        self.net4 = nn.Sequential(
            nn.Conv1d(in_channels=32*3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(0.3),
        )
        self.lin = nn.Linear(datanum*32, datanum)

    def forward(self, x):
        x = torch.reshape(x, [-1, 1, self.datanum])
        x = self.net0(x)
        out = self.net1(x)
        out1 = self.net1(out + x)
        out1 = out1 + out

        out = self.net2(x)
        out2 = self.net2(out + x)
        out2 = out2 + out

        out = self.net3(x)
        out3 = self.net3(out + x)
        out3 = out3 + out

        out = torch.cat([out1, out2, out3], 1)
        out = self.net4(out)
        out = out.view(-1, 32*self.datanum)
        out = self.lin(out)
        return out



