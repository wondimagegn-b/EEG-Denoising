import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Sequential

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


def Complex_CNN(datanum):
  model = Sequential()
  model.add(layers.Conv1D(32 ,5,activation = 'relu',strides=1,padding="same",input_shape=[ datanum, 1]))
  model.add(layers.Conv1D(32 ,5,activation = 'relu',strides=1,padding="same",input_shape=[ datanum, 1]))
  model.add(layers.Dropout(0.3))

  model.add(BasicBlockall())

  model.add(layers.Conv1D(32 ,1,activation = 'relu',strides=1,padding="same"))
  model.add(layers.Conv1D(32 ,1,activation = 'relu',strides=1,padding="same"))
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(datanum))

  model.build(input_shape=[ 1,datanum, 1] )
  model.summary()
  
  return model