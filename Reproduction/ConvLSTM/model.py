import tensorflow as tf

from avod.core.feature_extractors import img_feature_extractor
#from research.lstm_object_detection.lstm import lstm_cells
#from /home/chris/models-master/research/lstm_object_detection.lstm import rnn_decoder
#from tensorflow.python.framework import ops as tf_ops
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling3D
from keras.optimizers import SGD


from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import TimeDistributed


from tensorflow.contrib import slim as contrib_slim
from tensorflow.python.framework import ops as tf_ops

from tensorflow.keras import layers

def My_ConvLSTM_Model(frames, channels, pixels_x, pixels_y, categories):
  
    trailer_input  = Input(shape=(frames, channels, pixels_x, pixels_y)
                    , name='trailer_input')
    
    first_ConvLSTM = ConvLSTM2D(filters=20, kernel_size=(3, 3)
                       , data_format='channels_first'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same', return_sequences=True)(trailer_input)
    first_BatchNormalization = BatchNormalization()(first_ConvLSTM)
    first_Pooling = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first')(first_BatchNormalization)
    
    second_ConvLSTM = ConvLSTM2D(filters=10, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , padding='same', return_sequences=True)(first_Pooling)
    second_BatchNormalization = BatchNormalization()(second_ConvLSTM)
    second_Pooling = MaxPooling3D(pool_size=(1, 3, 3), padding='same', data_format='channels_first')(second_BatchNormalization)
    
    outputs = [branch(second_Pooling, 'cat_{}'.format(category)) for category in categories]
    
    seq = Model(inputs=trailer_input, outputs=outputs, name='Model ')
    
    return seq

def branch(last_convlstm_layer, name):
  
    branch_ConvLSTM = ConvLSTM2D(filters=5, kernel_size=(3, 3)
                        , data_format='channels_first'
                        , stateful = False
                        , kernel_initializer='random_uniform'
                        , padding='same', return_sequences=True)(last_convlstm_layer)
    branch_Pooling = MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_first')(branch_ConvLSTM)
    flat_layer = TimeDistributed(Flatten())(branch_Pooling)
    
    first_Dense = TimeDistributed(Dense(512,))(flat_layer)
    second_Dense = TimeDistributed(Dense(32,))(first_Dense)
    
    target = TimeDistributed(Dense(1), name=name)(second_Dense)
    
    return target

def generate_arrays(available_ids):
    
	from random import shuffle
    while True:
        
        shuffle(available_ids)
        for i in available_ids:
            
            scene = np.load('dataset/scene_{}.npy'.format(i))
            category = np.load('dataset/category_{}.npy'.format(i))
            yield (np.array([scene]), category)