#Sunspot
import copy
import multiprocessing
import os
import sys
import gc
import numpy as np
import random
import time
import operator
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import pickle
from sklearn import preprocessing
import os
import numpy as np
from numpy import array
import copy
import pandas as pd


train = pd.read_csv("/home/shravan/Desktop/Sunspot/deeplearning_timeseries-master/data/Sunspot/train1.csv")
test = pd.read_csv( "/home/shravan/Desktop/Sunspot/deeplearning_timeseries-master/data/Sunspot/test1.csv")
print(train.shape)
print(test.shape)
train.drop(labels=train.columns[0], axis=1, inplace=True)
test.drop(labels=train.columns[0], axis=1, inplace=True)

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        sequence = np.asarray(sequence)
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i][0:5], sequence[i][5:15]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

n_steps_in, n_steps_out = 5,10
train_X, train_Y = split_sequence(train, n_steps_in, n_steps_out)
test_X, test_Y= split_sequence(test, n_steps_in, n_steps_out)

train_X = train_X.reshape(571,1,5)[0:570].astype("float32")
train_Y = train_Y.reshape(571,1,10)[0:570].astype("float32")
test_X = test_X.reshape(371,1,5)[0:370].astype("float32")
test_Y = test_Y.reshape(371,1,10)[0:370].astype("float32")
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Conv1D, MaxPooling1D, Activation
from tensorflow.keras import Model
from tensorflow import keras
def data_load(data='train'):
    if data == 'test':
        
        a = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).shuffle(1000).batch(10) 
        

    else:
        a = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(1000).batch(10)
       
    data_loader = a
    return data_loader

import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Conv1D, MaxPooling1D, Activation
from tensorflow.keras import Model
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(64,3, activation = tf.nn.relu, input_shape = (5,1,))
        self.pool = tf.keras.layers.MaxPooling1D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units = 10, activation = tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units = 10)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
 
Model_1 = MyModel()
Model_1.compile(optimizer="Adam", loss="mse")
Model_1.predict(train_X[0])
