#!/usr/bin/env python
# coding: utf-8

# # CASO PRÁCTICO 2 - MINIMIZACIÓN DE COSTES

# ## BRAIN

# In[ ]:


#Import libraries
#import sys
#!{sys.executable} -m pip install h5py==2.8.0 
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

#BUILDING THE BRAIN

class Brain(object):
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(units = 64, activation= 'sigmoid')(states)
        y = Dense(units = 32, activation= 'sigmoid')(x)
        q_values = Dense(units = number_actions, activation= 'softmax')(y)
        self.model = Model(inputs = states, output = q_values)
        self.model.compile(loss = 'mse',
                           optimizer = Adam(lr = learning_rate)
                          )

