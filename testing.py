#!/usr/bin/env python
# coding: utf-8

# # CASO PRÁCTICO 2 - MINIMIZACIÓN DE COSTES

# ## TESTING 

# In[2]:


#Import libraries and another python files
import os
import numpy as np
import random as rn
from keras.models import load_model
import environment

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Set up reproducibility seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#SET UP OF THE PARAMETERS
number_actions = 5
direction_boundary = (number_actions - 1)/2
temperature_step = 1.5

#BUILD OF THE ENVIRONMENT BY THE CREATION OF ENVIRONMENT OBJECT
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#LOAD A PRE TRAINED MODEL
#model = load_model('model_dql.h5')
#model = load_model('model_dql_es1.h5')
model = load_model('model_dql_es2.h5')

#ELECTION OF TRAIN MODE
train = False

#RUN A YEAR OF SIMULATION IN INFERENCE MODE
env.train = train
current_state, _, _ = env.observe()
for timestep in range(0, 12*30*24*60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    if (action < direction_boundary):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, month= int(timestep/(30*24*60)))
    current_state = next_state
    print((timestep/60/24/30/12)*100, '%')

#PRINT TRINING RESULTS AT THE FINISH OF EPOCHS
print('\n')
print(' - TOTAL ENERGY WASTED BY THE SYSTEM WITHOUT AI: {:.0f} J'.format(env.total_energy_noai))
print(' - TOTAL ENERGY WASTED BY THE SYSTEM WITH AI: {:.0f} J'.format(env.total_energy_ai))
print(' - ENERGY SAVED: {:.0f} %'.format(((env.total_energy_noai-env.total_energy_ai)/env.total_energy_noai)*100))

