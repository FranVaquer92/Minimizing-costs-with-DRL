#!/usr/bin/env python
# coding: utf-8

# # CASO PRÁCTICO 2 - MINIMIZACIÓN DE COSTES

# ## TRAINING

# In[ ]:


#Import libraries and another python files
import os
import numpy as np
import random as rn

import environment
import brain
import dqn

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

#Set up reproducibility seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

#SET UP OF THE PARAMETERS
epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions - 1)/2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

#BUILD OF THE ENVIRONMENT BY THE CREATION OF ENVIRONMENT OBJECT
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#BUILD OF THE BRAIN BY THE CRATION OF BRAIN OBJECT
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

#BUILD OF THE DQN MODEL BY THE CREATION OF DQN OBJECT
dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

#ELECTION OF TRAIN MODE
train = True

#TRAINING THE AI
env.train = train
model = brain.model

early_stopping = True
patience = 10
min_loss = 1
patience_count = 0

if env.train:
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0,12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        #INITIALIZATION OF TIMESTEP BUCLE (Timestep = 1 minute) PER EPOCH
        while ((not game_over) and (timestep <= (5*30*24*60))):
            #RUN THE NEXT EXPLORATION ACTION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
            #RUN THE NEXT INFERENCE ACTION
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
            if (action < direction_boundary):
                direction = -1
            else:
                direction = 1
            energy_ai = abs(action - direction_boundary) * temperature_step
            #UPLOAD ENVIRONMENT TO GET THE NEXT STATE
            next_state, reward, game_over = env.update_env(direction, energy_ai, month= int(timestep/(30*24*60)))
            total_reward += reward
            
            #STORE THE NEXT TRANSITION IN THE MEMORY
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            #GET THE TWO BLOCKS DIVIDE BY ENTRIES AND OBJECTIVES
            inputs, targets = dqn.get_batch(model, batch_size)
            
            #CALCULATE THE LOSS FUNCTION USING ALL OF THE BLOCK OF ENTRIES AND OBJECTIVES
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
            
        #PRINT TRINING RESULTS AT THE FINISH OF EPOCHS
        print('\n')
        print('Epoch: {:03d}/{:03d}'.format(epoch, number_epochs))
        print(' - TOTAL ENERGY WASTED BY THE SYSTEM WITHOUT AI: {:.0f} J'.format(env.total_energy_noai))
        print(' - TOTAL ENERGY WASTED BY THE SYSTEM WITH AI: {:.0f} J'.format(env.total_energy_ai))
        
        #EARLY STOPPING
        if early_stopping:
            if loss >= min_loss:
                patience_count += 1
            else:
                min_loss = loss
                patience_count = 0
            if patience_count >= patience:
                print('Execution of early stopping')
                break
        
        #SAVE THE MODEL TO USE IN FUTURE JOBS
        model.save('model_dql_es2.h5')

