#!/usr/bin/env python
# coding: utf-8

# # CASO PRÁCTICO 2 - MINIMIZACIÓN DE COSTES

# ## ENVIROMENT

# In[1]:


import numpy as np


# In[ ]:


# CREATE A CLASS FOR ENVIRONMENT
class Environment(object):
    #PARAMETERS AND VARIABLES
    def __init__(self, optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 10, initial_rate_data = 60):
        
        #genric variables and parameters
        
        self.optimal_temperature = optimal_temperature
        self.initial_month = initial_month
        self.initial_number_users = initial_number_users
        self.initial_rate_data = initial_rate_data
        self.monthly_atmospheric_temperature = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.current_number_users = initial_number_users
        self.current_rate_data = initial_rate_data
        
        #specific variables to solve the energetyc problem
        
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsec_temperature
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        
    #METHOD TO ACTUALIZE THE ENVIRONMENT AFTER IA EXECUTE THE ACTION
    def update_env(self, direction, energy_ai, month):
        #OBTAINING REWARD
        
        #Calculating wasted energy by refrigeration system of server without AI
        energy_noai = 0
        if  self.temperature_noai < self.optimal_temperature[0]:
            energy_noai = self.optimal_temperature[0]-self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif self.temperature_noai > self.optimal_temperature[1]:
            energy_noai = self.temperature_noai-self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]
        
        #Calculating reward
        self.reward = energy_noai - energy_ai
        
        #Scaling reward
        self.reward = 1e-3*self.reward #normalized reward
        
        #OBTAINING NEXT STATE
        
        #Actualize atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        #Actualize number of users
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if self.current_number_users < self.min_number_users:
            self.current_number_users = self.min_number_users
        elif self.current_number_users > self.max_number_users:
            self.current_number_users = self.max_number_users
        #Actualize transfer data rate
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if self.current_rate_data < self.min_rate_data:
            self.current_rate_data = self.min_rate_data
        elif self.current_rate_data > self.max_rate_data:
            self.current_rate_data = self.max_rate_data
        #Calculating intrinsec temperature variation
        past_intrinsec_temperature = self.intrinsec_temperature
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_intrinsec_temperature = self.intrinsec_temperature - past_intrinsec_temperature
        
        #Calculating temperature variation caused by AI
        if direction == -1:
            delta_temperature_ai = -energy_ai
        elif direction == 1:
            delta_temperature_ai = energy_ai
        #Calculating new temperature of server when AI is on
        self.temperature_ai += delta_intrinsec_temperature + delta_temperature_ai
        #Calculating new temperature of server when AI is off
        self.temperature_noai += delta_intrinsec_temperature
        #OBTAINING GAME OVER
        if self.temperature_ai < self.min_temperature:
            if self.train == 1:
                self.game_over = 1
            else:
                self.total_energy_ai+= (self.optimal_temperature[0] - self.temperature_ai)
                self.temperature_ai = self.optimal_temperature[0]
        if self.temperature_ai > self.max_temperature:
            if self.train == 1:
                self.game_over = 1
            else:
                self.total_energy_ai += (self.temperature_ai - self.optimal_temperature[1])
                self.temperature_ai = self.optimal_temperature[1]
        #ACTUALIZE THE SCORES
        #Calculating total wasted energy by AI
        self.total_energy_ai += energy_ai
        #Calculating total wasted energy by refrigeration system of server without AI
        self.total_energy_noai += energy_noai
        #SCALING NEXT STATE
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        next_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        #RETURN NEXT STATE, REWARD AND GAME OVER
        return next_state, self.reward, self.game_over
    #METHOD TO REINITIALIZE THE ENVIRONMENT
    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[new_month]
        self.initial_month = new_month
        self.current_number_users = self.initial_number_users
        self.current_rate_data = self.initial_rate_data
        self.intrinsec_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsec_temperature
        self.temperature_noai = (self.optimal_temperature[0]+self.optimal_temperature[1])/2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1
        
    #METHOD TO OBTAIN IN ANY MOMENT THE STATE, LAST REWARD AND GAME OVER VALUE
    def observe(self):
        #SCALING CURRENT STATE
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature) / (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_number_users, scaled_rate_data])
        #RETURN CURRENT STATE, REWARD AND GAME OVER
        return current_state, self.reward, self.game_over

