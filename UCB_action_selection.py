# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:05:24 2018

@author: 0000011331748
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:42:03 2018

@author: 0000011331748

This program is for UCB action selection
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import random
import math

K = 10 # number of bandits
time_steps = 1000 # number of time steps
num_of_runs = 2000 # number of runs

epsilon = 0.1 # for epsilon greedy action
c = 2
mu = 0 # mean of increment
sigma = 1 # standard deviation of the increment


average_reward = np.zeros([2,time_steps])  # average across all runs for each time steps

for type in range(0,2):
    for runs in range(1, num_of_runs + 1):
        
        q_star = np.random.normal(mu, sigma, K) # distribution of mean is decided at the beginning fof each run    
        sample_average_reward = np.zeros(K) # initialization of each sample average reward            
        
        if type == 0:
            greedy_sel = bernoulli.rvs(1 - epsilon, size = time_steps) # making a list of what will mode will be selected in each time step
        N_A_t = np.zeros(K) # maintaining the array of number of times a particular action has been undertaken        

        for t in range(1, time_steps + 1):

            # action selection part
            if type == 0:  # epsilon-greedy action selection
                if greedy_sel[t-1] == 1:
                    A_t = np.argmax(sample_average_reward)    # greedy selection
                else:
                    A_t = random.sample(range(0,K),1)  # random selection
                    
            else:   # UCB action selection
                zero_checker = np.where(N_A_t == 0)[0] # checking the bandits whether they have occured once or not       
                if len(zero_checker) == 0:
                    # no zero occurance
                    # follow the step in (2.10)
                    var1 = sample_average_reward + c*np.sqrt(math.log(t)/N_A_t)
                    A_t = np.argmax(var1)
                else: 
                    # random select
                    A_t = random.sample(list(zero_checker), 1)
                
            R_t = np.random.normal(q_star[A_t], sigma, 1) # random selection of award from its probability distribution
            N_A_t[A_t] = N_A_t[A_t] + 1  # updating the frequency of actions
            
            
            
            sample_average_reward[A_t] = sample_average_reward[A_t] + (1/N_A_t[A_t])*(R_t - sample_average_reward[A_t])  # update the sample average
            average_reward[type][t-1] = average_reward[type][t-1] + (1/runs)*(R_t - average_reward[type][t-1]) # updating the average award for plot
            
            

plt.plot(average_reward[0], label = 'epsilon, greedy')
plt.plot(average_reward[1], label = 'UCB')               
plt.xlabel('time steps')
plt.ylabel('average reward')
plt.legend()
 
           
            

plt.show()    
            

