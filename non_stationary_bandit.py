# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:42:03 2018

@author: 0000011331748
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import random

K = 10 # number of bandits
time_steps = 10000 # number of time steps
epsilon = 0.1 # for epsilon greedy action
num_of_runs = 2000 # number of runs
alpha = 0.1 # constant step-size
mu = 0 # mean of increment
sigma = 0.01 # standard deviation of the increment

average_reward = np.zeros([2,time_steps])  # average across all runs for each time steps


for type in range(0,2):
    for runs in range(1, num_of_runs + 1):
        
        # we start random walk of q_star afresh in each run
        # note that these q_stars are the actual rewards
        q_star = np.zeros(K)# starting value is taken as 0 for all bandits
        sample_average_reward = np.zeros(K) # for keeping track of the sample average of each bandit 
        
        greedy_sel = bernoulli.rvs(1 - epsilon, size = time_steps) # making a list of what will mode will be selected in each time step
        
        if type == 0:
            frequency_of_occurance = np.zeros(K) # this stores the 
    
        for t in range(1, time_steps + 1):
            
            # variation in q_star
            # change q_star in each step with normal distributed increment
            q_star = q_star + np.random.normal(mu,sigma,K)
            if greedy_sel[t-1] == 1:
                A_t = np.argmax(sample_average_reward)    # greedy selection
            else:
                A_t = random.sample(range(0,K),1)  # random selection
            R_t = q_star[A_t]
            
            
            if type == 0:
                # incrementally computed
                frequency_of_occurance[A_t] = frequency_of_occurance[A_t] + 1
                step_size = 1/frequency_of_occurance[A_t]
            else:            
                # constant step-size
                step_size = alpha
            sample_average_reward[A_t] = sample_average_reward[A_t] + step_size*(R_t - sample_average_reward[A_t])  # update the sample average
            average_reward[type][t-1] = average_reward[type][t-1] + (1/runs)*(R_t - average_reward[type][t-1]) # updating the average award for plot
            
            

plt.plot(average_reward[0], label = 'incrementally computed')
plt.plot(average_reward[1], label = 'constant step-size')               
plt.xlabel('time steps')
plt.ylabel('average reward')
plt.legend()
plt.show()    
            
            



