# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:05:24 2018

@author: 0000011331748
"""

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
time_steps = 1000 # number of time steps
num_of_runs = 2000 # number of runs

epsilon = [0, 0.1] # for epsilon greedy action
Q_init = [5, 0] # initial values 

mu = 0 # mean of increment
sigma = 1 # standard deviation of the increment

alpha = 0.1
average_reward = np.zeros([2,time_steps])  # average across all runs for each time steps
percent_optimal_action = np.zeros([2,time_steps]) # percentage of optimal actions

for type in range(0,2):
    for runs in range(1, num_of_runs + 1):
        
        q_star = np.random.normal(mu, sigma, K) # distribution of mean is decided at the beginning fof each run    
        optimal_action = np.argmax(q_star)
        sample_average_reward = np.ones(K)*Q_init[type] # initialization of each sample average reward            
        greedy_sel = bernoulli.rvs(1 - epsilon[type], size = time_steps) # making a list of what will mode will be selected in each time step
        

        for t in range(1, time_steps + 1):
            if greedy_sel[t-1] == 1:
                A_t = np.argmax(sample_average_reward)    # greedy selection
            else:
                A_t = random.sample(range(0,K),1)  # random selection
            R_t = np.random.normal(q_star[A_t], sigma, 1) # random selection of award from its probability distribution
            
            
            step_size = alpha
            sample_average_reward[A_t] = sample_average_reward[A_t] + step_size*(R_t - sample_average_reward[A_t])  # update the sample average
            average_reward[type][t-1] = average_reward[type][t-1] + (1/runs)*(R_t - average_reward[type][t-1]) # updating the average award for plot
            
            # checking with optimal action
            if A_t == optimal_action:
                percent_optimal_action[type][t-1] = percent_optimal_action[type][t-1] + 1
            
            
percent_optimal_action = (percent_optimal_action/num_of_runs)*100

plt.figure(1)
plt.plot(average_reward[0], label = 'Optimistic, greedy')
plt.plot(average_reward[1], label = 'realistic, epsilon-greedy')               
plt.xlabel('time steps')
plt.ylabel('average reward')
plt.legend()
 
           
            
plt.figure(2)
plt.plot(percent_optimal_action[0], label = 'Optimistic, greedy')
plt.plot(percent_optimal_action[1], label = 'realistic, epsilon-greedy')               
plt.xlabel('time steps')
plt.ylabel('% optimal action')
plt.legend()


plt.show()    
            

