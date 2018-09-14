#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:24:46 2018

@author: soubhikdeb

This code shows the exploration vs. exploitation
"""





import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import random


K = 10 # number of bandits
time_steps = 2000 # number of time steps
bandit_problems = 2000 # number of k-armed bandit problems

mu = 1
sigma = 1
reward_sigma = 2


'''
greedy policy
'''
avg_reward_greedy = np.zeros(time_steps) # necessary for making the plot
percent_optimal_action_greedy = np.zeros(time_steps) # percentage of optimal actions 

for problem_num in range(1, bandit_problems + 1):
    
    
    # determine actual mean for each bandit in each problem
    q_star = np.random.normal(mu,sigma,K)
    sample_q_star = np.zeros(10) # for storing the sample means
    occurance_bandit_selected = np.zeros(10) # to keep a tab on how many times each bandit has been selected
    
    optimal_action = np.argmax(q_star) # determning optimal action for this particular bandit
    
    
    # now we start each bandit problem
    for t in range(1, time_steps + 1):
        A_t = np.argmax(sample_q_star) # the greedy selection of best bandit
        R_t = np.random.normal(q_star[A_t],reward_sigma) # reward obtained on pulling the greedily selected bandit
        sample_q_star[A_t] = (sample_q_star[A_t]*occurance_bandit_selected[A_t] + R_t)/(occurance_bandit_selected[A_t] + 1) # new sample average 
        occurance_bandit_selected[A_t] = occurance_bandit_selected[A_t] + 1  # update the number of occurance
        
        
        # update the avg_reward
        avg_reward_greedy[t-1] = (((problem_num - 1) * avg_reward_greedy[t-1])  +  R_t)/problem_num 
        
        # update of optimal selection tally
        if A_t == optimal_action:
            percent_optimal_action_greedy[t-1] = percent_optimal_action_greedy[t-1] + 1
            
            
percent_optimal_action_greedy = (percent_optimal_action_greedy/bandit_problems)*100        
        








'''
epsilon greedy policy
'''
avg_reward_epsilon_greedy = np.zeros([2,time_steps])
percent_optimal_action_epsilon_greedy = np.zeros([2, time_steps])
epsilon = [0.1, 0.01]   # for exploration


for num in [0,1]:
    for problem_num in range(1, bandit_problems + 1):
        
        
        # determine actual mean for each bandit in each problem
        q_star = np.random.normal(mu,sigma,K)
        sample_q_star = np.zeros(10) # for storing the sample means
        occurance_bandit_selected = np.zeros(10) # to keep a tab on how many times each bandit has been selected
        
        optimal_action = np.argmax(q_star)
        
        # now we start each bandit problem
        temp = bernoulli.rvs(1 - epsilon[num], size = time_steps)
        for t in range(1, time_steps + 1):
            
            if temp[t-1] == 1:
                A_t = np.argmax(sample_q_star) # the greedy selection of best bandit
            else:   
                A_t = random.sample(range(0,10),1)  # uniform selection
                
            R_t = np.random.normal(q_star[A_t],reward_sigma) # reward obtained on pulling the greedily selected bandit
            sample_q_star[A_t] = (sample_q_star[A_t]*occurance_bandit_selected[A_t] + R_t)/(occurance_bandit_selected[A_t] + 1) # new sample average 
            occurance_bandit_selected[A_t] = occurance_bandit_selected[A_t] + 1  # update the number of occurance
            
            
            # update the avg_reward
            avg_reward_epsilon_greedy[num][t-1] = (((problem_num - 1) * avg_reward_epsilon_greedy[num][t-1])  +  R_t)/problem_num 
            
            
    
            # update of optimal selection tally
            if A_t == optimal_action:
                percent_optimal_action_epsilon_greedy[num][t-1] = percent_optimal_action_epsilon_greedy[num][t-1] + 1
                
    

percent_optimal_action_epsilon_greedy = (percent_optimal_action_epsilon_greedy/bandit_problems)*100  

  
    

plt.plot(avg_reward_greedy, label = 'greedy')  
plt.plot(avg_reward_epsilon_greedy[0], label = 'epsilon = 0.1')  
plt.plot(avg_reward_epsilon_greedy[1], label = 'epsilon = 0.01')  
plt.xlabel('Steps')
plt.ylabel('Average Reward')

plt.legend()
plt.show()      



plt.plot(percent_optimal_action_greedy, label = 'greedy')  
plt.plot(percent_optimal_action_epsilon_greedy[0], label = 'epsilon = 0.1')   
plt.plot(percent_optimal_action_epsilon_greedy[1], label = 'epsilon = 0.01')         
plt.xlabel('Steps')
plt.ylabel('% optimal action')

plt.legend()
plt.show()

