#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created in January 2018

Following the tutorial: 
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
"""

import game3x3
import game3x3.Game as game
import game3x3.play as play
import numpy as np

actions = range (4)
states = range(9^9)
Q = np.zeros([9^9, 4]])

learningRate = .8
y = .95
discount = 0.3


num_episodes = 2000

#create lists to contain total rewards and steps per episode
rList = []

for i in range (num_episodes):
    #Reset environment and get first new observation
    s = game.state().copy()
    rALL = 0
    
    play()
    j = 0
    
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily picking from Q table
        a = game3x3.make_greedy_strategy(get_q_values)
        #Get new state and reward from environment
        newState = game.state().copy()
        reward = game.do_action(a)
        
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + learningRate *(reward + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += reward
        s = newState
        
    #jList.append(j)
    rList.append(rAll)


        
