#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created in January 2018

Following the tutorial: 
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
"""

from game3x3 import Game
import numpy as np


def play(q_values):
  """Plays one game. The game goes on as long as it is not game over and as 
        long as it has not reached the tile 64 (2**8).
  
  Args:
    strategy: A function that takes as argument a state and a list of available
        actions and returns an action from the list.

  Returns:
    the Q table (q_values) updated after each action for each state.
  """

  game = Game()
  
  state = game.state()
  game_over = game.game_over()
  score = 0
  learningRate = .8
  y = .95
  
  
  while not game_over and not game.reward == 100:
    
    old_state = state
    state_index = 10**8*old_state[0,0]+10**7*old_state[0,1]+10**6*old_state[0,2]+ 10**5*old_state[1,0]+10**4*old_state[1,1]+10**3*old_state[1,2]+10**2*old_state[2,0]+10**1*old_state[2,1]+10**0*old_state[2,2]
    sorted_actions = np.argsort(q_values[state_index, ])
    action = [a for a in sorted_actions if a in game.available_actions()][-1]
    
    reward = game.do_action(action)
    state = game.state()
    game_over = game.game_over()
    
    q_values[old_state, action] = q_values[old_state, action] + learningRate *(reward + y*np.max(q_values[state,:]) - q_values[old_state, action])
    
    score += reward
  
  #game.print_state()
  return q_values, score, reward


# set the q values to zero to initialize
q_values = np.zeros([10**9, 4])

# if you want to use our trained model, please comment out the initialization 
# to zero (line 55) and uncomment the following lines.
#with open ("q_values_20170110.dat") as input:
#   q_values = np.load(input)
    

#play the game 
num_episodes = 0


#create a list to store all scores
all_scores = []
all_rewards = []
    
#for i in range (num_episodes):
while num_episodes < 1000000:
      
    game = Game()
    
    q_values, score, reward = play(q_values)
    
    all_scores.append(score)
    all_rewards.append(reward)
    
    
    num_episodes += 1
    
    if num_episodes % 10000 == 0:
        print ("Number of episodes: %d" % (num_episodes))
        print ("The highest reward is %d." %(max(all_rewards)))
        print ("The game was won %d times in total, or %5.3f %% of the last 10'000 episodes." %(all_rewards.count(max(all_rewards)), all_rewards[-10000:].count(max(all_rewards))/10000.0))

#with open('q_values_20170110.dat', 'w') as output:
#    np.save(output, q_values)




        