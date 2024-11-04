# -*- coding: utf-8 -*-
"""
An agent class to play the game Avalon.

Currently the actions are random in their nature, with the exception of the
mission phase, in which the good agents are hard coded to succeed missions. This
is not necessarily true but would be expected during usual gameplay. It is currently
hardcoded to allow for more realistic gameplay in the non-learning based simulations
of the game.

@author: sggjone5
"""

import random
import parameters


class agent():
    
    def __init__(self, agent_idx, role, observation, secret_role_knowledge):
        
        self.agent_idx = agent_idx # this will be the agents player index
        self.role = role # this will be the named role
        self.observation = observation # this will include histories
        self.secret_role_knowledge = secret_role_knowledge # this will contain the names at the indexes that they know
        
        
        
        
    def select_action_proposal(self, observation):
        
        # return a [] of length num_players that sums to mission_size
        
        action = [0] * observation['num_players']
        
        selected_player_idx = random.sample(range(observation['num_players']), observation['mission_size'])
        
        for i in selected_player_idx:
            action[i] = 1
            
        return action
        
    
    def select_action_voting(self, observation):
        
        # return a random reject or accept
        
        action = random.sample([0,1], 1)
        
        
        return action[0]
    
    def select_action_mission(self, observation):
        
        # if the player is evil
        if self.role in parameters.evil_roles:
            
            # select randomly either pass or fail
            action = random.sample([0,1], 1)
            
            
        # otherwise they are good, so they should vote pass   
        else:
            action = [0]

        return action[0]
    
    def select_action_assassination(self, observation):
        
        # return a random person to assassinate
        action = [0] * observation['num_players'] 
        
        selected_idx = random.randint(0,7)
        
        action[selected_idx] = 1
        
        
        return action
        
        
        
        