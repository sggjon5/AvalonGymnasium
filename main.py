# -*- coding: utf-8 -*-
"""
The main file to run the board game Avalon.

imports the environment and eight copies of the agent class, to then
play the game.

parameters only stores common knowledge such as which roles are associatated 
with good and evil within the game.

The game is considered to have four phases (five including game over)

- proposal
    Where the leader proposes a team of the amount of players required for the round
    
- voting
    Every player in the game votes to accept or reject the proposed team. If
    accepted by majority vote, move to mission. If rejected, leadership moves on 
    and proposal phase begins again.
    
- mission
    The accepted proposed team anonymously vote to either succeed or fail
    the mission.

- assassination
    If the majority of rounds have been won by good, the Assassin (evil) gets
    the chance to steal the win by selecting who they believe to be Merlin.
    

The future work of this gymnasium setuup is to introduce reinforcement learning
in some capacity for each of the agents behaviours. Current thoughts are:
    
    - Initially, 1 RL agent trained against 7 random agents with fixed roles.
    - Either 1 RL agent to compelte all phases or four 'sub' agents within one
      main agent, each specialising in a particular phase, I think this would
      introduce a multi-agent approach to the problem.
    - Introduce communcication rounds into the game, utiling LLMs to generate
      conversation between the agents, with the agenda driven by the RL agent.
    
    


@author: George
"""

import numpy as np

from avalon_env import AvalonEnv
from agents import agent

env = AvalonEnv()

observation, _ = env.reset()

player_models = [agent(env.agents[i], env.roles[i], observation, {}) for i in env.agents]

# setting the secret knowldege of each agent

for player in player_models:
    player.secret_role_knowledge = env._get_secret_info(player.role)


while env.dones == False:
    
    
    if env.phase == 'proposal':
        
        actions = []
        
        # loop thgrough all players
        for player in player_models:
            
            
            # if the agent is the leader
            if env.leader == player.agent_idx:
                # let them propose a team
                action = player.select_action_proposal(observation) # will be a list of size env.num_players
                
                # otherwise, vote a 0 (i.e. no vote)
            else:
                action = [0] * env.num_players
                
            actions.append(action)
        
        action = np.array(actions).sum(axis = 0)
            
        
    elif env.phase == 'voting':
        
        actions = []
        # collect all the individual votes before doing the step
        for player in player_models:
            
            action = player.select_action_voting(observation)
            
            # collate the actions from each agent into one action in some way
            actions.append(action)
            
            
        action = np.array(actions)
        
    
    elif env.phase == 'mission':
        
        actions = [0] * env.num_players
        
        for i, player in enumerate(player_models):
            
            selected_player_idx = np.where(observation['proposed_team'] == 1)[0]
            
            # if the agent is in the selected team
            if player.agent_idx in np.where(observation['proposed_team'] == 1)[0]:
                
                action = player.select_action_mission(observation)
                
            else:
                action = 0 # always vote pass if not on the team
                
            actions[i] = action
    
        action = np.array(actions)
        
        
    elif env.phase == 'assassination':
                    
        actions = []
        
        # loop thgrough all players
        for player in player_models:
            
            
            # if the agent is the leader
            if player.agent_idx == env.assassin_idx:
                # let them prpose a kill
                
                action = player.select_action_assassination(observation) # will be a list of size env.num_players
                
                # otherwise, vote a 0 (i.e. no vote)
            else:
                action = [0] * env.num_players
                
            actions.append(action)
        
        action = np.array(actions).sum(axis = 0)
    
    observation, reward, terminated, truncated, info = env.step(action)
        








