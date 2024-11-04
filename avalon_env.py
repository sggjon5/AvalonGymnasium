# -*- coding: utf-8 -*-
"""

A custom gymnasium environment that simualates an 8 player setup of the Avalon 
board game.

Full game rules can be found here: 

Players included are:
    
    'Merlin',      # Good
    'Percival',    # Good
    'Loyal Servant',  # Good
    'Loyal Servant',  # Good
    'Loyal Servant',  # Good
    'Assassin',    # Evil
    'Mordred',     # Evil
    'Minion',      # Evil
    
    Merlin - knows who the evil players are, except Mordred
    Percival - knows who Merlin is
    Assassin - knows other evil players, except Mordred
    Minion - knows other evil players, except Mordred

@author: sggjone5
"""

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env

class AvalonEnv(gym.Env):
    """
    A custom gymnasium environment that simulates the Avalon board game.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players=8):
        super().__init__()
        self.num_players = num_players
        self.num_rounds = 5
        self.current_round = 0
        self.leader = 0  # Index of the current leader
        
        self.mission_history = np.zeros(self.num_rounds) # need to be wary of default 0
        
        self.successful_missions = 0
        self.failed_missions = 0
        
        # Phases: 'proposal', 'voting', 'mission', 'assassination', 'game_over'
        self.phase = 'proposal'  
        self.rendering_phase = 'proposal'

        # Define mission sizes for each round (standard for 8 players)
        self.mission_sizes = [3, 4, 4, 5, 5]

        # For Mission 4, two fails are required to fail the mission
        self.two_fails_required_round = 3  # zero-indexed

        # agents indexes to use for initialising agents
        self.agents = [i for i in range(self.num_players)]  # all players in the game
       
        # Assign roles
        self.roles = np.zeros(self.num_players, dtype=object)
        self.assign_roles()
        
        # boolean of assassin success
        self.assassin_kill = False

        # initalise action and observation spaces for an individual agent
        proposal_actions = spaces.MultiBinary(self.num_players) # 0 not selected, 1 selected
        voting_actions = spaces.Discrete(2) # 0 reject, 1 accept
        mission_actions = spaces.Discrete(2) # 0 fail, 1 succeed
        assassination_actions = spaces.MultiBinary(self.num_players) # 0 not selected, 1 selected, will use action masking to prevent self assassination 
        
        # need to think about how to make action space consistent across all phases
        # of the game, if using one agent across all four phases.
        # most likely do 8 * 4 length action space and look into action masking?
        
        total_action_space_length = self.num_players + 1 + 1 + self.num_players
        
        self.action_space = spaces.MultiBinary(total_action_space_length)
        
        self.observation_space = spaces.Dict({
            'phase': spaces.Discrete(5),
            'current_round': spaces.Discrete(self.num_rounds),
            'leader': spaces.Discrete(self.num_players),
            'proposed_team': spaces.MultiBinary(self.num_players),
            'votes': spaces.MultiBinary(self.num_players),
            'votes_history': spaces.MultiBinary((self.num_rounds, self.num_players)),
            'mission_history': spaces.MultiBinary(self.num_rounds),
            'successful_missions': spaces.Discrete(self.num_rounds + 1),
            'failed_missions': spaces.Discrete(self.num_rounds + 1),
            'num_players': spaces.Discrete(self.num_players + 1),
            'mission_size': spaces.Discrete(max(self.mission_sizes)+1)
        })
        
        self.dones = False # determines whether the game is over or not
        self.info = {}
        self.truncated = {}
        self.reset()

    def assign_roles(self):
        """
        Randomly assign specific roles to players.
        """
        roles_list = [
            'Merlin',      # Good
            'Percival',    # Good
            'Loyal Servant',  # Good
            'Loyal Servant',  # Good
            'Loyal Servant',  # Good
            'Assassin',    # Evil
            'Mordred',     # Evil
            'Minion',      # Evil
        ]
        random.shuffle(roles_list)
        self.roles = np.array(roles_list)
        
        self.assassin_idx = np.where( self.roles == 'Assassin')[0][0]

        # Creating the mappings
        self.good_roles = ['Merlin', 'Percival', 'Loyal Servant']
        self.evil_roles = ['Assassin', 'Mordred', 'Minion']

        # Determine who knows whose identity
        self.information = {}
        

    def reset(self, seed = 0):
        """
        Reset the environment to the initial state.
        """
        # reset the round and the leader
        self.current_round = 0
        self.leader = 0
        
        # Reset mission and votes history
        self.mission_history = np.zeros(self.num_rounds, dtype=np.int8)
        
        
        # reset counts of successful and failed missions
        self.successful_missions = 0
        self.failed_missions = 0
        
        # set the initial phase
        self.phase = 'proposal'
        self.rendering_phase = 'proposal'
        
        # reset proposed team and votes to prepare for new proposal
        self.proposed_team = np.zeros(self.num_players, dtype=np.int8)
        self.votes = np.zeros(self.num_players, dtype=np.int8)
        self.votes_history = np.zeros((self.num_rounds, self.num_players), dtype=np.int8)
        
        # reassign roles for each platey
        self.assign_roles()
    
        # reset the observvations for an agent
        self.observation = {
                'phase': self.phase_to_int(self.phase),
                'current_round': self.current_round,
                'leader': self.leader,
                'proposed_team': self.proposed_team.copy(),
                'votes': self.votes.copy(),
                'votes_history' : self.votes_history.copy(),
                'mission_history': self.mission_history.copy(),
                'successful_missions': self.successful_missions,
                'failed_missions': self.failed_missions,
                'num_players': self.num_players,
                'mission_size': self.mission_sizes[self.current_round]
            }
        
        
        
        self.info = {}
        self.rewards = {}
    
        
        return self.observation, self.info

    def _get_secret_info(self, agent_name):
        """
        Returns the secret information of the game, tailored to the player role.
        
        There will be a better way to encode this information, but for now it is 
        returned as a list of equal length to the amount of players, where the
        known player identities are in the index of that player. For example,
        if you know the Assassin's identity and they are at index 0
        
        ['Assassin', '', '', '', ..., '']
        """
        
        secret_knowledge = [''] * self.num_players
        
        # Merlin knows all evil except Mordred
        if agent_name == 'Merlin':
 
            evil_idxs = np.where((self.roles == 'Assassin') | (self.roles == 'Minion'))[0]
    
            # Set the names 'Assassin' or 'Minion' at those indices
            for idx in evil_idxs:
                secret_knowledge[idx] = self.roles[idx]
            
            
        if agent_name == 'Percival':
            
            merlin_idx = np.where(self.roles == 'Merlin')[0][0]
            
            secret_knowledge[merlin_idx] = 'Merlin'
            
        
        
        if agent_name == 'Assassin':
            
            minion_idx = np.where(self.roles == 'Minion')[0][0]
            
            secret_knowledge[minion_idx] = 'Minion'
            
        
        if agent_name == 'Minion':
            
            assassin_idx = np.where(self.roles == 'Assassin')[0][0]
            
            secret_knowledge[assassin_idx] = 'Assassin'
            
            
        return secret_knowledge  
   



    def phase_to_int(self, phase):
        phase_dict = {
            'proposal': 0,
            'voting': 1,
            'mission': 2,
            'assassination': 3,
            'game_over': 4,
        }
        return phase_dict[phase]
    
    

    def step(self, action):
        """
        Execute one time step within the environment. Operates on a per phase basis, not per player.
        If actions from multiple players are required, they are collated elsewhere before being passed
        to the step function.
        """
        dones = self.dones
        truncated = self.truncated
        info = self.info
        
        # action comes in as a (8,) for proposal
        if self.phase == 'proposal':
            
            self.rendering_phase = 'proposal'
        
            mission_size = self.mission_sizes[self.current_round]
            
            if np.sum(action) != mission_size:
                raise ValueError('Invalid team size proposed.')
            
            # retrive the index of the proposed team
            proposed_team = action
            
            
            self.proposed_team = proposed_team

            # Move to voting phase
            self.phase = 'voting'
            
        
            observation = {
                        'phase': self.phase,
                        'current_round': self.current_round,
                        'leader': self.leader,
                        'proposed_team': self.proposed_team.copy(),
                        'votes': self.votes.copy(),
                        'votes_history': self.votes_history.copy(),
                        'mission_history': self.mission_history.copy(),
                        'successful_missions': self.successful_missions,
                        'failed_missions': self.failed_missions,
                        'num_players': self.num_players,
                        'mission_size': self.mission_sizes[self.current_round]
                    } 
        
        
        # action comes in as a (8,) for voting, all votes processed at once
        elif self.phase == 'voting':
            
            self.rendering_phase = 'voting'
            self.votes = action
            
            # 0 indicating a reject or 1 accept
            
            # if the vote is passed (majority vote accept)
            if np.sum(action) > self.num_players / 2:
                # Team is approved
                self.phase = 'mission'
                
                
                observation = {
                        'phase': self.phase,
                        'current_round': self.current_round,
                        'leader': self.leader,
                        'proposed_team': self.proposed_team.copy(),
                        'votes': self.votes.copy(),
                        'votes_history': self.votes_history.copy(),
                        'mission_history': self.mission_history.copy(),
                        'successful_missions': self.successful_missions,
                        'failed_missions': self.failed_missions,
                        'num_players': self.num_players,
                        'mission_size': self.mission_sizes[self.current_round]
                    } 
                
                
                
            else:
                # Team is rejected; leadership passes to next player
                self.leader = (self.leader + 1) % self.num_players
                self.phase = 'proposal'
                
                observation = {
                        'phase': self.phase,
                        'current_round': self.current_round,
                        'leader': self.leader,
                        'proposed_team': self.proposed_team.copy(),
                        'votes': self.votes.copy(),
                        'votes_history': self.votes_history.copy(),
                        'mission_history': self.mission_history.copy(),
                        'successful_missions': self.successful_missions,
                        'failed_missions': self.failed_missions,
                        'num_players': self.num_players,
                        'mission_size': self.mission_sizes[self.current_round]
                    } 
                
                
              
        elif self.phase == 'mission':
            
             self.rendering_phase = 'mission'

             self.current_mission_actions = action
             
             # count how many votes for mission failure
             fail_votes = np.sum(action)
             
             # dealing with round that requires two fails
             if self.current_round == self.two_fails_required_round:
                 if fail_votes >= 2:
                     self.failed_missions += 1
                     mission_result = 'Fail'
                 else:
                     self.successful_missions += 1
                     mission_result = 'Success'
             
                # otherwise proceed as normal round
                else:
                 if fail_votes >= 1:
                     self.failed_missions += 1
                     mission_result = 'Fail'
                 else:
                     self.successful_missions += 1
                     mission_result = 'Success'
                    
             # Update game state
             self.current_round += 1
             self.leader = (self.leader + 1) % self.num_players 
             
             
             # as we have already updated the round, if it is drawed at 2, 2
             # we must (un)update the round counter. This is not a great way to
             # handle this but will do for now.
             if self.successful_missions == 2 and self.failed_missions == 2:
                 self.current_round -= 1
                 
             
             # Check for game end conditions
             if self.successful_missions >= 3 or self.failed_missions >= 3: 
                 
                 if self.successful_missions >= 3:
                     # Proceed to assassination phase
                     self.phase = 'assassination'
                     
                 else:
                     # Evil team wins
                     self.phase = 'game_over'
                     self.rewards = self.calculate_rewards(evil_win=True)
                     
                 observation = {
                       'phase': self.phase,
                       'current_round': self.current_round,
                       'leader': self.leader,
                       'proposed_team': self.proposed_team.copy(),
                       'votes': self.votes.copy(),
                       'votes_history': self.votes_history.copy(),
                       'mission_history': self.mission_history.copy(),
                       'successful_missions': self.successful_missions,
                       'failed_missions': self.failed_missions,
                       'num_players': self.num_players,
                       'mission_size': self.mission_sizes[self.current_round]
                   }
                     
                     
             else:
                 # Proceed to next proposal phase
                 self.phase = 'proposal'
                 
                 observation = {
                        'phase': self.phase,
                        'current_round': self.current_round,
                        'leader': self.leader,
                        'proposed_team': self.proposed_team.copy(),
                        'votes': self.votes.copy(),
                        'votes_history': self.votes_history.copy(),
                        'mission_history': self.mission_history.copy(),
                        'successful_missions': self.successful_missions,
                        'failed_missions': self.failed_missions,
                        'num_players': self.num_players,
                        'mission_size': self.mission_sizes[self.current_round]
                    }
             
        elif self.phase == 'assassination':
            
            self.rendering_phase = 'assassination'
            
            
            if np.sum(action) == 0: # if there is no action proposed, then the state does not change
                
                # so do nothing
                observation = {
                        'phase': self.phase,
                        'current_round': self.current_round,
                        'leader': self.leader,
                        'proposed_team': self.proposed_team.copy(),
                        'votes': self.votes.copy(),
                        'votes_history': self.votes_history.copy(),
                        'mission_history': self.mission_history.copy(),
                        'successful_missions': self.successful_missions,
                        'failed_missions': self.failed_missions,
                        'num_players': self.num_players,
                        'mission_size': self.mission_sizes[self.current_round]
                    } 
                
            
            # otherwise, the assassin has made a guess, and now its needs processing
            else:
                # find where merlin is
                merlin_idx = np.where(self.roles == 'Merlin')[0][0]
                
                if merlin_idx == np.where(action == 1)[0]:
                    # Assassin guessed correctly
                    self.phase = 'game_over'
                    self.rewards = self.calculate_rewards(evil_win=True)
                    self.assassin_kill = True

                else:
                    # Assassin guessed incorrectly
                    self.phase = 'game_over'
                    
                    self.rewards = self.calculate_rewards(evil_win=False)
                   
                observation = {
                        'phase': self.phase,
                        'current_round': self.current_round,
                        'leader': self.leader,
                        'proposed_team': self.proposed_team.copy(),
                        'votes': self.votes.copy(),
                        'votes_history': self.votes_history.copy(),
                        'mission_history': self.mission_history.copy(),
                        'successful_missions': self.successful_missions,
                        'failed_missions': self.failed_missions,
                        'num_players': self.num_players,
                        'mission_size': self.mission_sizes[self.current_round]
                    } 
                
        else:
            # game has ended
            self.dones = True
            self.rendering_phase = 'game_over'
            
            
            observation = {
                    'phase': self.phase,
                    'current_round': self.current_round,
                    'leader': self.leader,
                    'proposed_team': self.proposed_team.copy(),
                    'votes': self.votes.copy(),
                    'votes_history': self.votes_history.copy(),
                    'mission_history': self.mission_history.copy(),
                    'successful_missions': self.successful_missions,
                    'failed_missions': self.failed_missions,
                    'num_players': self.num_players,
                    'mission_size': self.mission_sizes[self.current_round]
                } 
            
        # print out what happened in that round.
        self.render()
        
        return observation, self.rewards, dones, truncated, info
                    
                
    
    def calculate_rewards(self, evil_win):
        """
        Calculate rewards for all players based on the game outcome.
        """
        rewards = {}
        for idx, role in enumerate(self.roles):
            agent = 'player_' + str(idx)
            if role in self.good_roles:
                rewards[agent] = -1 if evil_win else 1
            elif role in self.evil_roles:
                rewards[agent] = 1 if evil_win else -1
            else:
                rewards[agent] = 0  # Neutral roles if any
        return rewards

    def render(self, mode='human'):
        """
        Render the environment's current state.
        """
        print(f"Phase: {self.rendering_phase}")
        print(f"Round: {self.current_round}")
        print(f"Leader: Player {self.leader}")
        print(f"Successful Missions: {self.successful_missions}")
        print(f"Failed Missions: {self.failed_missions}")
        
        if self.rendering_phase == 'proposal':
            print(f"Proposed Team: {getattr(self, 'proposed_team', 'Not proposed yet')}")
            
        elif self.rendering_phase == 'voting':
            print(f"Votes: {self.votes} ")
            
            if np.sum(self.votes) > self.num_players / 2:
                print('Vote passed!')
                
            else:
                print('Vote rejectced!')
                
                
        elif self.rendering_phase == 'mission':
            print(f"Proposed Team: {getattr(self, 'proposed_team', 'Not proposed yet')}")
            print(f"Mission Actions: {self.current_mission_actions}")
            
            if np.sum(self.current_mission_actions) >= 1:
                print('Mission Failed!')
            else:
                print('Mission Succeeded!')
                
        elif self.rendering_phase == 'game_over':
            print(f'Rewards: {self.rewards}')
            print(f'Roles: {self.roles}')
            
            if self.assassin_kill == True:
                print('Evil wins, assination of Merlin successful')
                
            elif self.assassin_kill == False and self.successful_missions >= 3:
                print('Good wins, assassin did not assassinate Merlin')
                
            elif self.failed_missions >= 3:
                print('Evil wins by failing majority missions')
            
        print()

    def close(self):
        """
        Clean up the environment (not used here).
        """
        pass

# Example usage
if __name__ == "__main__":
    env = AvalonEnv()
    
    # as there are multiple phases and certain criteria needs to be met
    # this does not return good results as the size of the proposed mission
    # by chance, will liely not match what is required. These rules are handled
    # elsewhere.
    
    check_env(env)
  