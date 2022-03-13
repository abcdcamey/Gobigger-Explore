# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/3/12 8:38 下午
# Description : 
"""

import os
import sys
import logging
import importlib
import time
import argparse
import requests
import subprocess
from tqdm import tqdm
import traceback
from gobigger.agents import BotAgent
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_AUDIODRIVER'] = 'dsp'

from gobigger.agents import BotAgent
from gobigger.utils import Border
from gobigger.server import Server
from gobigger.render import RealtimeRender, RealtimePartialRender, EnvRender
from config.config import main_config
import random
from ding.utils import set_pkg_seed
from my_submission.policy.demo_bot_policy_v2_3 import MyBotAgent as MyBotAgentV23
from my_submission.policy.demo_bot_policy_v4 import MyBotAgent as MyBotAgentV4
logging.basicConfig(level=logging.DEBUG)


class BaseRulePolicy:

    def __init__(self, team_name, player_names):
        self.team_name = team_name
        self.player_names = player_names
    def get_actions(self, obs):
        '''
        Overview:
            You must implement this function.
        '''
        raise NotImplementedError


class MyRulePolicy(BaseRulePolicy):

    def __init__(self, BotAgent, team_name, player_names):
        super(MyRulePolicy, self).__init__(team_name, player_names)
        self.agents = {}
        for player_name in self.player_names:
            self.agents[player_name] = BotAgent(team_name=team_name, player_name=player_name)

    def get_actions(self, obs):
        global_state, player_states = obs
        actions = {}

        agent_obs = {}
        for k, v in player_states.items():
            if v["team_name"] == self.team_name:
                agent_obs[k] = v

        for player_name, agent in self.agents.items():
            action = agent.step([global_state, agent_obs])
            actions[player_name] = action
        return actions



if __name__ == '__main__':
    seed = 10000
    #seed = random.randint(0, 999999)

    server = Server(main_config.env)
    render = EnvRender(server.map_width, server.map_height)
    server.set_render(render)
    server.reset()

    set_pkg_seed(seed, use_cuda=main_config.policy.cuda)

    # 构造agents
    agents = []
    team_player_names = server.get_team_names()
    team_names = list(team_player_names.keys())

    agents.append(MyRulePolicy(MyBotAgentV23, team_name=team_names[0],
                                 player_names=team_player_names[team_names[0]]))

    for i in range(1, server.team_num):
        agents.append(MyRulePolicy(MyBotAgentV4, team_name=team_names[i],
                                 player_names=team_player_names[team_names[i]]))

    #save_frame_tick_list = [50]
    save_frame_tick_list = []

    for i in tqdm(range((1+main_config.env.match_time*server.action_tick_per_second))):
        obs = server.obs()
        global_state, player_states = obs
        actions = {}
        for agent in agents:
            agent_obs = [global_state, {
                player_name: player_states[player_name] for player_name in agent.player_names
            }]
            actions.update(agent.get_actions(agent_obs))
        if i in save_frame_tick_list:
            finish_flag = server.step(actions=actions, save_frame_full_path=f'./save_path/frame_{i}.pkl')
        else:
            finish_flag = server.step(actions=actions)

        if finish_flag:
            logging.debug('Game Over!')
            break
    server.close()