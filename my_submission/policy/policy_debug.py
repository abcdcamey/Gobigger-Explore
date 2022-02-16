# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/1/18 1:33 下午
# Description : 
"""

import os
import sys
sys.path.append('..')
import logging
import importlib
import time
import argparse
import requests
import subprocess
from tqdm import tqdm
import traceback
from demo_bot_policy_v2 import MyBotAgent
#from bot_policy import MyBotAgent
from gobigger.utils import Border
from gobigger.server import Server
from gobigger.render import RealtimeRender, RealtimePartialRender, EnvRender
from collections import defaultdict

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关
logging.basicConfig(level=logging.DEBUG)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/Logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)

log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)

def test():
    save_path = './video'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    server = Server(dict(
            team_num=4, # 队伍数量
            player_num_per_team=3, # 每个队伍的玩家数量
            match_time=60*5, # 每场比赛的持续时间,
            save_video=True,
            save_quality='low',
            save_path=save_path,
            spatial=False,
            speed=True
            #save_bin=True,

    ))

    render = EnvRender(server.map_width, server.map_height)
    server.set_render(render)
    server.seed(0)
    server.start()
    agents = []
    team_player_names = server.get_team_names()
    team_names = list(team_player_names.keys())
    for index in range(server.team_num):
        try:
            if index==0:
                for j in range(server.player_num_per_team):
                    agents.append(MyBotAgent(str(index), str(index*server.player_num_per_team+j)))
            else:
                for j in range(server.player_num_per_team):
                    agents.append(MyBotAgent(str(index), str(index*server.player_num_per_team+j)))
        except Exception as e:
            logging.error(''.join(traceback.format_tb(e.__traceback__)))
            logging.error(sys.exc_info()[0])
            exit()
    # agents.append(MyBotAgent(str(0), str(0*server.player_num_per_team+0)))
    # agents.append(BotAgent(team_name="1", player_name="1"))

    pre_score = 0
    for i in tqdm(range((1+server.match_time)*server.action_tick_per_second)):
        obs = server.obs()
        global_state, player_states = obs
        total_time = global_state['total_time']
        last_time = global_state['last_time']
        rest_time = total_time - last_time
        leaderboard = global_state['leaderboard']
        logging.info("-------------------------------------")
        logging.info(f"last_time:{last_time},rest_time:{rest_time},leaderboard:{leaderboard}")
        now_score = int(leaderboard["0"])
        if (now_score+8)<pre_score:
            logging.debug(f"last_time:{last_time},score_change:{pre_score-now_score},pre_score:{pre_score},now_score{now_score}")
        pre_score = now_score
        actions = {}
        for agent in agents:
            agent_obs = [global_state, {
                v: player_states[v] for v in team_player_names[agent.team_name]}]
            #print(agent_obs)
            action = agent.step(agent_obs)
            actions.update({agent.player_name: action})
        logging.debug(f"actions:{actions}")

        finish_flag = server.step(actions=actions)
        if finish_flag:
            logging.debug('Game Over!')
            break
    server.close()
    print('Success!')

if __name__ == '__main__':
    #{v: player_states[v] for v in ['0','1']}
    #test()
    from obs_file.action_obs import *
    agent = MyBotAgent(str(3), str(11))
    agent_obs = action7_obs
    action = agent.step(agent_obs)
    # print(action)