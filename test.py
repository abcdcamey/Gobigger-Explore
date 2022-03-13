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

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_AUDIODRIVER'] = 'dsp'

from gobigger.agents import BotAgent
from gobigger.utils import Border
from gobigger.server import Server
from gobigger.render import RealtimeRender, RealtimePartialRender, EnvRender

logging.basicConfig(level=logging.DEBUG)


def test():
    save_path = './test_video'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    match_time = 6*5

    server = Server(dict(
            team_num=4, # 队伍数量
            player_num_per_team=3, # 每个队伍的玩家数量
            match_time=match_time, # 每场比赛的持续时间,
            save_video=True,
            save_quality='low',
            save_path=save_path
    ))

    render = EnvRender(server.map_width, server.map_height)
    server.set_render(render)
    server.start()
    agents = []
    team_player_names = server.get_team_names()
    team_names = list(team_player_names.keys())
    for index in range(server.team_num):
        try:
            p = importlib.import_module('my_submission.my_submission')
            agents.append(p.MySubmission(team_name=team_names[index],
                                         player_names=team_player_names[team_names[index]]))
        except Exception as e:
            logging.error(''.join(traceback.format_tb(e.__traceback__)))
            logging.error(sys.exc_info()[0])
            logging.error('You must implement `MySubmission` in model_submission.py !')
            exit()

    for i in tqdm(range((1+match_time)*server.action_tick_per_second)):
        obs = server.obs()
        global_state, player_states = obs
        actions = {}
        for agent in agents:
            agent_obs = [global_state, {
                player_name: player_states[player_name] for player_name in agent.player_names
            }]
            actions.update(agent.get_actions(agent_obs))
        finish_flag = server.step(actions=actions)
        if finish_flag:
            logging.debug('Game Over!')
            break
    server.close()
    print('Success!')

def postprocess():
    logging.debug('tar zcf ./submit_file/my_submission.tar.gz my_submission/')
    output = subprocess.getoutput('tar zcf ./submit_file/my_submission.tar.gz my_submission/')
    assert os.path.isfile('./submit_file/my_submission.tar.gz')
    logging.debug('###################################################################')
    logging.debug('#                                                                 #')
    logging.debug('#   Now you can upload my_submission.tar.gz as your submission.   #')
    logging.debug('#                                                                 #')
    logging.debug('###################################################################')

if __name__ == '__main__':
    test()
    postprocess()
