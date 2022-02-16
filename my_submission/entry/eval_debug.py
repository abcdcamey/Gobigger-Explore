# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/2/14 4:55 下午
# Description : 
"""
import os
import numpy as np
import copy
from tensorboardX import SummaryWriter
import sys

sys.path.append('..')

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from policy.gobigger_policy import DQNPolicy
from policy.my_gobigger_policy_v1 import MyDQNPolicy
from ding.utils import set_pkg_seed
#from gobigger.agents import BotAgent
import random
from envs import GoBiggerSimpleEnv, MyGoBiggerEnvV1
from model import GoBiggerHybridActionSimpleV3, MyGoBiggerHybridActionV1
from config.gobigger_no_spatial_config_my_v1 import main_config
import torch
import argparse
from policy.demo_bot_policy_v1 import MyBotAgent as MyBotAgentV1
from policy.demo_bot_policy_v2 import MyBotAgent as MyBotAgentV2
import logging


class MyRulePolicyV2:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [MyBotAgentV2(str(team_id), str(i)) for i in range(start, end)]

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, data[env_id]['collate_ignore_raw_obs']):
                global_state = raw_obs["global_state"]

                player_state = {bot.player_name: {"rectangle": raw_obs["rectangle"], "overlap": raw_obs["overlap"],"team_name":raw_obs["team_name"]}}
                obs = (global_state, player_state)
                #raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])]  for x in raw_obs['overlap']['clone']]
                action.append(bot.step(obs))
            #print(env_id, self.team_id, action)
            ret[env_id] = {'action': np.array(action)}

        return ret

    def reset(self, data_id: list = []) -> None:
        pass

class MyRulePolicyV1:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [MyBotAgentV1(str(team_id), str(i)) for i in range(start, end)]

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, data[env_id]['collate_ignore_raw_obs']):
                global_state = raw_obs["global_state"]

                player_state = {bot.player_name: {"rectangle": raw_obs["rectangle"], "overlap": raw_obs["overlap"],"team_name":raw_obs["team_name"]}}
                obs = (global_state, player_state)
                #raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])]  for x in raw_obs['overlap']['clone']]
                action.append(bot.step(obs))
            #print(env_id, self.team_id, action)
            ret[env_id] = {'action': np.array(action)}

        return ret

    def reset(self, data_id: list = []) -> None:
        pass


class RulePolicy:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [MyBotAgentV1(str(i)) for i in range(start, end)]

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, data[env_id]['collate_ignore_raw_obs']):
                raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])]  for x in raw_obs['overlap']['clone']]
                action.append(bot.step(raw_obs))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


def main(cfg, ckpt_path, seed=0):
    logging.info(f"eval start, seed:{seed}")
    # Evaluator Setting
    cfg.exp_name = 'gobigger_vsbot_eval'
    cfg.env.spatial = False  # necessary
    cfg.env.evaluator_env_num = 1
    cfg.env.n_evaluator_episode = 3

    cfg = compile_config(
        cfg,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )

    evaluator_env_num = cfg.env.evaluator_env_num

    rule_env_cfgs = []
    for i in range(evaluator_env_num):
        rule_env_cfg = copy.deepcopy(cfg.env)

        rule_env_cfg.train = False
        # if i==0:
        rule_env_cfg.save_video = True
        # else:
        # rule_env_cfg.save_video = False

        rule_env_cfg.save_quality = 'low'
        rule_env_cfg.save_path = './{}/rule'.format(cfg.exp_name)
        rule_env_cfg.match_time = 60 * 10
        if not os.path.exists(rule_env_cfg.save_path):
            os.makedirs(rule_env_cfg.save_path)
        rule_env_cfgs.append(rule_env_cfg)

    rule_evaluator_env = BaseEnvManager(
        env_fn=[lambda: MyGoBiggerEnvV1(x) for x in rule_env_cfgs], cfg=cfg.env.manager
    )

    rule_evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = MyGoBiggerHybridActionV1(**cfg.policy.model)
    policy = MyDQNPolicy(cfg.policy, model=model)

    if ckpt_path is not None:
        f = torch.load(ckpt_path,map_location='cpu')
        policy.collect_mode.load_state_dict(f)
        policy.learn_mode.load_state_dict(f)
        policy.eval_mode.load_state_dict(f)
        print(f'load model from {ckpt_path}')

    team_num = cfg.env.team_num
    rule_eval_policy = [MyRulePolicyV1(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    rule_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        rule_evaluator_env, [policy.eval_mode] + rule_eval_policy,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='rule_evaluator'
    )
    rule_evaluator.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--ckpt', '-c', help='checkpoint for evaluation')
    args = parser.parse_args()
    seed = random.randint(0, 999999)
    main(main_config, ckpt_path=args.ckpt, seed=seed)
