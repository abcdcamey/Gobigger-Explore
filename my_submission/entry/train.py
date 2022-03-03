# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/1/5 11:39 上午
# Description : 
"""
import os
import sys
sys.path.append('..')
import numpy as np
from ding.envs import BaseEnvManager
import copy
from ding.envs import SyncSubprocessEnvManager
from config.gobigger_no_spatial_config_my_v1 import main_config

from ding.config import compile_config
from policy.gobigger_policy import DQNPolicy
#from policy.my_gobigger_policy_v1 import MyDQNPolicy
from policy.my_gobigger_policy_v2 import MyDQNPolicy

from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from envs import GoBiggerSimpleEnv, MyGoBiggerEnvV1, MyGoBiggerEnvV2
from ding.utils import set_pkg_seed
from model import GoBiggerHybridActionSimpleV3, MyGoBiggerHybridActionV1
from gobigger.agents import BotAgent
from policy.demo_bot_policy_v3 import MyBotAgent as MyBotAgentV3
from policy.demo_bot_policy_v2 import MyBotAgent as MyBotAgentV2
from policy.demo_bot_policy_v1 import MyBotAgent as MyBotAgentV1

from ding.rl_utils import get_epsilon_greedy_fn
from tensorboardX import SummaryWriter
import torch
import logging
logging.basicConfig(level=logging.INFO)
from io import StringIO
import traceback
import time
import pdb
import pickle
import argparse

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
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass

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
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass

class MyRulePolicyV3:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [MyBotAgentV3(str(team_id), str(i)) for i in range(start, end)]

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
        self.bot = [BotAgent(str(i)) for i in range(start, end)]

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


def main(cfg,ckpt_path=None, seed=0, max_iterations=int(1e10)):
    cfg.exp_name = 'my_gobigger-v1'
    #print(ckpt_path)
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        MyDQNPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.train = True
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.train = False

    collector_env = SyncSubprocessEnvManager(
        env_fn=[lambda: MyGoBiggerEnvV2(collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    rule_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: MyGoBiggerEnvV2(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    rule_evaluator_env.seed(seed+10000, dynamic_seed=True)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = MyGoBiggerHybridActionV1(**cfg.policy.model)
    policy = MyDQNPolicy(cfg.policy, model=model)

    if ckpt_path is not None:
        f = torch.load(ckpt_path)
        policy.collect_mode.load_state_dict(f)
        logging.debug(f'load model from {ckpt_path}')


    team_num = cfg.env.team_num
    rule_collect_policy = [MyRulePolicyV2(team_id, cfg.env.player_num_per_team) for team_id in range(1, 2)]+\
                          [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, 1)]+\
                          [MyRulePolicyV3(team_id, cfg.env.player_num_per_team) for team_id in range(2, 4)]
    rule_eval_policy = [MyRulePolicyV2(team_id, cfg.env.player_num_per_team) for team_id in range(1, 2)]+ \
                       [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, 1)] + \
                       [MyRulePolicyV3(team_id, cfg.env.player_num_per_team) for team_id in range(2, 4)]
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner'
    )
    collector = BattleSampleSerialCollector(
        cfg.policy.collect.collector,
        collector_env, [policy.collect_mode] + rule_collect_policy,
        tb_logger,
        exp_name=cfg.exp_name
    )
    rule_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        rule_evaluator_env, [policy.eval_mode] + rule_eval_policy,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='rule_evaluator'
    )
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, exp_name=cfg.exp_name)

    for k in range(max_iterations):
        if learner.train_iter>=20000 and rule_evaluator.should_eval(learner.train_iter):
            rule_stop_flag, rule_reward, _ = rule_evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep
            )
            if rule_stop_flag:
                break

        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data, _ = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data[0], cur_collector_envstep=collector.envstep)

        for i in range(cfg.policy.learn.update_per_collect):

            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector.envstep)

        torch.cuda.empty_cache()
        logging.info(f"iterations:{k+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='debug')
    parser.add_argument('--ckpt', '-c', help='checkpoint for evaluation')
    args = parser.parse_args()
    main(main_config, ckpt_path = args.ckpt)
