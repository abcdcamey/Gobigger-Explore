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
from config.gobigger_no_spatial_config_debug import main_config
from ding.config import compile_config
from policy.gobigger_policy import DQNPolicy
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from envs import GoBiggerSimpleEnv
from ding.utils import set_pkg_seed
from model import GoBiggerHybridActionSimpleV3
from gobigger.agents import BotAgent
from ding.rl_utils import get_epsilon_greedy_fn
from tensorboardX import SummaryWriter
import torch
import logging
logging.basicConfig(level=logging.DEBUG)

import time
import pdb
import pickle
import argparse

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

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.train = True
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.train = False
    evaluator_env_cfg.match_time = 60 * 10

    collector_env = BaseEnvManager(
        env_fn=[lambda: GoBiggerSimpleEnv(collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    rule_evaluator_env = BaseEnvManager(
        env_fn=[lambda: GoBiggerSimpleEnv(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    rule_evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = GoBiggerHybridActionSimpleV3(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    if ckpt_path is not None:

        f = torch.load(ckpt_path)
        policy.eval_mode.load_state_dict(f)
        logging.debug(f'load model from {ckpt_path}')

    # fw = open('./test_ori2.pkl', "wb")
    # pickle.dump(policy, fw)
    # fw.close()

    # policy.eval_mode.load_state_dict(f)
    # fw = open('./test_0_loaded.pkl', 'wb')
    # pickle.dump(policy, fw)
    # fw.close()

    team_num = cfg.env.team_num
    rule_collect_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]
    rule_eval_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]
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

    # rule_stop_flag, rule_reward, _ = rule_evaluator.eval(
    #     learner.save_checkpoint, learner.train_iter, collector.envstep
    # )




    for k in range(max_iterations):

        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data, _ = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data[0], cur_collector_envstep=collector.envstep)

        train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
        learner.train(train_data, collector.envstep)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='debug')
    parser.add_argument('--ckpt', '-c', help='checkpoint for evaluation')
    args = parser.parse_args()
    main(main_config, ckpt_path = args.ckpt)