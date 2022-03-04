# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/2/12 11:02 上午
# Description : 
"""

import copy
import math
from collections import OrderedDict
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from .gobigger_env import GoBiggerEnv
from collections import defaultdict
import logging
from typing import Any, List, Union, Optional, Tuple


def unit_id(unit_player, unit_team, ego_player, ego_team, team_size):
    '''
    :return: 把player_name和team_name由[0~team_size*team_id-1,0~team_id-1] 转换为 [0~team_size-1,0~team_id-1]
    和ego_player和ego_team相同的clone永远为[0,0],其他的player_id和team_id依次+1或者保持不变
    '''
    unit_player, unit_team, ego_player, ego_team = int(unit_player) % team_size, int(unit_team), int(ego_player) % team_size, int(ego_team)
    # The ego team's id is always 0, enemey teams' ids are 1,2,...,team_num-1
    # The ego player's id is always 0, allies' ids are 1,2,...,player_num_per_team-1
    if unit_team != ego_team:
        player_id = unit_player
        team_id = unit_team if unit_team > ego_team else unit_team + 1
    else:
        if unit_player != ego_player:
            player_id = unit_player if unit_player > ego_player else unit_player + 1
        else:
            player_id = 0
        team_id = 0

    return [team_id, player_id]

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
#case:
# 自己1,0 其他 2,0->2,0;   0,0->1,0   ;    3,1-> 0,1;
# 自己10,3 其他 11,3->2,0  ;9,3->1,0  ;3,1-> 0,2
#print(unit_id(3,1,10,3,3))



@ENV_REGISTRY.register('gobigger_simple',force_overwrite=True)
class MyGoBiggerEnvV2(GoBiggerEnv):
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._player_num_per_team = cfg.player_num_per_team
        self._team_num = cfg.team_num
        self._player_num = self._player_num_per_team * self._team_num
        self._match_time = cfg.match_time
        self._map_height = cfg.map_height
        self._map_width = cfg.map_width
        self._spatial = cfg.spatial
        self._train = cfg.train
        self._last_team_size = None
        self._init_flag = False
        self._speed = cfg.speed
        self._all_vision = cfg.all_vision
        self._cfg['obs_settings'] = dict(
                with_spatial=self._spatial,
                with_speed=self._speed,
                with_all_vision=self._all_vision)
        self._last_player_size = None

    def _unit_id(self, unit_player, unit_team, ego_player, ego_team, team_size):
        return unit_id(unit_player, unit_team, ego_player, ego_team, team_size) # 似乎都返回[0,0]


    def _obs_transform(self, obs: tuple) -> list:
        global_state, player_state = obs
        player_state = OrderedDict(player_state)

        # global
        total_time = global_state['total_time']
        last_time = global_state['last_time']
        rest_time = total_time - last_time
        map_width, map_height = global_state['border']
        # player
        obs = []
        for n, value in player_state.items():
            # scalar feat
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = value['rectangle'] #视野矩形框,默认300*300
            center_x, center_y = (left_top_x + right_bottom_x) / 2, (left_top_y + right_bottom_y) / 2 #视野的中心点
            left_margin, right_margin = left_top_x, self._map_width - right_bottom_x
            top_margin, bottom_margin = left_top_y, self._map_height - right_bottom_y
            scalar_obs = np.array([rest_time / 600, left_margin / 1000, right_margin / 1000, top_margin / 1000, bottom_margin / 1000])  # dim = 5

            team_id, player_id = self._unit_id(n, value['team_name'], n, value['team_name'], self._player_num_per_team) #always [0,0]
            team_name = value['team_name']
            overlap = value['overlap']

            foods = overlap['food']+overlap['spore']
            fake_thorn = np.array([[center_x, center_y, 0]]) if not self._speed else np.array(
                [[center_x, center_y, 0, 0, 0]])
            fake_clone = np.array([[center_x, center_y, 0, 0, 0]]) if not self._speed else np.array(
                [[center_x, center_y, 0, 0, 0, 0, 0]])
            thorns = np.array(overlap['thorns']) if len(overlap['thorns']) > 0 else fake_thorn #二维数组
            clones = np.array([[*x[:-2], *self._unit_id(x[-2], x[-1], n, value['team_name'], self._player_num_per_team)] for x in overlap['clone']]) if len(overlap['clone']) > 0 else fake_clone

            #remove speed info
            overlap['spore'] = [x[:3] for x in overlap['spore']]
            overlap['thorns'] = [x[:3] for x in overlap['thorns']]
            overlap['clone'] = [[*x[:3], int(x[-2]), int(x[-1])] for x in overlap['clone']]

            # encode units
            food_map, clone_with_food_relation = food_encode(clones, foods, left_top_x, left_top_y, right_bottom_x, right_bottom_y, team_id, player_id)

            clone_ego = np.where((clones[:, -2] == team_id) & (clones[:, -1] == player_id))
            clone_ego = clones[clone_ego]

            clone_ego = clone_ego[clone_ego[:,2].argsort()[::-1]]

            clone_team = np.where((clones[:, -2] == team_id) & (clones[:, -1] != player_id))
            clone_team = clones[clone_team]

            clone_enemy = np.where((clones[:, -2] != team_id) | (clones[:, -1] != player_id))
            clone_enemy = clones[clone_enemy]

            if clone_team.size == 0:
                clone_team = np.array([[center_x, center_y, 0, 0, 0]]) if not self._speed else np.array([[center_x, center_y, 0, 0, 0, 0, 0]])

            if clone_enemy.size == 0:
                clone_enemy = np.array([[center_x, center_y, 0, 0, 0]]) if not self._speed else np.array([[center_x, center_y, 0, 0, 0, 0, 0]])

            clones = clone_encode(clone_ego, self._map_width, self._map_height)

            clone_with_thorn_relation = relation_thorn_encode(clone_ego, thorns)
            clone_with_team_relation = relation_clone_encode(clone_ego, clone_team)
            clone_with_enemy_relation = relation_clone_encode(clone_ego, clone_enemy)

            player_obs = {
                'scalar': scalar_obs.astype(np.float32),
                'food_map': food_map.astype(np.float32),
                'clone_with_food_relation': clone_with_food_relation.astype(np.float32),
                'clone_with_thorn_relation': clone_with_thorn_relation.astype(np.float32),
                'clones': clones.astype(np.float32),
                'clone_with_team_relation': clone_with_team_relation.astype(np.float32),
                'clone_with_enemy_relation': clone_with_enemy_relation.astype(np.float32),

                # 'collate_ignore_raw_obs': {'overlap': overlap,'player_bot_obs':player_bot_obs},
                # 'collate_ignore_raw_obs': {'overlap': overlap, 'global_state': global_state, 'player_state': {n:value}}
                'collate_ignore_raw_obs': {'overlap': overlap, 'global_state': global_state, 'rectangle': value['rectangle'], 'team_name': team_name}
            }
            obs.append(player_obs)
        team_obs = []
        for i in range(self._team_num):
            team_obs.append(team_obs_stack(obs[i * self._player_num_per_team: (i + 1) * self._player_num_per_team]))
        return team_obs


    def _obs_transform_eval(self, obs: tuple) -> list:
        global_state, player_state = obs
        player_state = OrderedDict(player_state)
        logging.info(global_state)
        # global
        total_time = global_state['total_time']
        last_time = global_state['last_time']
        rest_time = total_time - last_time
        map_width, map_height = global_state['border']
        # player
        obs = []
        for n, value in player_state.items():
            # scalar feat
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = value['rectangle'] #视野矩形框,默认300*300
            center_x, center_y = (left_top_x + right_bottom_x) / 2, (left_top_y + right_bottom_y) / 2 #视野的中心点
            left_margin, right_margin = left_top_x, self._map_width - right_bottom_x
            top_margin, bottom_margin = left_top_y, self._map_height - right_bottom_y
            scalar_obs = np.array([rest_time / 600, left_margin / 1000, right_margin / 1000, top_margin / 1000, bottom_margin / 1000])  # dim = 5

            team_id, player_id = self._unit_id(n, value['team_name'], n, value['team_name'], self._player_num_per_team) #always [0,0]
            team_name = value['team_name']
            overlap = value['overlap']

            foods = overlap['food']+overlap['spore']
            fake_thorn = np.array([[center_x, center_y, 0]]) if not self._speed else np.array(
                [[center_x, center_y, 0, 0, 0]])
            fake_clone = np.array([[center_x, center_y, 0, 0, 0]]) if not self._speed else np.array(
                [[center_x, center_y, 0, 0, 0, 0, 0]])
            thorns = np.array(overlap['thorns']) if len(overlap['thorns']) > 0 else fake_thorn #二维数组
            clones = np.array([[*x[:-2], *self._unit_id(x[-2], x[-1], n, value['team_name'], self._player_num_per_team)] for x in overlap['clone']]) if len(overlap['clone']) > 0 else fake_clone

            #remove speed info
            overlap['spore'] = [x[:3] for x in overlap['spore']]
            overlap['thorns'] = [x[:3] for x in overlap['thorns']]
            overlap['clone'] = [[*x[:3], int(x[-2]), int(x[-1])] for x in overlap['clone']]

            # encode units
            food_map, clone_with_food_relation = food_encode(clones, foods, left_top_x, left_top_y, right_bottom_x, right_bottom_y, team_id, player_id)

            clone_ego = np.where((clones[:, -2] == team_id) & (clones[:, -1] == player_id))
            clone_ego = clones[clone_ego]

            clone_ego = clone_ego[clone_ego[:,2].argsort()[::-1]]

            clone_team = np.where((clones[:, -2] == team_id) & (clones[:, -1] != player_id))
            clone_team = clones[clone_team]

            clone_enemy = np.where((clones[:, -2] != team_id) | (clones[:, -1] != player_id))
            clone_enemy = clones[clone_enemy]

            if clone_team.size == 0:
                clone_team = np.array([[center_x, center_y, 0, 0, 0]]) if not self._speed else np.array([[center_x, center_y, 0, 0, 0, 0, 0]])

            if clone_enemy.size == 0:
                clone_enemy = np.array([[center_x, center_y, 0, 0, 0]]) if not self._speed else np.array([[center_x, center_y, 0, 0, 0, 0, 0]])

            clones = clone_encode(clone_ego, self._map_width, self._map_height)

            clone_with_thorn_relation = relation_thorn_encode(clone_ego, thorns)
            clone_with_team_relation = relation_clone_encode(clone_ego, clone_team)
            clone_with_enemy_relation = relation_clone_encode(clone_ego, clone_enemy)

            player_obs = {
                'scalar': scalar_obs.astype(np.float32),
                'food_map': food_map.astype(np.float32),
                'clone_with_food_relation': clone_with_food_relation.astype(np.float32),
                'clone_with_thorn_relation': clone_with_thorn_relation.astype(np.float32),
                'clones': clones.astype(np.float32),
                'clone_with_team_relation': clone_with_team_relation.astype(np.float32),
                'clone_with_enemy_relation': clone_with_enemy_relation.astype(np.float32),

                # 'collate_ignore_raw_obs': {'overlap': overlap,'player_bot_obs':player_bot_obs},
                # 'collate_ignore_raw_obs': {'overlap': overlap, 'global_state': global_state, 'player_state': {n:value}}
                'collate_ignore_raw_obs': {'overlap': overlap, 'global_state': global_state, 'rectangle': value['rectangle'], 'team_name': team_name}
            }
            obs.append(player_obs)
        team_obs = []

        team_obs.append(team_obs_stack(obs[:self._player_num_per_team]))
        return team_obs

    def step(self, action: list) -> BaseEnvTimestep:
        action = self._act_transform(action)
        done = self._env.step(action)
        raw_obs = self._env.obs()
        obs = self._obs_transform(raw_obs)
        rew = self._get_reward(raw_obs)
        info = [{} for _ in range(self._team_num)]

        for i, team_reward in enumerate(rew):
            self._final_eval_reward[i] += np.array([sum(team_reward)]) #modify
        if done:
            for i in range(self._team_num):
                info[i]['final_eval_reward'] = self._final_eval_reward[i]
                info[i]['leaderboard'] = self._last_team_size
            leaderboard = self._last_team_size
            leaderboard_sorted = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
            win_rate = self.win_rate(leaderboard_sorted)
            print('win_rate:{:.3f}, leaderboard_sorted:{}'.format(win_rate, leaderboard_sorted))
            logging.info('win_rate:{:.3f}, leaderboard_sorted:{}'.format(win_rate, leaderboard_sorted))
        return BaseEnvTimestep(obs, rew, done, info)


    def _get_reward(self, obs: tuple) -> list:
        global_state, player_state = obs
        cur_player_size = defaultdict(int)
        cur_player_cl_num = defaultdict(int)
        for n in self._player_names:
            _clone = player_state[str(n)]['overlap']['clone']
            for cl in _clone:
                if cl[-2] == int(n):
                    cur_player_size[str(n)] = cur_player_size[str(n)] + (cl[2] * cl[2])
                    cur_player_cl_num[str(n)] = cur_player_cl_num[str(n)]+1
        if self._last_player_size is None:
            team_reward = [np.array([0. for _ in range(self._player_num_per_team)]) for __ in range(self._team_num)]
        else:
            reward_single = defaultdict(int)
            reward_team = defaultdict(int)
            reward_cl_num = defaultdict(int)
            reward_weight = [0.6, 0.2, 0.2]
            reward = defaultdict(int)
            for n in self._player_names:
                last_size = self._last_player_size[str(n)]
                cur_size = cur_player_size[str(n)]
                symbol = 1.0 if cur_size >= last_size else -1.0
                reward_single[str(n)] = np.clip(symbol*(math.sqrt(abs(cur_size-last_size)/last_size)), -1, 1)

                team_name = str(int(n) // self._player_num_per_team)
                last_size = self._last_team_size[team_name]
                cur_size = global_state['leaderboard'][team_name]
                symbol = 1.0 if cur_size >= last_size else -1.0
                reward_team[str(n)] = np.clip(symbol*(math.sqrt(abs(cur_size-last_size)/last_size)), -1, 1)

                last_player_cl_num = self._last_player_cl_num[str(n)]
                reward_cl_num[str(n)] = np.clip((last_player_cl_num-cur_player_cl_num[str(n)])/2, -1, 1)

                reward[str(n)] = reward_single[str(n)]*reward_weight[0]+reward_team[str(n)]*reward_weight[1]+\
                                 reward_cl_num[str(n)]*reward_weight[2]

            team_reward = []
            for i in range(self._team_num):
                player_reward = []
                for j in range(i * self._player_num_per_team,(i + 1) * self._player_num_per_team):
                    player_reward.append(reward[str(j)])
                player_reward = np.array(player_reward)
                team_reward.append(player_reward)
        self._last_player_size = cur_player_size
        self._last_team_size = global_state['leaderboard']
        self._last_player_cl_num = cur_player_cl_num
        return team_reward

    @staticmethod
    def _to_raw_action(act: int) -> Tuple[float, float, int]:
        assert 0 <= act < 20
        # -1, 0, 1, 2(noop, eject, split, gather)
        # 0, 1, 2, 3, 4(up, down, left, right, None)
        direction, action_type = act // 4, act % 4
        action_type = action_type - 1
        if direction == 0:
            x, y = 0, 1
        elif direction == 1:
            x, y = 0, -1
        elif direction == 2:
            x, y = -1, 0
        elif direction == 3:
            x, y = 1, 0
        else:
            x, y = None, None
        return [x, y, action_type]

    @staticmethod
    def raw_action_to_int(action_ret: Tuple[float, float, int]) -> int:
        x, y = action_ret[:2]
        if action_ret[0] is None or action_ret[1] is None:
            direction = 4
        elif x==0 and y==1:
            direction = 0
        elif x==0 and y==-1:
            direction = 1
        elif x==-1 and y==0:
            direction = 2
        elif x==1 and y==0:
            direction = 2

        action_type = (action_ret[2]+1)

        return int(direction*4+action_type)
def food_encode(clones, foods, left_top_x, left_top_y, right_bottom_x, right_bottom_y, team_id, player_id):
    # food_map's shape: 2,h,w   2,300/16,300/16
    # food_map[0,:,:] represent food density map
    # food_map[1,:,:] represent cur clone ball density map

    # food_grid's shape: h_,w_ 300/8,300/8
    # food_frid[:] represent food information(x,y,r)
    # food_relation'shape: len(my_clones),7*7+1,3
    # food_relation represent food and clone in 7*7 grid (offset_x, offset_y, r)
    #修改1:只计算my_clones的relation,同时relation取每个grid离clone最近的计算
    my_clones = [clone for clone in clones if clone[-2] == team_id and clone[-1]==player_id ]

    w = (right_bottom_x - left_top_x) // 16 + 1
    h = (right_bottom_y - left_top_y) // 16 + 1
    food_map = np.zeros((2, h, w))

    w_ = (right_bottom_x - left_top_x) // 8 + 1
    h_ = (right_bottom_y - left_top_y) // 8 + 1
    food_grid = [ [ [] for j in range(w_) ] for i in range(h_) ]
    food_relation = np.zeros((len(my_clones), 7 * 7 + 1, 4))

    for p in foods:
        x = min(max(p[0], left_top_x), right_bottom_x) - left_top_x #相对视野框的坐标
        y = min(max(p[1], left_top_y), right_bottom_y) - left_top_y
        radius = p[2]
        # encode food density map
        i, j = int(y // 16), int(x // 16)
        food_map[0, i, j] += radius * radius
        # encode food fine grid
        i, j = int(y // 8), int(x // 8)
        food_grid[i][j].append([(x - 8 * j) / 8, (y - 8 * i) / 8, radius, x , y]) # 在8*8的小单元格内的偏置坐标

    for c_id, p in enumerate(my_clones):
        x = min(max(p[0], left_top_x), right_bottom_x) - left_top_x
        y = min(max(p[1], left_top_y), right_bottom_y) - left_top_y
        radius = p[2]
        # encode food density map
        i, j = int(y // 16), int(x // 16)
        food_map[1, i, j] += radius * radius

        # encode food fine grid
        i, j = int(y // 8), int(x // 8)
        t, b, l, r = max(i - 3, 0), min(i + 4, h_), max(j - 3, 0), min(j + 4, w_)
        for ii in range(t, b):
            for jj in range(l, r):
                for f in food_grid[ii][jj]:
                    if food_relation[c_id][(ii - t) * 7 + jj - l][2] == 0: #保存距离最近的food做为relation
                        food_relation[c_id][(ii - t) * 7 + jj - l][0] = f[0]
                        food_relation[c_id][(ii - t) * 7 + jj - l][1] = f[1]
                        food_relation[c_id][(ii - t) * 7 + jj - l][2] += f[2] * f[2]
                        food_relation[c_id][(ii - t) * 7 + jj - l][3] = (f[3]-x)*(f[3]-x)+(f[4]-y)*(f[4]-y) #clone和food的距离
                    else:
                        new_distance = (f[3]-x)*(f[3]-x)+(f[4]-y)*(f[4]-y)
                        if new_distance < food_relation[c_id][(ii - t) * 7 + jj - l][3]:
                            food_relation[c_id][(ii - t) * 7 + jj - l][0] = f[0]
                            food_relation[c_id][(ii - t) * 7 + jj - l][1] = f[1]
                            food_relation[c_id][(ii - t) * 7 + jj - l][2] += f[2] * f[2]
                            food_relation[c_id][(ii - t) * 7 + jj - l][3] = new_distance

        food_relation[c_id][-1][0] = (x - j * 8) / 8
        food_relation[c_id][-1][1] = (y - i * 8) / 8
        food_relation[c_id][-1][2] = radius / 10

    food_map[0, :, :] = np.sqrt(food_map[0, :, :]) / 2    #food
    food_map[1, :, :] = np.sqrt(food_map[1, :, :]) / 10   #cur clone
    food_relation[:, :-1, 2] = np.sqrt(food_relation[:, :-1, 2]) / 2
    food_relation = food_relation[:, :, :-1]
    food_relation = food_relation.reshape(len(my_clones), -1)
    return food_map, food_relation #dim=2,dim=len(my_clones)




def clone_encode(clones, map_width, map_height):
    pos = clones[:, :2] / 100
    rds = clones[:, 2:3] / 10
    ids = np.zeros((len(clones), 12))
    ids[np.arange(len(clones)), (clones[:, -2] * 3 + clones[:, -1]).astype(np.int64)] = 1.0

    #球离边缘的距离
    left_margin = (clones[:, 0:1]-clones[:, 2:3])/100
    right_margin = (map_width-(clones[:, 0:1]+clones[:, 2:3]))/100
    top_margin = (clones[:, 1:2]-clones[:, 2:3])/100
    down_margin = (map_height-(clones[:, 1:2]+clones[:, 2:3]))/100

    clones = np.concatenate([pos, rds, ids, left_margin, right_margin, top_margin, down_margin], axis=1)  # dim = 19

    return clones




def relation_thorn_encode(point_1, point_2):
    pos_rlt_1 = point_2[None,:,:2] - point_1[:,None,:2] # relative position
    pos_rlt_2 = np.linalg.norm(pos_rlt_1, ord=2, axis=2, keepdims=True) # distance
    pos_rlt_3 = point_1[:,None,2:3] - pos_rlt_2 # whether source collides with target
    pos_rlt_4 = point_2[None,:,2:3] - pos_rlt_2 # whether target collides with source
    pos_rlt_5 = (2 + np.sqrt(0.5)) * point_1[:,None,2:3] - pos_rlt_2 # whether source's split collides with target
    #pos_rlt_6 = (2 + np.sqrt(0.5)) * point_2[None,:,2:3] - pos_rlt_2 # whether target's split collides with source
    rds_rlt_1 = point_1[:,None,2:3] - point_2[None,:,2:3] # whether source can eat target
    rds_rlt_2 = np.sqrt(0.5) * point_1[:,None,2:3] - point_2[None,:,2:3] # whether source's split can eat target
    #rds_rlt_3 = np.sqrt(0.5) * point_2[None,:,2:3] - point_1[:,None,2:3] # whether target's split can eat source
    rds_rlt_4 = point_1[:,None,2:3].repeat(len(point_2), axis=1) # source radius
    rds_rlt_5 = point_2[None,:,2:3].repeat(len(point_1), axis=0) # target radius
    relation = np.concatenate([pos_rlt_1 / 100, pos_rlt_2 / 100, pos_rlt_3 / 100, pos_rlt_4 / 100, pos_rlt_5 / 100, rds_rlt_1 / 10, rds_rlt_2 / 10, rds_rlt_4 / 10, rds_rlt_5 / 10], axis=2)
    return relation


def relation_clone_encode(point_1, point_2):
    pos_rlt_1 = point_2[None,:,:2] - point_1[:,None,:2] # relative position
    pos_rlt_2 = np.linalg.norm(pos_rlt_1, ord=2, axis=2, keepdims=True) # distance
    pos_rlt_3 = point_1[:,None,2:3] - pos_rlt_2 # whether source collides with target
    pos_rlt_4 = point_2[None,:,2:3] - pos_rlt_2 # whether target collides with source
    pos_rlt_5 = (2 + np.sqrt(0.5)) * point_1[:,None,2:3] - pos_rlt_2 # whether source's split collides with target
    pos_rlt_6 = (2 + np.sqrt(0.5)) * point_2[None,:,2:3] - pos_rlt_2 # whether target's split collides with source
    rds_rlt_1 = point_1[:,None,2:3] - point_2[None,:,2:3] # whether source can eat target
    rds_rlt_2 = np.sqrt(0.5) * point_1[:,None,2:3] - point_2[None,:,2:3] # whether source's split can eat target
    rds_rlt_3 = np.sqrt(0.5) * point_2[None,:,2:3] - point_1[:,None,2:3] # whether target's split can eat source
    rds_rlt_4 = point_1[:,None,2:3].repeat(len(point_2), axis=1) # source radius
    rds_rlt_5 = point_2[None,:,2:3].repeat(len(point_1), axis=0) # target radius
    relation = np.concatenate([pos_rlt_1 / 100, pos_rlt_2 / 100, pos_rlt_3 / 100, pos_rlt_4 / 100, pos_rlt_5 / 100, pos_rlt_6 / 100, rds_rlt_1 / 10, rds_rlt_2 / 10, rds_rlt_3 / 10, rds_rlt_4 / 10, rds_rlt_5 / 10], axis=2)
    return relation


def team_obs_stack(team_obs):
   if len(team_obs)>0:
        result = {}
        for k in team_obs[0].keys():
            result[k] = [o[k] for o in team_obs]
        return result
   else:
       return {}