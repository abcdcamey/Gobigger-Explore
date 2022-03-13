# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/1/23 11:27 上午
# Description : 
"""
# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2021/12/14 4:54 下午
# Description : 
"""
import random
import logging
import copy
import queue
from pygame.math import Vector2

from gobigger.agents.base_agent import BaseAgent
import sys, os


curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
import logging
import time
import numpy as np
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
import math
from collections import defaultdict

def distance(position_1,position_2):
    return (position_1 - position_2).length()


def check_split_eat(my_clone_balls, enemy_clone_balls):
    for i in range(len(enemy_clone_balls)):
        enemy_clone = enemy_clone_balls[i]
        dis = (my_clone_balls[0]["position"]-enemy_clone['position']).length()
        if dis>(3*my_clone_balls[0]["radius"]+enemy_clone["radius"]):
            continue
        rl_position = (enemy_clone['position']-my_clone_balls[0]['position']).normalize() # 方向向量
        new_my_clone_balls = []
        for my_clone in my_clone_balls:
            if my_clone["radius"]>=10: #半径大于10才能分裂
                dis = 2*my_clone["radius"]
                new_position = Vector2(my_clone['position'].x+dis*rl_position.x, my_clone['position'].y+dis*rl_position.y)
                new_radius = np.sqrt(0.5)*my_clone["radius"]
                new_my_clone_balls.append({"position": new_position, "radius": new_radius})
        eat_radius_sum = 0
        eated_radius_sum = 0
        dangerous_radius_sum = 0
        for new_clone in new_my_clone_balls:
            for j in range(len(enemy_clone_balls)):
                target_ball = enemy_clone_balls[j]
                _dis = (new_clone["position"]-target_ball["position"]).length()
                if new_clone['radius']>target_ball['radius'] and _dis<=new_clone['radius']:
                    eat_radius_sum = eat_radius_sum +target_ball['radius']
                if new_clone['radius']<target_ball['radius'] and _dis<=target_ball['radius']:
                    eated_radius_sum = eated_radius_sum+new_clone['radius']
                if new_clone['radius']<target_ball['radius'] and _dis<=(new_clone['radius']+target_ball['radius']):
                    dangerous_radius_sum = dangerous_radius_sum+new_clone['radius']
        if eat_radius_sum>dangerous_radius_sum+eated_radius_sum:
            return rl_position
    return None





def check_clone_dangerous(my_clone_balls, enemy_clone_balls):
    dangerous_status = 0

    safe_direction_list = [] # Vector2(0, action_ret[1]).normalize()

    for my_clone_ball in my_clone_balls:
        for enemy_clone_ball in enemy_clone_balls:
            if my_clone_ball["radius"] < enemy_clone_ball["radius"]:
                distance = (my_clone_ball['position'] - enemy_clone_ball['position']).length()
                #if distance<center_distance*0.95 and distance < min((enemy_clone_ball["radius"]+my_clone_ball["radius"])*1.1, (enemy_clone_ball["radius"]+my_clone_ball["radius"])+6):
                if distance < min((enemy_clone_ball["radius"] + my_clone_ball["radius"]) * 1.1, (enemy_clone_ball["radius"] + my_clone_ball["radius"]) + 6):
                    safe_direction = (my_clone_ball['position'] - enemy_clone_ball['position']).normalize()
                    safe_direction_list.append(safe_direction)
                    dangerous_status = 1
    if dangerous_status>0:
        safe_direction = safe_direction_list[0]
        for i in range(1,len(safe_direction_list)):
            safe_direction = safe_direction+safe_direction_list[i]
        safe_direction = safe_direction.normalize()
        return dangerous_status, safe_direction, len(safe_direction_list)
    else:
        safe_direction_list = []
        #target split dangerous
        my_team_id = my_clone_balls[0]["team"]
        my_player_id = my_clone_balls[0]["player"]
        enemy_clone_balls_dict = defaultdict(list)
        for enemy_clone_ball in enemy_clone_balls:
            enemy_clone_balls_dict[enemy_clone_ball["player"]].append(enemy_clone_ball)
        for i in range(min(len(my_clone_balls),2)):
            my_clone_ball = my_clone_balls[i]
            for k,v in enemy_clone_balls_dict.items():
                if len(v) >= 2:
                    v.sort(key=lambda a: a['radius'], reverse=True)
                    tatget_top1_2_dis = (v[0]["position"]-v[1]["position"]).length()
                    dis = (v[0]["position"]-my_clone_ball["position"]).length()
                    target_top_area = sum([v[i]["radius"]*v[i]["radius"] for i in range(0,min(2,len(v)))])
                    target_radius = math.sqrt(target_top_area)
                    if tatget_top1_2_dis <= (v[0]["radius"]+v[1]["radius"]) and \
                        target_radius>my_clone_ball["radius"] and target_radius*3 > dis:
                        dangerous_status = 2
                        safe_direction = (my_clone_ball['position'] - v[0]['position']).normalize()
                        safe_direction_list.append(safe_direction)
        if dangerous_status>1:
            safe_direction = safe_direction_list[0]
            for i in range(1, len(safe_direction_list)):
                safe_direction = safe_direction + safe_direction_list[i]
            safe_direction = safe_direction.normalize()
            return dangerous_status, safe_direction, len(safe_direction_list)
        else:
            safe_direction = Vector2(0, 0)
            return dangerous_status, safe_direction,0



def check_edge(global_state, my_clone_balls):
    map_width, map_height = global_state["border"]
    edge_torch = [0, 0, 0, 0]# left top right down
    left_edge = Vector2(0, my_clone_balls[0]['position'].y)
    top_edge = Vector2(my_clone_balls[0]['position'].x, 0)
    right_edge = Vector2(map_width-1, my_clone_balls[0]['position'].y)
    down_edge = Vector2(my_clone_balls[0]['position'].x, map_height-1)

    if (my_clone_balls[0]['position'] - left_edge).length() <= my_clone_balls[0]['radius']+1.5:
        edge_torch[0] = 1
    if (my_clone_balls[0]['position'] - top_edge).length() <= my_clone_balls[0]['radius']+1.5:
        edge_torch[1] = 1
    if (my_clone_balls[0]['position'] - right_edge).length() <= my_clone_balls[0]['radius']+1.5:
        edge_torch[2] = 1
    if (my_clone_balls[0]['position'] - down_edge).length() <= my_clone_balls[0]['radius']+1.5:
        edge_torch[3] = 1

    return edge_torch

def adjust_direction(global_state, my_clone_balls, action_ret):
    if action_ret[2]!=-1:
        return action_ret
    edge_torch = check_edge(global_state, my_clone_balls)
    if action_ret[0] is not None and action_ret[1] is not None:
        if edge_torch[0] > 0 and action_ret[0] < 0 and abs(action_ret[1]) > 0.0001:
            new_direction = Vector2(0.001, action_ret[1]).normalize()
            action_ret[0] = new_direction.x
            action_ret[1] = new_direction.y
        if edge_torch[1] > 0 and action_ret[1] < 0 and abs(action_ret[0]) > 0.0001:
            new_direction = Vector2(action_ret[0], 0.001).normalize()
            action_ret[0] = new_direction.x
            action_ret[1] = new_direction.y
        if edge_torch[2] > 0 and action_ret[0] > 0 and abs(action_ret[1]) > 0.0001:
            new_direction = Vector2(0.001, action_ret[1]).normalize()
            action_ret[0] = new_direction.x
            action_ret[1] = new_direction.y
        if edge_torch[3] > 0 and action_ret[1] > 0 and abs(action_ret[0]) > 0.0001:
            new_direction = Vector2(action_ret[0], 0.001).normalize()
            action_ret[0] = new_direction.x
            action_ret[1] = new_direction.y
        if edge_torch[0]>0 and edge_torch[1]>0 and action_ret[0]<0 and action_ret[1]<0 and len(my_clone_balls)<=2: #被逼到左上角
            if abs(action_ret[0])>abs(action_ret[1]):
                action_ret[0]= 0.0001
                action_ret[1] = 0.99
            else:
                action_ret[0] = 0.99
                action_ret[1] = 0.0001
        if edge_torch[2]>0 and edge_torch[1]>0 and action_ret[0]>0 and action_ret[1]<0 and len(my_clone_balls)<=2:# 右上角
            if abs(action_ret[0])>abs(action_ret[1]):
                action_ret[0]= 0.0001
                action_ret[1] = 0.99
            else:
                action_ret[0] = -0.99
                action_ret[1] = 0.0001
        if edge_torch[2]>0 and edge_torch[3]>0 and action_ret[0]>0 and action_ret[1]>0 and len(my_clone_balls)<=2:# 右下角
            if abs(action_ret[0])>abs(action_ret[1]):
                action_ret[0] = 0.0001
                action_ret[1] = -0.99
            else:
                action_ret[0] = -0.99
                action_ret[1] = 0.0001
        if edge_torch[0]>0 and edge_torch[3]>0 and action_ret[0]<0 and action_ret[1]>0 and len(my_clone_balls)<=2: #左下角
            if abs(action_ret[0])>abs(action_ret[1]):
                action_ret[0] = 0.0001
                action_ret[1] = -0.99
            else:
                action_ret[0] = 0.99
                action_ret[1] = 0.0001
    return action_ret


class MyBotAgent(BaseAgent):
    '''
    Overview:
        A simple script bot
    '''

    def __init__(self, team_name = None, player_name = None, level=3):
        self.team_name = team_name
        self.name = player_name

        self.player_name = player_name
        self.actions_queue = queue.Queue()
        self.last_clone_num = 1
        self.last_total_size = 0
        self.level = level




    def step(self, obs):
        global_state, player_state = obs

        if self.level == 1:
            return self.step_level_1(player_state.get(self.player_name))
        if self.level == 2:
            return self.step_level_2(player_state.get(self.player_name))
        if self.level == 3:
            #logging.info(f"player_name:{self.player_name},time:{global_state.get('last_time')},leaderboard:{global_state.get('leaderboard')}")
            #logging.info(f"overlap:{player_state.get(self.player_name).get('overlap')}")
            action_ret = self.step_level_3(global_state, player_state.get(self.player_name))
            #logging.info(f"action_ret before conv:{action_ret}")
            #logging.info(f"action_ret after conv:{action_ret}")
            return action_ret

    def step_level_1(self, obs):
        if self.actions_queue.qsize() > 0:
            return self.actions_queue.get()
        overlap = obs['overlap']
        overlap = self.preprocess(overlap)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']

        my_clone_balls, others_clone_balls = self.process_clone_balls(clone_balls)
        min_distance, min_food_ball = self.process_food_balls(food_balls, my_clone_balls[0])
        #print("my_clone_balls.radius:", my_clone_balls[0]['radius'])
        if min_food_ball is not None:
            direction = (min_food_ball['position'] - my_clone_balls[0]['position']).normalize()
        else:
            direction = (Vector2(0, 0) - my_clone_balls[0]['position']).normalize()
        action_type = -1
        #print(my_clone_balls[0]['radius'],distance(my_clone_balls[0]['position'], min_food_ball['position']))
        if my_clone_balls[0]['radius']<5.5 and distance(my_clone_balls[0]['position'], min_food_ball['position'])<10:
            self.actions_queue.put([direction.x, direction.y, action_type])
            direction = self.add_noise_to_direction(direction, noise_ratio=0.5)
            self.actions_queue.put([direction.x, direction.y, action_type])
        else:
            self.actions_queue.put([direction.x, direction.y, action_type])
        action_ret = self.actions_queue.get()
        return action_ret

    def step_level_2(self, obs):
        if self.actions_queue.qsize() > 0:
            return self.actions_queue.get()
        overlap = obs['overlap']
        overlap = self.preprocess(overlap)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']

        my_clone_balls, others_clone_balls = self.process_clone_balls(clone_balls)
        min_distance, min_thorns_ball = self.process_thorns_balls(thorns_balls, my_clone_balls[0])
        if min_thorns_ball is not None:
            direction = (min_thorns_ball['position'] - my_clone_balls[0]['position']).normalize()
        else:
            direction = (Vector2(0, 0) - my_clone_balls[0]['position']).normalize()
        action_type = -1
        self.actions_queue.put([direction.x, direction.y, action_type])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        action_ret = self.actions_queue.get()
        return action_ret

    def step_level_3(self, global_state, obs):
        overlap = obs['overlap']
        overlap = self.preprocess(overlap)

        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        food_balls = food_balls + spore_balls  # modify1
        clone_balls = overlap['clone']
        my_clone_balls, teammate_clone_balls, enemy_clone_balls = self.process_clone_balls(clone_balls)  # modify2

        if self.actions_queue.qsize() > 0:
            dangerous_status, safe_direction, direction_len = check_clone_dangerous(my_clone_balls, enemy_clone_balls)
            if dangerous_status==1:
                self.actions_queue = queue.Queue()
                action_ret = [safe_direction.x, safe_direction.y, -1]
                action_ret = adjust_direction(global_state, my_clone_balls, action_ret)
                return action_ret
            elif dangerous_status==2 and direction_len <= 1:
                self.actions_queue = queue.Queue()
                action_ret = [safe_direction.x, safe_direction.y, -1]
                action_ret = adjust_direction(global_state, my_clone_balls, action_ret)
                return action_ret
            else:
                action_ret = self.actions_queue.get()
                action_ret = adjust_direction(global_state, my_clone_balls, action_ret)
                return action_ret


        #print(my_clone_balls[0]['position'], my_clone_balls[0]['radius'])
        if len(my_clone_balls) >= 9 and my_clone_balls[4]['radius'] >= 12:
            self.actions_queue.put([None, None, 2])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            action_ret = self.actions_queue.get()
            return action_ret
        min_distance, min_thorns_ball = self.process_thorns_balls(thorns_balls, my_clone_balls[0])
        min_distance, min_food_ball = self.process_food_balls(food_balls, my_clone_balls[0])
        dangerous_status, safe_direction, direction_len = check_clone_dangerous(my_clone_balls, enemy_clone_balls)
        if len(enemy_clone_balls) > 0 and \
                (my_clone_balls[0]['radius'] < enemy_clone_balls[0]['radius'] or dangerous_status > 0):#modify3
            if my_clone_balls[0]['radius'] < enemy_clone_balls[0]['radius']:
                direction = (my_clone_balls[0]['position'] - enemy_clone_balls[0]['position']).normalize()
            if dangerous_status > 0:
                direction = safe_direction
            action_type = -1
        else:
            split_eat_result = check_split_eat(my_clone_balls, enemy_clone_balls)
            if split_eat_result is not None:
                direction = split_eat_result
                action_type = 1
                return [direction.x, direction.y, action_type]
            elif min_thorns_ball is not None:
                direction = (min_thorns_ball['position'] - my_clone_balls[0]['position']).normalize()
            else:
                if min_food_ball is not None:
                    direction = (min_food_ball['position'] - my_clone_balls[0]['position']).normalize()
                else:
                    direction = (Vector2(0, 0) - my_clone_balls[0]['position']).normalize()
            action_random = random.random()
            if action_random < 0.02:
                action_type = 1
            if action_random < 0.04 and action_random > 0.02:
                action_type = 2
            else:
                action_type = -1
        if my_clone_balls[0]['radius'] < 5.5 and min_food_ball is not None and distance(my_clone_balls[0]['position'],
                                                          min_food_ball['position']) < 10:
            self.actions_queue.put([direction.x, direction.y, action_type])
            direction = self.add_noise_to_direction(direction, noise_ratio=0.4)
            self.actions_queue.put([direction.x, direction.y, action_type])
        else:
            direction = self.add_noise_to_direction(direction)
            self.actions_queue.put([direction.x, direction.y, action_type])
        action_ret = self.actions_queue.get()

        action_ret = adjust_direction(global_state,my_clone_balls,action_ret)

        return action_ret

    def process_clone_balls(self, clone_balls):
        my_clone_balls = []
        teammate_clone_balls = []
        enemy_clone_balls = []
        for clone_ball in clone_balls:
            if clone_ball['player'] == self.player_name:
                my_clone_balls.append(copy.deepcopy(clone_ball))
        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        for clone_ball in clone_balls:
            if clone_ball['team'] == self.team_name and clone_ball['player'] != self.player_name:
                teammate_clone_balls.append(copy.deepcopy(clone_ball))
        teammate_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        for clone_ball in clone_balls:
            if clone_ball['player'] != self.player_name and clone_ball['team'] != self.team_name:
                enemy_clone_balls.append(copy.deepcopy(clone_ball))
        enemy_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        return my_clone_balls, teammate_clone_balls, enemy_clone_balls

    def process_thorns_balls(self, thorns_balls, my_max_clone_ball):
        min_distance = 10000
        min_thorns_ball = None
        for thorns_ball in thorns_balls:
            if thorns_ball['radius'] < my_max_clone_ball['radius']:
                distance = (thorns_ball['position'] - my_max_clone_ball['position']).length()
                if distance < min_distance:
                    min_distance = distance
                    min_thorns_ball = copy.deepcopy(thorns_ball)
        return min_distance, min_thorns_ball

    def process_food_balls(self, food_balls, my_max_clone_ball):
        min_distance = 10000
        min_food_ball = None
        for food_ball in food_balls:
            distance = (food_ball['position'] - my_max_clone_ball['position']).length()
            if distance < min_distance:
                min_distance = distance
                min_food_ball = copy.deepcopy(food_ball)
        return min_distance, min_food_ball

    def preprocess(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            if k == 'clone':
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {}
                    tmp['position'] = Vector2(vv[0], vv[1])
                    tmp['radius'] = vv[2]
                    tmp['player'] = str(int(vv[-2]))
                    tmp['team'] = str(int(vv[-1]))
                    new_overlap[k].append(tmp)
            else:
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {}
                    tmp['position'] = Vector2(vv[0], vv[1])
                    tmp['radius'] = vv[2]
                    new_overlap[k].append(tmp)
        return new_overlap

    def preprocess_tuple2vector(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            new_overlap[k] = []
            for index, vv in enumerate(v):
                new_overlap[k].append(vv)
                new_overlap[k][index]['position'] = Vector2(*vv['position'])
        return new_overlap

    def add_noise_to_direction(self, direction, noise_ratio=0.1):
        direction = direction + Vector2(((random.random() * 2 - 1) * noise_ratio) * direction.x,
                                        ((random.random() * 2 - 1) * noise_ratio) * direction.y)
        return direction
