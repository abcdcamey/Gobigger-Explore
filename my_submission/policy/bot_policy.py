# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/1/17 7:35 下午
# Description : 
"""
from gobigger.agents.base_agent import BaseAgent
import queue
import logging
import copy
logging.basicConfig(level=logging.DEBUG)
from pygame.math import Vector2
from collections import defaultdict
import random
import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

def get_min_distance_ball(my_max_clone_ball, balls,compare_radius=False):
    min_distance = 10000
    min_distance_ball = None
    for ball in balls:
        if compare_radius and my_max_clone_ball["radius"]<ball["radius"]:
            continue
        distance = (ball['position'] - my_max_clone_ball['position']).length()
        if distance < min_distance:
            min_distance = distance
            min_distance_ball = copy.deepcopy(ball)
    return min_distance, min_distance_ball


def get_safety_direction(my_clone_balls,enemy_clone_balls):

    my_area = 0
    for my_clone_ball in my_clone_balls:
        area = my_clone_ball['radius'] * my_clone_ball['radius']
        my_area += area

    enemy_player_name_area_dict = defaultdict(int)
    for enemy_clone_ball in enemy_clone_balls:
        player_name = enemy_clone_ball['player']
        area = enemy_clone_ball['radius'] * enemy_clone_ball['radius']
        enemy_player_name_area_dict[player_name] += area
    dangerous_enemy_player_name_dict = defaultdict(list)
    for player_name, area in enemy_player_name_area_dict.items():
        if area > my_area:
            for enemy_clone_ball in enemy_clone_balls:
                if enemy_clone_ball['player'] == player_name:
                    dangerous_enemy_player_name_dict[player_name].append(copy.deepcopy(enemy_clone_ball))

    # 求平均
    dangerous_position_list = []
    for player_name, ball_list in dangerous_enemy_player_name_dict.items():
        enemy_mean_area = enemy_player_name_area_dict[player_name] / len(ball_list)
        if enemy_mean_area >= my_area / len(my_clone_balls):
            max_radius = 0
            max_ball = None
            for ball in ball_list:
                if ball["radius"]>max_radius:
                    max_radius = ball["radius"]
                    max_ball = ball
            dangerous_position_list.append(max_ball["position"])

    if len(dangerous_position_list)==0:
        return [[random.uniform(-1, 1), random.uniform(-1, 1), -1]]
    dangerous_position = dangerous_position_list[0]
    for i in range(1,len(dangerous_position_list)):
        dangerous_position = dangerous_position+dangerous_position_list[i]

    direction = (my_clone_balls[0]['position'] - dangerous_position).normalize()
    return [[direction.x, direction.y, -1]]



def get_my_clone_ball_status(my_clone_balls) -> (int,int):
    my_clone_balls_num = len(my_clone_balls)
    area_sum = 0
    for ball in my_clone_balls:
        area_sum = area_sum + ball['radius']*ball['radius']
    #半径小于10的，算小质量
    if area_sum < 180:
        quality_type = 1
    # 半径小于20的，算中质量
    elif area_sum < 550:
        quality_type = 2
    else:
        quality_type = 3
    return my_clone_balls_num, quality_type

def preprocess(overlap):
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


def get_dangerous_status(my_clone_balls, enemy_clone_balls):
    dangerous_status = 0
    my_area = 0
    for my_clone_ball in my_clone_balls:
        area = my_clone_ball['radius'] * my_clone_ball['radius']
        my_area += area

    enemy_player_name_area_dict = defaultdict(int)
    for enemy_clone_ball in enemy_clone_balls:
        player_name = enemy_clone_ball['player']
        area = enemy_clone_ball['radius'] * enemy_clone_ball['radius']
        enemy_player_name_area_dict[player_name] += area
    dangerous_enemy_player_name_dict = defaultdict(list)
    for player_name, area in enemy_player_name_area_dict.items():
        if area > my_area:
            for enemy_clone_ball in enemy_clone_balls:
                if enemy_clone_ball['player']==player_name:
                    dangerous_enemy_player_name_dict[player_name].append(copy.deepcopy(enemy_clone_ball))

    if len(dangerous_enemy_player_name_dict)==0:
        dangerous_status=0
    else:
        all_ball_min_distance = 100000
        for my_clone_ball in my_clone_balls:
            for enemy_clone_ball in enemy_clone_balls:
                if enemy_clone_ball['player'] in dangerous_enemy_player_name_dict.keys():
                    distance = (enemy_clone_ball['position'] - my_clone_ball['position']).length()
                    if distance < all_ball_min_distance:
                        all_ball_min_distance = distance

        if all_ball_min_distance > 80:
            dangerous_status = 1
        #求平均
        else:
            dangerous_status = 2
            for player_name, ball_list in dangerous_enemy_player_name_dict.items():
                enemy_mean_area = enemy_player_name_area_dict[player_name]/len(ball_list)
                logging.info(f"enemy_player_name:{player_name},enemy_mean_area:{enemy_mean_area},my_area{my_area / len(my_clone_balls)}")
                if (enemy_mean_area*1.2) >= my_area/len(my_clone_balls):
                    dangerous_status = 3

    logging.info(f"dangerous_status:{dangerous_status}")
    return dangerous_status


class AgentAction:
    '''
    action_type表示需要执行的动作类型:
    1:移动吃离自己最近的食物或孢子
    2:吃总质量比自己小的敌人分身球
    3:主动分裂
    4:吃荆刺球分裂
    5:合并给队友
    '''
    def __init__(self, action_type=1):
        self.action_type = action_type
        self.call_num= 0
    def get_action_type(self):
        return self.action_type

    def get_action(self, **kwargs):
        self.call_num = self.call_num+1
        if self.action_type==1:
            return self.action1(**kwargs)
        if self.action_type==2:
            return self.action2(**kwargs)
        if self.action_type==3:
            return self.action3(**kwargs)
        if self.action_type==4:
            return self.action4(**kwargs)
        if self.action_type==5:
            return self.action5(**kwargs)
        if self.action_type==6:
            return self.action6(**kwargs)

    def check_action_condition(self, **kwargs):
        my_clone_balls_num = kwargs.get("my_clone_balls_num")
        quality_type = kwargs.get("quality_type")
        my_clone_balls = kwargs.get("my_clone_balls")
        food_balls = kwargs.get("food_balls")
        enemy_clone_balls = kwargs.get("enemy_clone_balls")
        teammate_clone_balls = kwargs.get("teammate_clone_balls")
        thorns_balls = kwargs.get("thorns_balls")

        if self.action_type == 1:
            return AgentAction.check_action1_condition(my_clone_balls, food_balls)
        if self.action_type == 2:
            if quality_type == 1:
                min_distance_threshold = 80
            elif quality_type == 2:
                min_distance_threshold = 150
            else:
                min_distance_threshold = 200
            return AgentAction.check_action2_condition(my_clone_balls, enemy_clone_balls,min_distance_threshold)
        if self.action_type == 3:
            if quality_type == 2:
                min_distance_threshold = 80
            else:
                min_distance_threshold = 120
            return AgentAction.check_action3_condition(my_clone_balls, enemy_clone_balls, thorns_balls, min_distance_threshold)
        if self.action_type == 4:
            if quality_type == 2:
                min_distance_threshold = 80
            else:
                min_distance_threshold = 120
            return AgentAction.check_action4_condition(my_clone_balls, thorns_balls, enemy_clone_balls, min_distance_threshold)

        if self.action_type==5:
            return AgentAction.check_action5_condition(my_clone_balls, teammate_clone_balls, enemy_clone_balls)

        if self.action_type==6:
            if my_clone_balls_num>1:
                return AgentAction.check_action6_condition(my_clone_balls, enemy_clone_balls, 40, 250)
            else:
                return False,[None,None]

    def action1(self, **kwargs):
        my_clone_balls = kwargs.get("my_clone_balls")
        food_balls = kwargs.get("food_balls")
        min_distance, min_distance_ball = get_min_distance_ball(my_clone_balls[0], food_balls)
        direction = (min_distance_ball['position'] - my_clone_balls[0]['position']).normalize()
        return [[direction.x, direction.y, -1]]

    @staticmethod
    def check_action1_condition(my_clone_balls, food_balls):
        min_distance = 100000
        target_ball = None
        for food_ball in food_balls:
            distance = (food_ball['position'] - my_clone_balls[0]['position']).length()
            if distance < min_distance:
                min_distance = distance
                target_ball = copy.deepcopy(food_ball)
        if min_distance != 100000:
            return True, [min_distance, target_ball]
        else:
            return False, [min_distance, target_ball]
    def action2(self, **kwargs):
        my_clone_balls = kwargs.get("my_clone_balls")
        enemy_clone_balls = kwargs.get("enemy_clone_balls")
        my_area = 0
        for my_clone_ball in my_clone_balls:
            area = my_clone_ball['radius']*my_clone_ball['radius']
            my_area += area

        enemy_player_name_dict = defaultdict(int)
        for enemy_clone_ball in enemy_clone_balls:
            player_name = enemy_clone_ball['player']
            area = enemy_clone_ball['radius']*enemy_clone_ball['radius']
            enemy_player_name_dict[player_name]+=area

        min_distance, min_distance_ball = get_min_distance_ball(my_clone_balls[0],enemy_clone_balls)
        direction = (min_distance_ball['position'] - my_clone_balls[0]['position']).normalize()
        return [[direction.x, direction.y, -1]]

    @staticmethod
    def check_action2_condition(my_clone_balls, enemy_clone_balls, min_distance_threshold):
        min_distance, min_distance_ball = get_min_distance_ball(my_clone_balls[0], enemy_clone_balls,
                                                                compare_radius=True)
        if min_distance_ball is not None and min_distance < min_distance_threshold:
            return True, [min_distance, min_distance_ball]
        return False, [None, None]


    def action3(self, **kwargs):
        return [[None, None, 1]]

    @staticmethod
    def check_action3_condition(my_clone_balls, enemy_clone_balls, thorns_balls, min_distance_threshold):
        min_distance = 100000
        my_min_clone_ball = my_clone_balls[-1]
        for enemy_clone_ball in enemy_clone_balls:
            if enemy_clone_ball["radius"] > my_min_clone_ball["radius"]:
                distance = (my_min_clone_ball['position'] - enemy_clone_ball['position']).length()
                if distance < min_distance:
                    min_distance = distance
        #计算平均质量
        my_area = 0
        for my_clone_ball in my_clone_balls:
            area = my_clone_ball['radius'] * my_clone_ball['radius']
            my_area += area
        if min_distance < min_distance_threshold or my_area/len(my_clone_balls) < 50 or len(my_clone_balls)>3:
            return False, [None, None]

        #找距离最近的荆刺球
        min_distance = 100000
        for thorns_ball in thorns_balls:
            distance = (thorns_ball['position'] - my_clone_balls[0]['position']).length()
            if distance < min_distance:
                min_distance = distance
        if min_distance < min_distance_threshold:
            return False, [None, None]
        if my_clone_balls[0]["radius"]>30:
            return True, [None, None]
        elif my_clone_balls[0]["radius"]>10:
            return True, [None, None]
        else:
            return False, [None, None]
    def action4(self, **kwargs):
        my_clone_balls = kwargs.get("my_clone_balls")
        thorns_balls = kwargs.get("thorns_balls")
        available_thorns_balls = []
        for thorns_ball in thorns_balls:
            if my_clone_balls[0]['radius']>thorns_ball['radius']:
                available_thorns_balls.append(copy.deepcopy(thorns_ball))
        if len(available_thorns_balls)>0:
            min_distance, min_distance_ball = get_min_distance_ball(my_clone_balls[0], available_thorns_balls)
            direction = (min_distance_ball['position'] - my_clone_balls[0]['position']).normalize()
            return [[direction.x, direction.y, -1]]
        else:
            return [[random.uniform(-1, 1), random.uniform(-1, 1), -1]]

    @staticmethod
    def check_action4_condition(my_clone_balls, thorns_balls, enemy_clone_balls, min_distance_threshold):
        dangerous_status = get_dangerous_status(my_clone_balls, enemy_clone_balls)
        if dangerous_status==0 or dangerous_status==1:
            min_distance = 100000
            target_ball = None
            for thorns_ball in thorns_balls:
                if thorns_ball["radius"] < my_clone_balls[0]["radius"]:
                    distance = (my_clone_balls[0]['position'] - thorns_ball['position']).length()
                    if distance < min_distance:
                        min_distance = distance
                        target_ball = thorns_ball
            if min_distance < min_distance_threshold:
                return True,[min_distance, target_ball]
            else:
                return False, [None, None]
        else:
            return False,[None, None]

    def action5(self, **kwargs):
        my_clone_balls = kwargs.get("my_clone_balls")
        teammate_clone_balls = kwargs.get("teammate_clone_balls")

        direction = (teammate_clone_balls[0]['position'] - my_clone_balls[0]['position']).normalize()
        return [[direction.x, direction.y, -1]]

    @staticmethod
    def check_action5_condition(my_clone_balls, teammate_clone_balls, enemy_clone_balls):

        my_area = 0
        for my_clone_ball in my_clone_balls:
            area = my_clone_ball['radius'] * my_clone_ball['radius']
            my_area += area

        enemy_player_name_area_dict = defaultdict(int)
        for enemy_clone_ball in enemy_clone_balls:
            player_name = enemy_clone_ball['player']
            area = enemy_clone_ball['radius'] * enemy_clone_ball['radius']
            enemy_player_name_area_dict[player_name] += area
        max_enemy_area=0
        if len(enemy_player_name_area_dict)>0:
            max_enemy_area = max(enemy_player_name_area_dict.values())

        teammate_player_name_area_dict = defaultdict(int)
        for teammate_clone_ball in teammate_clone_balls:
            player_name = teammate_clone_ball['player']
            area = teammate_clone_ball['radius'] * teammate_clone_ball['radius']
            teammate_player_name_area_dict[player_name] += area

        max_teammate_area = 0
        max_teammate_player_name = None
        for player_name, area in teammate_player_name_area_dict.items():
            if area > max_teammate_area:
                max_teammate_area = area
                max_teammate_player_name = player_name

        if max_teammate_player_name is not None and max_enemy_area>max_teammate_area and my_area+max_teammate_area>(1.2*max_enemy_area):
            max_teammate_ball_radius = 0
            max_teammate_ball = None
            distance = 0
            for teammate_clone_ball in teammate_clone_balls:
                if max_teammate_player_name == teammate_clone_ball["player"] and teammate_clone_ball["radius"]>max_teammate_ball_radius:
                    max_teammate_ball_radius = teammate_clone_ball["radius"]
                    max_teammate_ball = teammate_clone_ball
                    distance = (teammate_clone_ball['position'] - my_clone_balls[0]['position']).length()
            return True, [distance,max_teammate_ball]
        return False, [None, None]


    def action6(self, **kwargs):
        my_clone_balls = kwargs.get("my_clone_balls")
        mean_distance = 0
        mean_radius = 0
        for i in range(1, len(my_clone_balls)):
            distance = (my_clone_balls[0]['position'] - my_clone_balls[i]['position']).length()
            mean_distance += distance
            mean_radius += (my_clone_balls[i]["radius"]+my_clone_balls[0]["radius"])
        mean_distance = mean_distance / (len(my_clone_balls) - 1)
        mean_radius = mean_radius/ (len(my_clone_balls) - 1)
        actions = []
        #logging.debug(f"mean_distance:{mean_distance},mean_radius:{((mean_radius)*1.9)}")
        if self.call_num<=1:
            actions.append([None, None, 2])
        if len(my_clone_balls)>=9 and my_clone_balls[4]['radius'] > 14:
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
            actions.append([None, None, 0])

        elif len(my_clone_balls)>6 and my_clone_balls[4]['radius'] > 10:
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
            actions.append([None, None, -1])
        elif len(my_clone_balls)>4 and my_clone_balls[3]['radius'] > 7:
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
            actions.append([None, None, 0])
        else:
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
            actions.append([None, None, -1])
        # elif mean_distance<(mean_radius*1.6) and len(my_clone_balls)>4 and my_clone_balls[4]['radius'] > 10:
        #     # actions.append([None, None, 2])
        #     actions.append([None, None, -1])
        #     actions.append([None, None, -1])
        #     # actions.append([None, None, 0])
        #     # actions.append([None, None, 0])
        #     # actions.append([None, None, 0])
        #     # actions.append([None, None, 0])
        # else:
        #     # actions.append([None, None, 2])
        #     actions.append([None, None, -1])
        #     actions.append([None, None, -1])
        #     # actions.append([None, None, -1])
        #     # actions.append([None, None, -1])
        #     # actions.append([None, None, -1])
        return actions

    @staticmethod
    def check_action6_condition(my_clone_balls, enemy_clone_balls, min_distance_threshold,max_distance_threshold):

        my_area = 0
        for my_clone_ball in my_clone_balls:
            area = my_clone_ball['radius'] * my_clone_ball['radius']
            my_area += area

        mean_distance = 0
        for i in range(1,len(my_clone_balls)):
            distance = (my_clone_balls[0]['position']-my_clone_balls[i]['position']).length()
            mean_distance+=distance
        mean_distance = mean_distance/(len(my_clone_balls)-1)

        enemy_player_name_area_dict = defaultdict(int)
        enemy_player_name_ball_dict = defaultdict(list)
        for enemy_clone_ball in enemy_clone_balls:
            player_name = enemy_clone_ball['player']
            area = enemy_clone_ball['radius'] * enemy_clone_ball['radius']
            enemy_player_name_area_dict[player_name] += area
            enemy_player_name_ball_dict[player_name].append(enemy_clone_ball)

        dangerous_cnt = 0
        dangerous_min_distance = 100000
        dangerous_area = None
        for player_name, ball_list in enemy_player_name_ball_dict.items():
            enemy_mean_area = enemy_player_name_area_dict[player_name] / len(ball_list)
            min_distance, min_distance_ball = get_min_distance_ball(my_clone_balls[0], ball_list)
            if my_area/len(my_clone_balls)< enemy_mean_area and my_area>(1.2*enemy_player_name_area_dict[player_name]):
                dangerous_cnt+=1
                if min_distance<dangerous_min_distance:
                    dangerous_min_distance=min_distance

        if len(my_clone_balls)>5 and my_clone_balls[0]['radius']>14 and dangerous_min_distance>min_distance_threshold:
            return True, [len(my_clone_balls), my_clone_balls[0]]
        if len(my_clone_balls)>5 and my_clone_balls[0]['radius']>10 and mean_distance>(my_clone_balls[0]['radius']*2.5):
            return True, [len(my_clone_balls), my_clone_balls[0]]
        return False, [None, None]



class MyBotAgent(BaseAgent):
    def __init__(self, team_name=None, player_name=None):
        self.team_name = team_name
        self.player_name = player_name
        self.actions_todo_queue = queue.Queue()
        self.actions_done_list = []
        self.pre_agent_action = None

    def policy(self, **kwargs):
        my_clone_balls_num = kwargs.get("my_clone_balls_num")
        quality_type = kwargs.get("quality_type")
        my_clone_balls = kwargs.get("my_clone_balls")
        food_balls = kwargs.get("food_balls")
        enemy_clone_balls = kwargs.get("enemy_clone_balls")
        teammate_clone_balls = kwargs.get("teammate_clone_balls")
        thorns_balls = kwargs.get("thorns_balls")
        fn_choices = []
        weights = []

        if quality_type == 1:
            fn_choices.append(AgentAction(1))
            weights.append(1)
            action2_condition, action2_condition_info = AgentAction.check_action2_condition(my_clone_balls, enemy_clone_balls, 80)
            if action2_condition:
                fn_choices.append(AgentAction(2))
                weights.append(2)
            #有多个分身
            if my_clone_balls_num > 1:
                action6_condition, action6_condition_info = AgentAction.check_action6_condition(my_clone_balls, enemy_clone_balls, 40, 250)
                if action6_condition:
                    fn_choices.append(AgentAction(6))
                    weights.append(1)
        if quality_type == 2:
            fn_choices.append(AgentAction(1))
            weights.append(1)
            action2_condition, action2_condition_info = AgentAction.check_action2_condition(my_clone_balls, enemy_clone_balls, 150)
            if action2_condition:
                fn_choices.append(AgentAction(2))
                weights.append(10)
            action3_condition, action3_condition_info = AgentAction.check_action3_condition(my_clone_balls, enemy_clone_balls, thorns_balls, 80)
            if action3_condition:
                fn_choices.append(AgentAction(3))
                weights.append(0.2)
            action4_condition, action4_condition_info = AgentAction.check_action4_condition(my_clone_balls, thorns_balls, enemy_clone_balls, 80)
            if action4_condition:
                fn_choices.append(AgentAction(4))
                weights.append(2)
            action5_condition, action5_condition_info = AgentAction.check_action5_condition(my_clone_balls, teammate_clone_balls, enemy_clone_balls)
            if action5_condition:
                fn_choices.append(AgentAction(5))
                weights.append(1)
            if my_clone_balls_num > 1:
                action6_condition, action6_condition_info = AgentAction.check_action6_condition(my_clone_balls, enemy_clone_balls, 40, 250)
                if action6_condition:
                    fn_choices.append(AgentAction(6))
                    weights.append(1)
        if quality_type == 3:
            action2_condition, action2_condition_info = AgentAction.check_action2_condition(my_clone_balls, enemy_clone_balls, 200)
            if action2_condition:
                fn_choices.append(AgentAction(2))
                weights.append(10)
            action3_condition, action3_condition_info = AgentAction.check_action3_condition(my_clone_balls, enemy_clone_balls, thorns_balls, 120)
            if action3_condition:
                fn_choices.append(AgentAction(3))
                weights.append(0.2)
            action4_condition, action4_condition_info = AgentAction.check_action4_condition(my_clone_balls, thorns_balls,
                                                                       enemy_clone_balls, 120)
            if action4_condition:
                fn_choices.append(AgentAction(4))
                weights.append(2)
            action5_condition, action5_condition_info = AgentAction.check_action5_condition(my_clone_balls, teammate_clone_balls,
                                                                       enemy_clone_balls)
            if action5_condition:
                fn_choices.append(AgentAction(5))
                weights.append(1)

            if my_clone_balls_num > 1:
                action6_condition, action6_condition_info = AgentAction.check_action6_condition(my_clone_balls, enemy_clone_balls, 40, 250)
                if action6_condition:
                    fn_choices.append(AgentAction(6))
                    weights.append(1)
        if len(fn_choices)==0:
            fn_choices.append(AgentAction(1))
            weights.append(1)
        agent_action = random.choices(fn_choices, weights)[0]
        logging.info(f"choice:{agent_action.get_action_type()}, random choices:{[c.get_action_type() for c in fn_choices]},weights:{[w for w in weights]}")

        return agent_action

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

    def step(self, obs):
        global_state, player_state = obs
        total_time = global_state['total_time']
        last_time = global_state['last_time']
        rest_time = total_time - last_time
        leaderboard = global_state['leaderboard']
        player_state_single = player_state[self.player_name]
        player_state_single.pop('feature_layers', None)
        player_state_single_log = copy.deepcopy(player_state_single)
        player_state_single_log["overlap"]["food"] = player_state_single_log["overlap"]["food"][:3]
        logging.info(f"player_name:{self.player_name},player_state_sigle:{player_state_single_log}")

        left_top_x, left_top_y, right_bottom_x, right_bottom_y = player_state_single['rectangle']
        overlap = player_state_single['overlap']
        overlap = preprocess(overlap)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']

        food_balls = food_balls+spore_balls # 把苞子球当做食物球处理

        clone_balls = overlap['clone']
        my_clone_balls, teammate_clone_balls, enemy_clone_balls = self.process_clone_balls(clone_balls)

        # 获取分身状态
        my_clone_balls_num, quality_type = get_my_clone_ball_status(my_clone_balls)
        dangerous_status = get_dangerous_status(my_clone_balls, enemy_clone_balls)

        if dangerous_status == 0 or dangerous_status == 1 or dangerous_status == 2:
            param = {"my_clone_balls":my_clone_balls,"teammate_clone_balls":teammate_clone_balls,"enemy_clone_balls":enemy_clone_balls,
                                    "food_balls":food_balls, "thorns_balls":thorns_balls,"my_clone_balls_num":my_clone_balls_num,"quality_type":quality_type}
            if self.pre_agent_action is not None:
                #检查action
                pre_agent_action_log = copy.deepcopy(self.pre_agent_action)
                condition, condition_info = self.pre_agent_action.check_action_condition(**param)

                if condition:#依然满足action的条件，可以继续执行action队列
                    #logging.debug(f"continue:{pre_agent_action_log.get_action_type()}")
                    if self.actions_todo_queue.qsize() > 0:
                        action_ret = self.actions_todo_queue.get()
                        self.actions_done_list.append(action_ret)
                    else:
                        action_list = self.pre_agent_action.get_action(**param)
                        for action in action_list:
                            self.actions_todo_queue.put([action, copy.deepcopy(self.pre_agent_action), copy.deepcopy(condition_info)])
                        action_ret = self.actions_todo_queue.get()
                        self.actions_done_list.append(action_ret)
                    if self.pre_agent_action.get_action_type()==2 and len(self.actions_done_list)>6:#避免无休止追击敌人
                        action_ret_trace = self.actions_done_list[-6:]
                        flag = False
                        for _action_ret in action_ret_trace:
                            if _action_ret[1].get_action_type()!=2:
                                flag = True
                                break
                        if flag==False and condition_info[0]>(action_ret_trace[0][2][0]*0.85): #distance
                            self.pre_agent_action=None
                            self.actions_todo_queue = queue.Queue()
                            action_ret = [[random.uniform(-1, 1), random.uniform(-1, 1), -1],None, None]

                    elif self.pre_agent_action.get_action_type()==1 and len(self.actions_done_list)>8:# 避免绕圈
                        action_ret_trace = self.actions_done_list[-8:]
                        flag = False
                        for _action_ret in action_ret_trace:
                            if _action_ret[1].get_action_type()!= 1:
                                flag = True
                                break
                            #logging.debug(f"{_action_ret[1].get_action_type()},{_action_ret[2][0]}")
                        #logging.debug(f"{action_ret_trace[0]},{action_ret_trace[0][1].get_action_type()}")
                        if flag==False and condition_info[0] > (action_ret_trace[0][2][0] * 0.85):  # distance
                            self.pre_agent_action = None
                            self.actions_todo_queue = queue.Queue()
                            action_ret = [[random.uniform(-1, 1), random.uniform(-1, 1), -1], None, None]
                    elif self.pre_agent_action.get_action_type()==3:
                        self.pre_agent_action = None
                        self.actions_todo_queue = queue.Queue()
                    logging.info(f"continue:{pre_agent_action_log.get_action_type()}")
                else:
                    agent_action = self.policy(**param)
                    self.pre_agent_action = agent_action
                    action_list = agent_action.get_action(**param)
                    new_condition, new_condition_info = self.pre_agent_action.check_action_condition(**param)
                    for action in action_list:
                        self.actions_todo_queue.put([action, copy.deepcopy(agent_action),copy.deepcopy(new_condition_info)])
                    action_ret = self.actions_todo_queue.get()
                    self.actions_done_list.append(action_ret)
            else:
                agent_action = self.policy(**param)
                self.pre_agent_action = agent_action
                new_condition, new_condition_info = self.pre_agent_action.check_action_condition(**param)

                action_list = agent_action.get_action(**param)
                for action in action_list:
                    self.actions_todo_queue.put([action, copy.deepcopy(agent_action),copy.deepcopy(new_condition_info)])
                action_ret = self.actions_todo_queue.get()
                self.actions_done_list.append(action_ret)
            # if action_ret[1] is not None:
            #     logging.debug(f"action type:{action_ret[1].get_action_type()},action:{action_ret[0]}")
            # else:
            #     logging.debug(f"action type:{None},action:{action_ret[0]}")
            logging.info(f"action:{action_ret[0]}")

            return action_ret[0]
        else:
            action = get_safety_direction(my_clone_balls, enemy_clone_balls)
            self.pre_agent_action = None
            self.actions_todo_queue = queue.Queue()
            self.actions_done_list = []
            logging.info(f"action:{action[0]}")

            return action[0]




class RandomAgent(BaseAgent):
    def __init__(self, team_name=None, player_name=None):
        self.team_name = team_name
        self.player_name = player_name

    def step(self, obs):
        return [random.uniform(-1, 1), random.uniform(-1, 1), -1]