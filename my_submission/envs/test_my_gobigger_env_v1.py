# -*- coding: utf-8 -*-
"""
# Author      : Camey
# DateTime    : 2022/2/12 1:37 下午
# Description : 
"""
from time import time
import copy
import pytest
from easydict import EasyDict
import numpy as np
from my_gobigger_env_v1 import MyGoBiggerEnvV1
from config.gobigger_no_spatial_config import main_config
from policy.obs_file.action_obs import *
@pytest.mark.unittest
class TestMyGoBiggerEnvV1:

    #@pytest.mark.parametrize('use_spatial', [False])
    # def test_env_by_server(self, use_spatial):
    #     cfg = copy.deepcopy(main_config)

    @pytest.mark.parametrize('use_spatial', [False])
    def test_env_by_obs_file(self):
        env = MyGoBiggerEnvV1(main_config.env)
        obs = action6_obs
        env.obs_transform(obs)
        


test = TestMyGoBiggerEnvV1()
test.test_env_by_obs_file()


# clones = np.array([[1,2,3,0,0],[1,2,3,1,0]])
# team_id,player_id = 1,0
# my_clones = [clone for clone in clones if clone[-2] == team_id and clone[-1]==player_id ]
# print(my_clones)