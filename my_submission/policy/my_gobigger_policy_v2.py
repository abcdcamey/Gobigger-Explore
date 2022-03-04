from typing import List, Dict, Any, Tuple
from collections import namedtuple
import copy
import torch

from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_nstep_return_data, get_train_sample, q_nstep_td_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy
#from ding.policy.common_utils import default_preprocess_learn
import torch.nn.functional as F
import logging
import time
import os
import numpy as np
from torch.optim.lr_scheduler import *
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

def gobigger_collate(data):
    r"""
    Arguments:
        - data (:obj:`list`): Lsit type data, [{scalar:[player_1_scalar, player_2_scalar, ...], ...}, ...]
    """
    B, player_num_per_team = len(data), len(data[0]['scalar'])
    data = {k: sum([d[k] for d in data], []) for k in data[0].keys() if not k.startswith('collate_ignore')}
    clone_num = max([x.shape[0] for x in data['clones']])
    thorn_num = max([x.shape[1] for x in data['clone_with_thorn_relation']])
    food_h = max([x.shape[1] for x in data['food_map']])
    food_w = max([x.shape[2] for x in data['food_map']])
    data['scalar'] = torch.stack([torch.as_tensor(x) for x in data['scalar']]).float() # [B*player_num_per_team,5]
    data['food_map'] = torch.stack([F.pad(torch.as_tensor(x), (0, food_w - x.shape[2], 0, food_h - x.shape[1])) for x in data['food_map']]).float()
    data['clone_with_food_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['clone_with_food_relation']]).float()
    data['thorn_mask'] = torch.stack([torch.arange(thorn_num) < x.shape[1] for x in data['clone_with_thorn_relation']]).float()
    data['clone_with_thorn_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, thorn_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['clone_with_thorn_relation']]).float()
    data['clone_mask'] = torch.stack([torch.arange(clone_num) < x.shape[0] for x in data['clones']]).float()
    data['clones'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[0])) for x in data['clones']]).float()
    data['clone_with_team_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['clone_with_team_relation']]).float()
    data['clone_with_enemy_relation'] = torch.stack([F.pad(torch.as_tensor(x), (0, 0, 0, clone_num - x.shape[1], 0, clone_num - x.shape[0])) for x in data['clone_with_enemy_relation']]).float()

    data['batch'] = B
    data['player_num_per_team'] = player_num_per_team
    return data


def default_preprocess_learn(
        data: List[Any],
        use_priority_IS_weight: bool = False,
        use_priority: bool = False,
        use_nstep: bool = False,
        ignore_done: bool = False,
) -> dict:

    # caculate max clone num
    tmp = [d['obs'] for d in data]
    tmp = {k: sum([d[k] for d in tmp], []) for k in tmp[0].keys() if not k.startswith('collate_ignore')}
    max_clone_num = max([x.shape[0] for x in tmp['clones']])
    limit = 52
    #print('max_clone_num:{}, limit:{}'.format(max_clone_num,limit))
    mini_bs = int(len(data)//2)
    if max_clone_num > limit: # 限制52个clone_num
        split_data1 = data[:mini_bs]
        split_data2 = data[mini_bs:]

        re = []
        for dt in (split_data1, split_data2):
            obs = [d['obs'] for d in dt]
            next_obs = [d['next_obs'] for d in dt]
            for i in range(len(dt)):
                dt[i] = {k: v for k, v in dt[i].items() if not 'obs' in k}
            dt = default_collate(dt)
            dt['obs'] = gobigger_collate(obs)
            dt['next_obs'] = gobigger_collate(next_obs)
            if ignore_done:
                dt['done'] = torch.zeros_like(dt['done']).float()
            else:
                dt['done'] = dt['done'].float()
            if use_priority_IS_weight:
                assert use_priority, "Use IS Weight correction, but Priority is not used."
            if use_priority and use_priority_IS_weight:
                dt['weight'] = dt['IS']
            else:
                dt['weight'] = dt.get('weight', None)
            if use_nstep:
                # Reward reshaping for n-step
                reward = dt['reward']
                if len(reward.shape) == 1:
                    reward = reward.unsqueeze(1)
                # reward: (batch_size, nstep) -> (nstep, batch_size)
                dt['reward'] = reward.permute(1, 0).contiguous()
            re.append(dt)
        return re

    # data collate
    obs = [d['obs'] for d in data]
    next_obs = [d['next_obs'] for d in data]
    for i in range(len(data)):
        data[i] = {k: v for k, v in data[i].items() if not 'obs' in k}
    data = default_collate(data)
    data['obs'] = gobigger_collate(obs)
    data['next_obs'] = gobigger_collate(next_obs)
    if ignore_done:
        data['done'] = torch.zeros_like(data['done']).float()
    else:
        data['done'] = data['done'].float()
    if use_priority_IS_weight:
        assert use_priority, "Use IS Weight correction, but Priority is not used."
    if use_priority and use_priority_IS_weight:
        data['weight'] = data['IS']
    else:
        data['weight'] = data.get('weight', None)
    if use_nstep:
        # Reward reshaping for n-step
        reward = data['reward']
        # if len(reward.shape) == 2:
        #     reward = reward.unsqueeze(2)
        # reward: (batch_size, nstep) -> (nstep, batch_size)
        #data['reward'] = reward.permute(1, 0).contiguous()
    return data

def view_similar(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    size = list(x.shape) + [1 for _ in range(len(target.shape) - len(x.shape))]
    return x.view(*size)
nstep_return_data = namedtuple('nstep_return_data', ['reward', 'next_value', 'done'])
import torch.nn as nn
from typing import Any, Optional, Callable

def nstep_return(data: namedtuple, gamma: float, nstep: int, value_gamma: Optional[torch.Tensor] = None):
    reward, next_value, done = data
    assert reward.shape[0] == nstep
    device = reward.device
    reward_factor = torch.ones(nstep).to(device)
    for i in range(1, nstep):
        reward_factor[i] = gamma * reward_factor[i - 1]
    reward_factor = view_similar(reward_factor, reward)
    reward = reward.mul(reward_factor).sum(0)
    if value_gamma is None:
        return_ = reward + (gamma ** nstep) * next_value * (1 - done)
    else:
        return_ = reward + value_gamma * next_value * (1 - done)
    return return_


def q_nstep_td_error(
        data: namedtuple,
        gamma: float,
        nstep: int = 1,
        cum_reward: bool = False,
        value_gamma: Optional[torch.Tensor] = None,
        criterion: torch.nn.modules = nn.L1Loss(reduction='none'),
) -> torch.Tensor:
    """
    Overview:
        Multistep (1 step or n step) td_error for q-learning based algorithm
    Arguments:
        - data (:obj:`q_nstep_td_data`): the input data, q_nstep_td_data to calculate loss
        - gamma (:obj:`float`): discount factor
        - cum_reward (:obj:`bool`): whether to use cumulative nstep reward, which is figured out when collecting data
        - value_gamma (:obj:`torch.Tensor`): gamma discount value for target q_value
        - criterion (:obj:`torch.nn.modules`): loss function criterion
        - nstep (:obj:`int`): nstep num, default set to 1
    Returns:
        - loss (:obj:`torch.Tensor`): nstep td error, 0-dim tensor
        - td_error_per_sample (:obj:`torch.Tensor`): nstep td error, 1-dim tensor
    Shapes:
        - data (:obj:`q_nstep_td_data`): the q_nstep_td_data containing\
            ['q', 'next_n_q', 'action', 'reward', 'done']
        - q (:obj:`torch.FloatTensor`): :math:`(B, N)` i.e. [batch_size, action_dim]
        - next_n_q (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - next_n_action (:obj:`torch.LongTensor`): :math:`(B, )`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is timestep(nstep)
        - done (:obj:`torch.BoolTensor`) :math:`(B, )`, whether done in last timestep
        - td_error_per_sample (:obj:`torch.FloatTensor`): :math:`(B, )`
    """
    q, next_n_q, action, next_n_action, reward, done, weight = data
    if weight is None:
        weight = torch.ones_like(reward)
    if len(action.shape) > 1:  # MARL case
        #reward = reward.unsqueeze(0)
        #weight = weight.unsqueeze(-1)
        done = done.unsqueeze(-1)
        if value_gamma is not None:
            value_gamma = value_gamma.unsqueeze(-1)

    q_s_a = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    target_q_s_a = next_n_q.gather(-1, next_n_action.unsqueeze(-1)).squeeze(-1)

    if cum_reward:
        if value_gamma is None:
            target_q_s_a = reward + (gamma ** nstep) * target_q_s_a * (1 - done)
        else:
            target_q_s_a = reward + value_gamma * target_q_s_a * (1 - done)
    else:

        target_q_s_a = nstep_return(nstep_return_data(reward, target_q_s_a, done), gamma, nstep, value_gamma)
    td_error_per_sample = criterion(q_s_a, target_q_s_a.detach())
    return (td_error_per_sample * weight).mean(), td_error_per_sample




@POLICY_REGISTRY.register('my_gobigger_dqn')
class MyDQNPolicy(Policy):
    r"""
    Overview:
        Policy class of DQN algorithm, extended by Double DQN/Dueling DQN/PER/multi-step TD.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      dqn            | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        7  ``nstep``            int      1,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  | ``learn.update``   int      3              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        9  | ``learn.multi``    bool     False          | whether to use multi gpu during
           | ``_gpu``
        10 | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        13 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        14 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        15 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        16 | ``other.eps.type`` str      exp            | exploration rate decay type            | Support ['exp',
                                                                                                 | 'linear'].
        17 | ``other.eps.       float    0.95           | start value of exploration rate        | [0,1]
           |  start``
        18 | ``other.eps.       float    0.1            | end value of exploration rate          | [0,1]
           |  end``
        19 | ``other.eps.       int      10000          | decay length of exploration            | greater than 0. set
           |  decay``                                                                            | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        == ==================== ======== ============== ======================================== =======================
    """

    config = dict(
        type='dqn',
        cuda=False,
        on_policy=False,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        discount_factor=0.97,
        nstep=1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=8,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay length(env step)
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``, initialize the optimizer, algorithm arguments, main \
            and target model.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        self._lr_scheduler = MultiStepLR(self._optimizer, milestones=[40000, 50000, 70000, 80000], gamma=np.float64(0.6))
        #self._lr_scheduler = MultiStepLR(self._optimizer, milestones=[10, 15, 25000, 30000, 33000], gamma=np.float64(0.6))

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Forward computation graph of learn mode(updating policy).
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``
            - optional: ``value_gamma``, ``IS``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``, ``priority``
            - optional: ``action_distribution``
        """
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )

        ####################################################################
        if isinstance(data, list):
            self._optimizer.zero_grad()
            for dt in data:
                if self._cuda:
                    dt = to_device(dt, self._device)
                # ====================
                # Q-learning forward
                # ====================
                self._learn_model.train()
                self._target_model.train()
                # Current q value (main model)
                q_value = self._learn_model.forward(dt['obs'])['logit']
                # Target q value
                with torch.no_grad():
                    target_q_value = self._target_model.forward(dt['next_obs'])['logit']
                    # Max q value action (main model)
                    target_q_action = self._learn_model.forward(dt['next_obs'])['action']

                data_n = q_nstep_td_data(
                    q_value, target_q_value, dt['action'], target_q_action, dt['reward'], dt['done'], dt['weight']
                )
                value_gamma = dt.get('value_gamma')
                loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma)

                # ====================
                # Q-learning update
                # ====================
                loss.backward()
            if self._cfg.learn.multi_gpu:
                self.sync_gradients(self._learn_model)
            self._optimizer.step()
            self._lr_scheduler.step()
            # =============
            # after update
            # =============
            self._target_model.update(self._learn_model.state_dict())
            return {
                'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
                'total_loss': loss.item(),
                'q_value': q_value.mean().item(),
                'priority': td_error_per_sample.abs().tolist(),
                # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
                # '[histogram]action_distribution': data['action'],
            }
        ####################################################################

        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        q_value = self._learn_model.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
        )
        value_gamma = data.get('value_gamma')
        cum_reward = True if self._nstep>1 else False
        loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep, cum_reward=cum_reward, value_gamma=value_gamma)

        # ====================
        # Q-learning update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.learn.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()
        self._lr_scheduler.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.state_dict()['param_groups'][0]['lr'],
            'total_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'total_loss', 'q_value']

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model, \
            enable the eps_greedy_sample for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of collect mode(collect training data), with eps_greedy for exploration.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting policy_output(action) for the interaction with \
                env and the constructing of transition.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``logit``, ``action``
        """
        data_id = list(data.keys())
        data = gobigger_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. A train sample can be a processed transition(DQN with nstep TD) \
            or some continuous transitions(DRQN).
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): The list of training samples.

        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procecure by overriding this two methods and collector \
            itself.
        """
        cum_reward = True
        data = get_nstep_return_data(data, self._nstep, cum_reward = cum_reward, gamma=self._gamma)

        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, policy_output: Dict[str, Any], timestep: namedtuple) -> Dict[str, Any]:
        """
        Overview:
            Generate a transition(e.g.: <s, a, s', r, d>) for this algorithm training.
        Arguments:
            - obs (:obj:`Any`): Env observation.
            - policy_output (:obj:`Dict[str, Any]`): The output of policy collect mode(``self._forward_collect``),\
                including at least ``action``.
            - timestep (:obj:`namedtuple`): The output after env step(execute policy output action), including at \
                least ``obs``, ``reward``, ``done``, (here obs indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        #logging.info(f"data:{data[0]['collate_ignore_raw_obs']}")

        data = gobigger_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()

        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        logging.info(f"output:{output}")
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For DQN, ``ding.model.template.q_learning.DQN``
        """
        return 'dqn', ['ding.model.template.q_learning']
