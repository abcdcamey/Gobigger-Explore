#from policy.bot_policy import MyBotAgent
from .policy.demo_bot_policy_v2_3 import MyBotAgent

class BaseSubmission:

    def __init__(self, team_name, player_names):
        self.team_name = team_name
        self.player_names = player_names

    def get_actions(self, obs):
        '''
        Overview:
            You must implement this function.
        '''
        raise NotImplementedError


class MySubmission(BaseSubmission):

    def __init__(self, team_name, player_names):
        super(MySubmission, self).__init__(team_name, player_names)
        self.agents = {}
        for player_name in self.player_names:
            self.agents[player_name] = MyBotAgent(team_name=team_name, player_name=player_name)

    def get_actions(self, obs):
        global_state, player_states = obs
        actions = {}

        agent_obs = {}
        for k, v in player_states.items():
            if v["team_name"] == self.team_name:
                agent_obs[k] = v

        for player_name, agent in self.agents.items():
            action = agent.step([global_state, agent_obs])
            actions[player_name] = action
        return actions
