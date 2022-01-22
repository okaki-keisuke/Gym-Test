import torch
from torch.nn import parameter
import copy
import ray
import gym
from collections import namedtuple
from model import Net
from utils import initial_state, input_state


ENV = "BreakoutDeterministic-v4"

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Agent:
    def __init__(self, advanced_step: int):
        self.multi_step = advanced_step
        self.state = []
        self.reward = []
        self.store_state = []
    
    def state_storage(self, state: torch.Tensor) -> None:
        self.store_state.append(state)
        if len(self.store_state) > self.multi_step + 1:
            del self.store_state[0]
        self.state = copy.deepcopy(self.store_state)
        del self.state[1:-1]
        assert len(self.state) <= 2

    def reward_storage(self, reward: torch.Tensor) -> None:
        self.reward.append(reward) 
        if len(self.reward) > self.multi_step:
            del self.reward[0]

    def data_full(self) -> bool:
        if len(self.reward) == self.multi_step and len(self.store_state) == self.multi_step + 1:
            return True
        
        return False
    
    def reset(self):
        self.state = []
        self.reward = []
        self.store_state = []


@ray.remote
class Environment:
    def __init__(self, pid: int, gamma: float, advanced_step: int, epsilon: float, local_cycle: int):
        self.pid = pid
        self.env_name = ENV
        self.env = gym.make(ENV)
        self.action_space = self.env.action_space.n
        self.q_network = Net(action_space=self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.advanced_step = advanced_step
        state = self.env.reset()
        self.agent = Agent(advanced_step=self.advanced_step)
        self.agent.state_storage(initial_state(state))
        self.episode_reward = 0
        self.local_cycle = local_cycle
    
    def rollout(self, weights: parameter) -> list:

        #print("Start:{}-Environment".format(self.pid))

        self.q_network.load_state_dict(weights) 
        buffer = []

        for _ in range(self.local_cycle):

            action = self.q_network.get_action(self.agent.state[-1], epsilon=self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.state_storage(input_state(next_state, self.agent.state[-1]))
            self.episode_reward += reward
            self.agent.reward_storage(reward)
            if self.agent.data_full():
                transition = Transition(self.agent.state[0],
                                        torch.LongTensor([action]),
                                        self.agent.state[1], 
                                        torch.FloatTensor([self.agent.reward]),
                                        torch.BoolTensor([done])
                                        )
                buffer.append(transition)
            if done:
               self.agent.reset()
               state = self.env.reset()
               self.agent.state_storage(initial_state(state))
               self.episode_reward = 0
                
        td_error , transitions = self.init_prior(buffer)

        return td_error, transitions, self.pid
    
    def init_prior(self, transition: list) -> list:

        qnet_script = torch.jit.script(self.q_network.eval())
        
        batch = Transition(*zip(*transition))
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        next_state = torch.cat(batch.next_state)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.done)
        
        qvalue = qnet_script(state)
        action_onehot = torch.eye(self.action_space)[action]
        Q = torch.sum(qvalue * action_onehot, dim=1, keepdim=True).squeeze()
        next_qvalue = qnet_script(next_state)
        next_action = torch.argmax(next_qvalue, dim=1)# argmaxQ
        next_action_onehot = torch.eye(self.action_space)[next_action]
        next_maxQ = torch.sum(next_qvalue * next_action_onehot, dim=1, keepdim=True)
        reward_sum = torch.zeros_like(reward[:, 0].unsqueeze(1))
        for r in range(self.advanced_step):
            reward_sum += self.gamma ** r * reward[:, r].unsqueeze(1)
        TQ = (reward_sum + self.gamma ** self.advanced_step * (1 - done.int().unsqueeze(1)) * next_maxQ).squeeze()
        td_error = torch.square(Q - TQ)
        td_errors = td_error.cpu().detach().numpy().flatten()

        return td_errors, transition
    
    def get_action_space(self) -> int:
        
        return self.action_space