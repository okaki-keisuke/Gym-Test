from importlib_metadata import collections
import torch
from torch.nn import parameter
import copy
import ray
import gym
from collections import deque, namedtuple
from model import Net
from utils import preproccess
import numpy as np

ENV = "BreakoutDeterministic-v4"

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Agent:
    def __init__(self, advanced_step: int, gamma: float):
        self.multi_step = advanced_step
        self.state = collections.deque(maxlen=self.multi_step)
        self.store_reward = []
        self.reward_nstep = 0
        self.store_state = []
        self.gamma = gamma

    def state_storage(self, statorch: torch.Tensor) -> torch.Tensor:
        
        self.state.append(statorch)
        

    def reward_storage(self, reward: np) -> None:
        self.store_reward.append(reward)
        if len(self.store_reward) == self.multi_step + 1:
            del self.store_reward[0]
            self.reward_nstep = 0
            for step in range(self.multi_step):
                self.reward_nstep += self.gamma ** step * self.store_reward[step]
        
    def data_full(self) -> bool:
        if len(self.state) == self.multi_step:
            return True
        
        return False
    
    def reset(self):
        self.state.clear()
        self.store_reward = []
        self.store_state = []
        self.reward_nstep = 0


@ray.remote
class Environment:
    def __init__(self, pid: int, gamma: float, advanced_step: int, epsilon: float, local_cycle: int, n_frame: int):
        self.pid = pid
        self.env_name = ENV
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space.n
        self.q_network = Net(action_space=self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_frame = n_frame
        self.frames = collections.deque(maxlen=self.n_frame)
        self.advanced_step = advanced_step
        self.agent = Agent(advanced_step=self.advanced_step, gamma=self.gamma)
        state = preproccess(self.env.reset())
        for _ in range(self.n_frame):
            self.frames.append(state)
        self.agent.state_storage(torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...]))
        self.episode_reward = 0
        self.local_cycle = local_cycle
    
    def rollout(self, weights: parameter) -> list:

        #print("Start:{}-Environment".format(self.pid))

        self.q_network.load_state_dict(weights) 
        buffer = []
        
        state = torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...])
        for _ in range(self.local_cycle):
            
            action = self.q_network.get_action(state, epsilon=self.epsilon)
            next_frame, reward, done, _ = self.env.step(action)
            self.frames.append(preproccess(next_frame))
            next_state = torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...])
            self.episode_reward += reward
            self.agent.reward_storage(reward)
            if self.agent.data_full():
                transition = Transition(self.agent.state.popleft(),
                                        torch.LongTensor([action]),
                                        next_state, 
                                        torch.FloatTensor([[self.agent.reward_nstep]]),
                                        torch.BoolTensor([done])
                                        )
                buffer.append(transition)

            if done:
                self.agent.reset()
                state = preproccess(self.env.reset())
                for _ in range(self.n_frame):
                    self.frames.append(state)
                self.agent.state_storage(torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...]))
                self.episode_reward = 0
            
            self.agent.state_storage(next_state)
            state = next_state
        td_error , transitions = self.init_prior(buffer)

        return td_error, transitions, self.pid
    
    def init_prior(self, transition: list) -> list:

        self.q_network.eval()
        with torch.no_grad():
            
            batch = Transition(*zip(*transition))
            state = torch.cat(batch.state)
            action = torch.cat(batch.action)
            next_state = torch.cat(batch.next_state)
            reward = torch.cat(batch.reward)
            done = torch.cat(batch.done)
            
            qvalue = self.q_network(state)
            action_onehot = torch.eye(self.action_space)[action]
            Q = torch.sum(qvalue * action_onehot, dim=1, keepdim=True).squeeze()
            next_qvalue = self.q_network(next_state)
            next_action = torch.argmax(next_qvalue, dim=1)# argmaxQ
            next_action_onehot = torch.eye(self.action_space)[next_action]
            next_maxQ = torch.sum(next_qvalue * next_action_onehot, dim=1, keepdim=True)
            TQ = (reward + self.gamma ** self.advanced_step * (1 - done.int().unsqueeze(1)) * next_maxQ).squeeze()
            td_error = torch.square(Q - TQ)
            td_errors = td_error.detach().numpy().flatten()

        return td_errors, transition
    
    def get_action_space(self) -> int:
        
        return self.action_space

@ray.remote
class Tester:

    def __init__(self, action_space: int, n_frame: int):
        self.q_network = Net(action_space)
        self.epsilon = 0.01
        self.env_name = ENV
        self.env = gym.make(self.env_name)
        self.n_frame = n_frame
        self.frames = collections.deque(maxlen=self.n_frame)


    
    def test_play(self, current_weights: parameter, step: int) -> list:
        self.q_network.load_state_dict(current_weights)
        state = preproccess(self.env.reset())
        for _ in range(self.n_frame):
            self.frames.append(state)
        state = torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...])
        total_reward = 0
        episode = 0
        done = False
        while not done:
            action = self.q_network.get_action(state=state, epsilon=self.epsilon)
            new_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            episode += 1
            if episode > 1000 and total_reward < 10:
                break
            self.frames.append(preproccess(new_state))
            state = torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...])
        
        return total_reward, step

if __name__=="__main__":
    pass