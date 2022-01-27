import torch
from torch.nn import parameter
import ray
import gym
from collections import deque, namedtuple
from model import Net
from utils import preproccess
import numpy as np
from dataclasses import dataclass

ENV = "BreakoutDeterministic-v4"

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool

class Local_Buffer:
    def __init__(self, advanced_step: int, gamma: float, reward_clip: bool):
        self.buffer = []
        self.n_step = advanced_step
        self.reward_clip = reward_clip
        self.temp_buffer = deque(maxlen=self.n_step)
        self.gamma = gamma
    
    def __len__(self):

        return len(self.buffer)
    
    def push(self, transition):
        '''
            transition : tuple(state, action, reward, next_state, done)
        '''
        self.temp_buffer.append(Experience(*transition))
        if len(self.temp_buffer) == self.n_step:

            nstep_reward = 0
            has_done = False
            for i, onestep_exp in enumerate(self.temp_buffer):
                reward, done = onestep_exp.reward, onestep_exp.done
                reward = np.clip(reward, -1, 1) if self.reward_clip else reward
                nstep_reward += self.gamma ** i * (1 - done) * reward
                if done:
                    has_done = True
                    break
            
            nstep_exp = Transition(self.temp_buffer[0].state,
                                    torch.IntTensor([self.temp_buffer[0].action]),
                                    self.temp_buffer[-1].next_state,
                                    torch.FloatTensor([[nstep_reward]]),
                                    torch.BoolTensor([has_done]))
            
            self.buffer.append(nstep_exp)
        
    def pull(self):
        experiences = self.buffer
        self.buffer = []
        
        return experiences

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
        self.frames = deque(maxlen=self.n_frame)
        self.advanced_step = advanced_step
        self.local_buffer = Local_Buffer(advanced_step=self.advanced_step, gamma=self.gamma, reward_clip=False)
        state = preproccess(self.env.reset())
        for _ in range(self.n_frame):
            self.frames.append(state)
        self.episode_reward = 0
        self.local_cycle = local_cycle
        self.lives = 5
        self.episode = 0
    
    def rollout(self, weights: parameter) -> list:

        #print("Start:{}-Environment".format(self.pid))

        self.q_network.load_state_dict(weights) 
        buffer = []
        
        state = torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...])
        for _ in range(self.local_cycle):
            
            action = self.q_network.get_action(state, epsilon=self.epsilon)
            next_frame, reward, done, info = self.env.step(action)
            self.frames.append(preproccess(next_frame))
            next_state = torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...])
            self.episode_reward += reward
            state = next_state
            if self.lives != info["ale.lives"]:
                transition = Transition(state, action, next_state, reward, True)
                self.lives = info["ale.lives"]
            else:
                transition = Transition(state, action, next_state, reward, done)
            self.local_buffer.push(transition)

            if done:
                self.env_reset()

        buffer = self.local_buffer.pull()

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
            td_errors = (Q - TQ).detach().numpy().flatten()

        return td_errors, transition
    
    def get_action_space(self) -> int:
        
        return self.action_space
    
    def env_reset(self) -> None:

        state = preproccess(self.env.reset())
        for _ in range(self.n_frame):
            self.frames.append(state)
        self.episode_reward = 0
        self.lives = 5
        self.episode = 0


@ray.remote
class Tester:

    def __init__(self, action_space: int, n_frame: int):
        self.q_network = Net(action_space)
        self.epsilon = 0.05
        self.env_name = ENV
        self.env = gym.make(self.env_name)
        self.n_frame = n_frame
        self.frames = deque(maxlen=self.n_frame)
    
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
            new_frame, reward, done, _ = self.env.step(action)
            total_reward += reward
            episode += 1
            if episode > 1000 and total_reward < 10:
                break
            self.frames.append(preproccess(new_frame))
            state = torch.FloatTensor(np.stack(self.frames, axis=0)[np.newaxis, ...])
        
        return total_reward, episode, step

if __name__=="__main__":
    pass