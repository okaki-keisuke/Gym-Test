import numpy as np
import collections
import torch
import random
from dataclasses import dataclass
from util import *
import ray

@dataclass
class Experience:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    reward: float
    mask: float

Transition_2 = collections.namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'mask'))

@ray.remote
class ExperienceReplay:
    def __init__(self, args):

        random.seed(args.seed)
        self.capacity = args.capacity
        self.local_buffer = LocalBuffer(args)
        self.buffer = []
        self.position = 0

    def push(self, transition):

        n_transition = self.local_buffer.push(transition=transition)
        
        if n_transition is not None:
        
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
        
            exp = zlib.compress(pickle.dumps(n_transition))
            self.buffer[self.position] = exp
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        
        experiences = random.sample(self.buffer, batch_size)
        #dataをzipで圧縮していた

        transitions = [pickle.loads(zlib.decompress(exp)) for exp in experiences]

        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        next_states = torch.cat(batch.next_state)
        rewards = torch.cat(batch.reward)
        masks = torch.cat(batch.mask)
        
        return (states, actions, next_states, rewards, masks)

    def get_memory_size(self):
        
        return len(self.buffer)
    
    def nstep_reset(self):
        self.local_buffer.reset()

    def ready(self):
        pass

class LocalBuffer: 
    def __init__(self, args):
        self.buffer = []
        self.n_step = args.n_step
        self.reward_clip = args.reward_clip
        self.temp_buffer = collections.deque(maxlen=self.n_step)
        self.gamma = args.gamma
        self.is_full = False
    
    def __len__(self):

        return len(self.buffer)
    
    def push(self, transition):
        '''
            transition : tuple(state, action, next_state, reward, mask)
        '''
        self.temp_buffer.append(Experience(*transition))
        if len(self.temp_buffer) == self.n_step:
            self.is_full = True
            nstep_reward = 0
            for i, onestep_exp in enumerate(self.temp_buffer):
                reward, mask = onestep_exp.reward, onestep_exp.mask
                reward = np.clip(reward, -1, 1) if self.reward_clip else reward
                nstep_reward += self.gamma ** i * (1 - int(mask)) * reward
                if 1 - int(mask) == 0:
                    break
            
            nstep_exp = Transition(self.temp_buffer[0].state,
                                    self.temp_buffer[0].action,  
                                    self.temp_buffer[-1].next_state,
                                    nstep_reward,
                                    self.temp_buffer[-1].mask)
            
            #self.buffer.append(nstep_exp)
        
        return nstep_exp if self.is_full else None

    def pull(self):

        assert len(self.buffer) == 1
        
        data = self.buffer[0]
        self.buffer = self.buffer[1:]
        
        return data
    
    def reset(self):

        self.temp_buffer.clear()
        self.is_full = False

if __name__ == "__main__":
    pass