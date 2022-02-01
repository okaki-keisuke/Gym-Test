import zlib
import pickle
from collections import namedtuple, deque
import gym
from Actor import ENV
from utils import preproccess
import torch
import numpy as np

env = gym.make(ENV)
state = env.reset()
state = preproccess(state)
frames = deque(maxlen=4)
for _ in range(4):
    frames.append(state)

state = torch.FloatTensor(np.stack(frames, axis=0)[np.newaxis, ...])

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

transition = Transition(state, 1, state, 0.0, True) 
#print(transition)
experiences = zlib.compress(pickle.dumps(transition))
#print(experiences)
zlib_transition = pickle.loads(zlib.decompress(experiences))
#print(zlib_transition)
print((transition.state == zlib_transition.state).all())
print(transition.action == zlib_transition.action)
print((transition.next_state == zlib_transition.next_state).all())
print(transition.reward == zlib_transition.reward)
print(transition.done == zlib_transition.done)