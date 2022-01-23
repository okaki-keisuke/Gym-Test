import torch
from utils import get_initial_state, input_image, initial_state, input_state, preproccess
import gym
from Breakout import ENV
import time
import numpy as np
import collections
import torchvision

env = gym.make(ENV)
state = env.reset()

start = time.time()
state_pil = initial_state(state)
print("======initial_state========")
print(time.time() - start)
print(state_pil.shape)

start = time.time()
state_pil = input_states(state, state_pil)
print("======input_states========")
print(time.time() - start)
print(state_pil.shape)

start= time.time()
state_torch = get_initial_state(state)
print("======get_initial_state========")
print(time.time() - start)
print(state_torch.shape)


start = time.time()
state_pil = input_image(state, state_torch)
print("======input_image========")
print(time.time() - start)
print(state_pil.shape)


start = time.time()
state_pil = initial_state(state)
print("=====initial_state=========")
print(time.time() - start)
print(state_pil.shape)

state_list = []
for _ in range(4):
    state_list.append(state)

start = time.time()
state_pil = input_state(state) 
print("=====input_state=========")
print(state_pil.shape)

frame = collections.deque(maxlen=4)
for _ in range(4):
    frame.append(state_pil)
state_np = np.stack(frame, axis=0)[np.newaxis, ...]
state_torch = torch.FloatTensor(state_np)
print(state_torch.shape)
print(time.time() - start)
