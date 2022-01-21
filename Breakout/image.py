import torch
from utils import get_initial_state, input_image, initial_state, input_state
import gym
from Breakout import ENV
import time
import numpy as np


env = gym.make(ENV)
state = env.reset()

start = time.time()
state_pil = initial_state(state)
print("==============")
print(time.time() - start)
print(state_pil.shape)

start = time.time()
state_pil = input_state(state, state_pil)
print("==============")
print(time.time() - start)
print(state_pil.shape)

start= time.time()
state_torch = get_initial_state(state)
print("==============")
print(time.time() - start)
print(state_torch.shape)


start = time.time()
state_pil = input_image(state, state_torch)
print("==============")
print(time.time() - start)
print(state_pil.shape)


start = time.time()
state_pil = initial_state(state)
print("==============")
print(time.time() - start)
print(state_pil.shape)

start = time.time()
state_pil = input_state(state, state_pil)
print("==============")
print(time.time() - start)
print(state_pil.shape)