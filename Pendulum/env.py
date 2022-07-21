import numpy as np
import ray
import gym
import torch

import warnings
warnings.filterwarnings("ignore")



@ray.remote
class Agent:

    def __init__(self, env_id):

        self.env = gym.make(env_id)
        self.trajectory = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}

    def reset(self):

        self.state = self.env.reset()
        return self.state

    def step(self, action):

        state = self.state
        next_state, reward, done, _ = self.env.step(action)

        if done:
            next_state = self.env.reset()

        self.trajectory["state"].append(state)
        self.trajectory["action"].append(action)
        self.trajectory["reward"].append(reward)
        self.trajectory["next_state"].append(next_state)
        self.trajectory["done"].append(done)

        self.state = next_state

        return next_state

    def get_trajectory(self):

        trajectory = self.trajectory

        trajectory["state"] = torch.FloatTensor(np.array(trajectory["state"], dtype=np.float32))
        trajectory["action"] = torch.FloatTensor(np.array(trajectory["action"], dtype=np.float32))
        trajectory["reward"] = torch.FloatTensor(np.array(trajectory["reward"], dtype=np.float32).reshape(-1, 1))
        trajectory["next_state"] = torch.FloatTensor(np.array(trajectory["next_state"], dtype=np.float32))
        trajectory["done"] = torch.FloatTensor(np.array(trajectory["done"], dtype=np.float32).reshape(-1, 1))

        self.trajectory = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}

        return trajectory


class VecEnv:

    def __init__(self, env_id, n_envs):

        ray.init()

        self.env_id = env_id
        self.n_envs = n_envs
        self.agents = [Agent.remote(self.env_id) for _ in range(self.n_envs)]

    def step(self, actions):

        next_states = ray.get(
            [agent.step.remote(action) for agent, action in zip(self.agents, actions)])

        return np.array(next_states)

    def reset(self):

        states = ray.get([agent.reset.remote() for agent in self.agents])

        return np.array(states)

    def get_trajectories(self):

        trajectories = ray.get([agent.get_trajectory.remote() for agent in self.agents])

        return trajectories

    def __len__(self):

        return self.n_envs

    def __del__(self):
        
        ray.shutdown()