
from copy import deepcopy
import ray
import gym
import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import datetime
import matplotlib.pyplot as plt
from util import *
from model import ActorNetwork2, CriticNetwork2
from env import Agent
from ReplayMemory import ExperienceReplay
import argparse

parser = argparse.ArgumentParser(description="DRL setting")

parser.add_argument("--seed", type=int, default=123456, help="random seed value")
parser.add_argument("--capacity", type=int, default=30000, help="experience replay memory size")
parser.add_argument("--min_replay", type=int, default=300, help="update start memory size")
parser.add_argument("--gamma", type=float, default=0.99, help="discount rate")
parser.add_argument("--batch_size", type=int, default=160, help="batch size")
parser.add_argument("--n_step", type=int, default=1, help="n step")
parser.add_argument("--reward_clip", action='store_true', help="reward clip(False)")
parser.add_argument("--tau", type=float, default=0.02 ,help="reward clip(False)")

# 結果を受ける
args = parser.parse_args()

class DDPGAgent():

    CAPACITY = args.capacity
    MIN_REPLAY = args.min_replay
    GAMMA = args.gamma
    BATCH = args.batch_size
    TAU = args.tau

    def __init__(self, env_id, action_space, n_envs=1):

        self.env_id = env_id
        self.n_envs = n_envs
        self.env = Agent.remote(env_id)
        self.Actor = ActorNetwork2(input_size=3, action_space=action_space)
        self.targetActor = ActorNetwork2(input_size=3, action_space=action_space)
        self.actorOptim = Adam(self.Actor.parameters(), lr=self.Actor.lr)
        self.Critic = CriticNetwork2(input_size=3)
        self.targetCritic = CriticNetwork2(input_size=3)
        self.criticOptim = Adam(self.Critic.parameters(), lr=self.Critic.lr)
        self.memory = ExperienceReplay.remote(args)
        self.path = f'graph/DDPG_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        self.writer = SummaryWriter(f'./{self.path}')
        self.update = 0
        self.total_steps = 0
        self.stdev = 0.2
    
    def run(self, n_updates: int):

        history = {"steps": [], "scores": []}
        state = torch.FloatTensor(ray.get(self.env.reset.remote()))
        losses = []
        reward_sum = [0]

        while self.update < n_updates:

            action = self.Actor.sample_action(state, self.stdev)
            next_state = torch.FloatTensor(ray.get(self.env.step.remote(action[0])))
    
            traj = ray.get(self.env.get_trajectory.remote())
            assert len(traj['state']) == 1
            
            self.memory.push.remote(Transition(traj['state'],
                                                traj['action'],
                                                traj['next_state'],
                                                traj['reward'],
                                                traj['done']))

            reward_sum[-1] += traj['reward'][0][0]
            if traj['done']: 
                reward_sum.append(0)

            state = deepcopy(next_state)
            self.total_steps += 1
            
            loss = self.update_network()
            if loss: losses.append(loss)
            if self.total_steps % 4 == 0 and self.update > 0: self.update_target_network()
            
            if self.total_steps % 160 == 0 and self.update > 0:
                test_scores = np.array(self.play(n=5))
                history["steps"].append(self.total_steps)
                history["scores"].append(test_scores)
                ma_score = sum(history["scores"][-10:]) / 10
                self.writer.add_scalar("test_score", test_scores, self.update / 10)
                print(f"Epoch {self.update / 10}, {self.total_steps//1000}K, {test_scores}")

            if self.update % 10 == 0 and self.update > 0:
                #print(losses)
                self.writer.add_scalar("value_loss", np.array(losses).mean(), self.update / 10)
                losses = []
                #print(reward_sum)
                self.writer.add_scalar("train_score", np.array(reward_sum).mean(), self.update / 10)
                reward_sum = [0]

        return history

    def update_network(self):

        if ray.get(self.memory.get_memory_size.remote()) < self.MIN_REPLAY:
            return

        self.update += 1
        (states, actions, next_states, rewards, masks) = ray.get(self.memory.sample.remote(self.BATCH))

        self.Critic.eval()
        self.targetCritic.eval()
        self.Actor.eval()
        self.targetActor.eval()
        #Update Critic-Network
        #target Q-value
        with torch.no_grad():
            next_actions = self.targetActor(next_states)
            next_qvalues = self.targetCritic(next_states, next_actions)
            target_qvalues = (rewards + self.GAMMA ** (1 - masks.int().unsqueeze(1)) * next_qvalues)

        #Q-value
        qvalues = self.Critic(states, actions)
        
        self.Critic.train()
        loss = torch.mean(torch.square(target_qvalues - qvalues))
        self.criticOptim.zero_grad(set_to_none=True)
        loss.backward()
        self.criticOptim.step()

        #Update Actor-Network
        J = -1 * torch.mean(self.Critic(states, self.Actor(states)))
        self.Actor.train()
        self.actorOptim.zero_grad(set_to_none=True)
        J.backward()
        self.actorOptim.step()

        return loss.detach()

    def soft_update(self, target, source):
        
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.TAU) + param.data * self.TAU)

    def update_target_network(self):
        
        self.soft_update(self.targetActor, self.Actor)

        self.soft_update(self.targetCritic, self.Critic)
    
    def play(self, n=1, monitordir=None, verbose=False):

        env = gym.make(self.env_id)

        total_rewards = []

        for i in range(n):
            if verbose: print(f"start_episode: {i}")
            state = torch.FloatTensor(env.reset())
            done = False
            total_reward = 0

            while not done:

                action = self.Actor.sample_action(state)

                #if verbose:
                    #mean, sd = self.policy(state)
                    #print(action, mean.detach().numpy(), sd.detach().numpy())
                if verbose: env.render()
                next_state, reward, done, _ = env.step(action[0])
                total_reward += reward
                
                if done:
                    break
                else:
                    state = torch.FloatTensor(next_state)

            total_rewards.append(total_reward)

        if verbose: env.close()

        
        return sum(total_rewards) // n

def main(env_id, action_space):

    agent = DDPGAgent(env_id=env_id, action_space=action_space,)

    MONITOR_DIR = Path(__file__).parent / agent.path

    history = agent.run(n_updates=50000)
    #print(history)
    plt.plot(history["steps"], history["scores"])
    plt.xlabel("steps")
    plt.ylabel("Total rewards")
    plt.savefig(MONITOR_DIR / "testplay.png")
    
    print(f"reward : {agent.play(n=5, monitordir=MONITOR_DIR, verbose=True)}")

if __name__ == "__main__":

    ray.init(local_mode=False)

    env_id = "Pendulum-v1"
    action_space = 1
    main(env_id, action_space)

    ray.shutdown()