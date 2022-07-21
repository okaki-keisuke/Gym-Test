
from time import strftime
import gym
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import datetime
import matplotlib.pyplot as plt
from util import *
from model import ActorNetwork, CriticNetwork
from env import VecEnv

class PPOAgent():

    GAMMA = 0.9
    GAE_LAMBDA = 0.95
    CLIPRANGE = 0.2
    OPT_ITER = 10

    
    def __init__(self, env_id, action_space, n_envs=1, trajectory_size=200):

        self.env_id = env_id
        self.n_envs = n_envs
        self.trajectory_size = trajectory_size
        self.vecenv = VecEnv(env_id=self.env_id, n_envs=self.n_envs)
        self.policy = ActorNetwork(input_size=3, action_space=action_space)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.policy.lr)
        self.critic = CriticNetwork(input_size=3)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic.lr)
        self.r_running_stats = RunningStats(shape=action_space)
        self.path = f'graph/PPO_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        self.writer = SummaryWriter(f'./{self.path}')

    def run(self, n_updates):

        history = {"steps": [], "scores": []}
        states = torch.FloatTensor(self.vecenv.reset())
        hiscore = None

        for epoch in range(n_updates):

            for _ in range(self.trajectory_size):
                
                actions = self.policy.sample_action(states)
                
                next_states = self.vecenv.step(actions)
                states = torch.FloatTensor(next_states)

            trajectories = self.vecenv.get_trajectories()

            for trajectory in trajectories:
                self.r_running_stats.update(trajectory["reward"])

            trajectories = self.compute_advantage(trajectories)

            states_batch, actions_batch, advantages, vtargs = self.create_minibatch(trajectories)
            
            vloss = self.update_critic(states_batch, vtargs)

            self.update_policy(states_batch, actions_batch, advantages)

            global_steps = (epoch+1) * self.trajectory_size * self.n_envs
            train_scores = np.array([traj["reward"].sum() for traj in trajectories])

            if epoch % 20 == 0:
                test_scores = np.array(self.play(n=5))
                history["steps"].append(global_steps)
                history["scores"].append(test_scores.mean())
                ma_score = sum(history["scores"][-10:]) / 10
                self.writer.add_scalar("test_score", test_scores.mean(), epoch)
                print(f"Epoch {epoch}, {global_steps//1000}K, {test_scores.mean()}")

            
            self.writer.add_scalar("value_loss", vloss, epoch)
            self.writer.add_scalar("train_score", train_scores.mean(), epoch)

        return history

    def compute_advantage(self, trajectories):

        """
            Generalized Advantage Estimation (GAE, 2016)
        """

        for trajectory in trajectories:
            
            #V(s)
            trajectory["v_pred"] = self.critic(trajectory["state"]).detach().numpy()

            trajectory["v_pred_next"] = self.critic(trajectory["next_state"]).detach().numpy()

            is_nonterminals = 1 - trajectory["done"]

            #normed_rewards = ((trajectory["r"] - self.r_running_stats.mean) / (np.sqrt(self.r_running_stats.var) + 1e-4))
            normed_rewards = (trajectory["reward"] / (np.sqrt(self.r_running_stats.var) + 1e-4))

            deltas = normed_rewards + self.GAMMA * is_nonterminals * trajectory["v_pred_next"] - trajectory["v_pred"]

            advantages = np.zeros_like(deltas, dtype=np.float32)

            lastgae = 0
            for i in reversed(range(len(deltas))):
                lastgae = deltas[i] + self.GAMMA * self.GAE_LAMBDA * is_nonterminals[i] * lastgae
                advantages[i] = lastgae

            trajectory["advantage"] = advantages

            trajectory["R"] = advantages + trajectory["v_pred"]

            """経験的return
            trajectory["R"] = np.zeros_like(trajectory["r"])
            R = (1 - trajectory["done"][-1]) * trajectory["v_pred_next"][-1]
            for i in reversed(range(trajectory["r"].shape[0])):
                R = trajectory["r"][i] / reward_std + (1 - trajectory["done"][i]) * self.GAMMA * R
                trajectory["R"][i] = R
            """

        return trajectories     

    def update_policy(self, states, actions, advantages):
            
        for _ in range(self.OPT_ITER):
            
            #self.policy.eval()
            
            with torch.no_grad():
                old_means, old_stdevs = self.policy(states)
                old_logprob = self.compute_logprob(old_means, old_stdevs, actions)
            
            new_means, new_stdevs = self.policy(states)
            new_logprob = self.compute_logprob(new_means, new_stdevs, actions)

            ratio = torch.exp(new_logprob - old_logprob)
            ratio_clipped = torch.clamp(ratio, 1 - self.CLIPRANGE, 1 + self.CLIPRANGE)

            loss_unclipped = ratio * advantages
            loss_clipped = ratio_clipped * advantages
            loss = torch.min(loss_unclipped, loss_clipped)
            loss = -1 * torch.mean(loss)

            #self.policy.train()

            self.policy_optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.policy_optim.step()

    def update_critic(self, states, v_targs):

        losses = []

        for _ in range(self.OPT_ITER):

            #self.critic.eval()
            
            with torch.no_grad():
                old_vpred = self.critic(states)
            
            vpred = self.critic(states)
            vpred_clipped = old_vpred + torch.clamp(vpred - old_vpred, -self.CLIPRANGE, self.CLIPRANGE)
            loss = torch.max(torch.square(v_targs - vpred), torch.square(v_targs - vpred_clipped))
            loss = torch.mean(loss)

            #self.critic.train()
            
            self.critic_optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optim.step()

            losses.append(loss.detach().numpy())

        return np.array(losses).mean()

    def compute_logprob(self, means, stdevs, actions):
        """ガウス分布の確率密度関数よりlogp(x)を計算
            logp(x) = -0.5 log(2π) - log(std)  -0.5 * ((x - mean) / std )^2
        """
        logprob = - 0.5 * np.log(2*np.pi)
        logprob += - torch.log(stdevs)
        logprob += - 0.5 * torch.square((actions - means) / stdevs)
        logprob = torch.sum(logprob, axis=1, keepdims=True)
        return logprob

    def create_minibatch(self, trajectories):

        states = np.vstack([traj["state"] for traj in trajectories])
        actions = np.vstack([traj["action"] for traj in trajectories])

        advantages = np.vstack([traj["advantage"] for traj in trajectories])

        v_targs = np.vstack([traj["R"] for traj in trajectories])

        return torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(advantages), torch.FloatTensor(v_targs)

    def play(self, n=1, monitordir=None, verbose=False):

        env = gym.make(self.env_id)

        total_rewards = []

        for i in range(n):
            if verbose: print(f"start_episode: {i}")
            state = torch.FloatTensor(env.reset())
            done = False
            total_reward = 0

            while not done:

                action = self.policy.sample_action(state)

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


    agent = PPOAgent(env_id=env_id, action_space=action_space,
                     n_envs=10, trajectory_size=16)
    
    MONITOR_DIR = Path(__file__).parent / agent.path

    history = agent.run(n_updates=5000)

    plt.plot(history["steps"], history["scores"])
    plt.xlabel("steps")
    plt.ylabel("Total rewards")
    plt.savefig(MONITOR_DIR / "testplay.png")
    
    print(f"reward : {agent.play(n=5, monitordir=MONITOR_DIR, verbose=True)}")

if __name__ == "__main__":
    
    env_id = "Pendulum-v1"
    action_space = 1
    main(env_id, action_space)