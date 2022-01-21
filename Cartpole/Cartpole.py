from typing import Tuple
import gym
import numpy as np
from collections import namedtuple
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import parameter, utils
from torch.utils.tensorboard import SummaryWriter
import datetime
import ray
from priority_tree import Tree
from model import Net, ACTION
import argparse
from tqdm import tqdm
import os

ENV = 'CartPole-v0' 

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.backends.cudnn.benchmark = True

def arg_get() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="myenv setting")

    parser.add_argument("--graph", action="store_true", help="show Graph")
    parser.add_argument("--save", action="store_true",help="save parameter")
    parser.add_argument("--gamma", default=0.98, type=float, help="learning rate")
    parser.add_argument("--batch", default=64, type=int, help="batch size")
    parser.add_argument("--capacity", default=2 ** 14, type=int, help="Replay memory size (2 ** x)")
    parser.add_argument("--epsilon", default=0.5, type=float, help="exploration rate")
    parser.add_argument("--advanced", default=5, type=int, help="number of advanced step")
    parser.add_argument("--td_epsilon", default=0.001, type=float, help="td error epsilon")
    parser.add_argument("--interval", default=5, type=int, help="Test interval")
    parser.add_argument("--update", default=200, type=int, help="number of update")

    # 結果を受ける
    args = parser.parse_args()

    return(args)
    
@ray.remote
class Environment:
    def __init__(self, pid: int, gamma: float, epsilon: float):
        self.pid = pid
        self.env_name = ENV
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space.n
        self.q_network = Net()
        self.epsilon = epsilon 
        self.gamma = gamma
        state = self.env.reset()
        self.state = torch.from_numpy(np.atleast_2d(state).astype(np.float32)).clone()
        self.episode_reward = 0
    
    def rollout(self, weights: parameter) -> list:

        self.q_network.load_state_dict(weights)
        buffer = []
        
        for _ in range(100):
            action = self.q_network.get_action(state=self.state, epsilon=self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.episode_reward += reward
            next_state = torch.from_numpy(np.atleast_2d(next_state).astype(np.float32)).clone()
            transition = Transition(self.state, torch.LongTensor([[action]]), next_state, torch.FloatTensor([reward]), torch.BoolTensor([done]))
            buffer.append(transition)
            if done:
                state = self.env.reset()
                self.state = torch.from_numpy(np.atleast_2d(state).astype(np.float32)).clone()
                self.episode_reward = 0
            else:
                self.state = next_state
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

        qvalue = qnet_script(state)# Q(s,a)-value
        action_onehot = torch.eye(self.action_space)[action.squeeze()]
        Q = torch.sum(qvalue * action_onehot, dim=1, keepdim=True).squeeze()
        next_qvalue = qnet_script(next_state)
        next_action = torch.argmax(next_qvalue, dim=1)# argmaxQ
        next_action_onehot = torch.eye(self.action_space)[next_action]
        next_maxQ = torch.sum(next_qvalue * next_action_onehot, dim=1, keepdim=True)
        TQ = (reward.unsqueeze(1) + self.gamma * (1 - done.int().unsqueeze(1)) * next_maxQ).squeeze()
        td_error = torch.square(Q - TQ)
        td_errors = td_error.detach().numpy().flatten()

        return td_errors, transition

class Experiment_Replay:

    def __init__(self, capacity: int, td_epsilon: float):
        self.capacity = capacity
        self.priority = Tree(capacity=self.capacity)
        self.memory = [None] * self.capacity
        self.index = 0
        self.is_full = False
        self.alpha = 0.6
        self.beta = 0.4
        self.td_epsilon = td_epsilon

    def push(self, td_errors: list, transitions: list) -> None:
        assert len(td_errors) == len(transitions)
        priorities = (np.abs(td_errors) + self.td_epsilon) ** self.alpha
        for priority, transition in zip(priorities, transitions):
            self.memory[self.index] = transition
            self.priority[self.index] = priority
            self.index += 1
            if self.capacity == self.index:
                self.index = 0
                self.is_full = True

    def update_priority(self, sampled_index: list, td_errors: list) -> None:
        assert len(sampled_index) == len(td_errors)
        for idx, td_error in zip(sampled_index, td_errors):
            priority = (abs(td_error) + self.td_epsilon) ** self.alpha
            self.priority[idx] = priority ** self.alpha

    def sample(self, batch_size: int) -> list:
        #index
        samples_index = [self.priority.sample() for _ in range(batch_size)]
        #weight
        weights = []
        current_size = len(self.memory) if self.is_full else self.index
        for idx in samples_index:
            prob = self.priority[idx] / self.priority.sum()
            weight = (prob * current_size) ** (-self.beta)
            weights.append(weight)
        weights = np.array(weights) / max(weights)
        #sample
        experience = [self.memory[idx] for idx in samples_index]
    
        return samples_index, weights, experience

    def __len__(self) -> int:
        return len(self.memory)

@ray.remote(num_gpus=0.5)
class Learner:
    def __init__(self, num_actions: int, batch_size: int, gamma: float):
        self.num_action = num_actions
        self.main_q_network = Net().to(device)
        self.target_q_network = Net().to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 30, gamma=0.5)
        self.batch_size = batch_size
        self.gamma = gamma
    
    def define_network(self) -> parameter:
        current_weights = self.main_q_network.to('cpu').state_dict()
        self.main_q_network.to(device)  
        self.update_target_q_network()  
        return current_weights

    def update(self, minibatch: Tuple) -> list:
        
        index_all = []
        td_error_all = []

        for (index, weight, transition) in minibatch:

            batch = Transition(*zip(*transition))
            state_batch = torch.cat(batch.state).to(device)
            action_batch = torch.cat(batch.action).to(device)
            reward_batch = torch.cat(batch.reward).to(device)
            next_state_batch = torch.cat(batch.next_state).to(device)
            done_batch = torch.cat(batch.done).to(device)
            weights = torch.from_numpy(np.atleast_2d(weight).astype(np.float32)).clone().to(device)
             
            self.main_q_network.eval()
            self.target_q_network.eval()

            #TDerror
            #with torch.no_grad():
            next_qvalue = self.main_q_network(next_state_batch)
            next_action = torch.argmax(next_qvalue, dim=1)# argmaxQ
            next_action_onehot = torch.eye(self.num_action)[next_action].to(device)
            target_qvalue = self.target_q_network(next_state_batch)
            next_maxQ = torch.sum(target_qvalue * next_action_onehot, dim=1, keepdim=True)
            TQ = (reward_batch.unsqueeze(1) + self.gamma* (1 - done_batch.int().unsqueeze(1)) * next_maxQ).squeeze()
        
            qvalue = self.main_q_network(state_batch)
            action_onehot = torch.eye(self.num_action)[action_batch.squeeze()].to(device)
            Q = torch.sum(qvalue * action_onehot, dim=1, keepdim=True).squeeze()#Q(s,a)-value
            
            td_error = torch.square(Q - TQ)
            td_errors = td_error.cpu().detach().numpy().flatten()

            self.main_q_network.train()
            #print(TQ.dtype, Q.dtype)
            loss = torch.mean(weights * td_error)
            self.optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(self.main_q_network.parameters(), max_norm=40.0, norm_type=2.0)

            self.optimizer.step()   

            index_all += index
            td_error_all += td_errors.tolist()  
        
        self.scheduler.step()
        current_weight = self.main_q_network.to('cpu').state_dict()
        self.main_q_network.to(device)

        return current_weight, index_all, td_error_all

    def update_target_q_network(self) -> None:
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

@ray.remote
class Tester:

    def __init__(self):
        self.q_network = Net()
        self.epsilon = 0.01
    
    def test_play(self, current_weights: parameter, step: int) -> list:
        self.q_network.load_state_dict(current_weights)
        env = gym.make(ENV)
        observation = env.reset()
        observation = torch.from_numpy(np.atleast_2d(observation).astype(np.float32)).clone()
        total_reward = 0
        done = False
        while not done:
            action = self.q_network.get_action(state=observation, epsilon=self.epsilon)
            new_observation, reward, done, _ = env.step(action)
            total_reward += reward
            observation = torch.from_numpy(np.atleast_2d(new_observation).astype(np.float32)).clone()
        
        return total_reward, step
    
def main(num_envs: int) -> None:

    args = arg_get()
    if args.save:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%-M")
        os.makedirs('/home/mukai/params/run_Ape-X_CartPole_{}'.format(date, exist_ok=True))
        model_path = '/home/mukai/params/run_Ape-X_CartPole_{}'.format(date)

    ray.init()
    print("START")
    epsilons = np.linspace(0.01, 0.5, num_envs)
    envs = [Environment.remote(pid=i, gamma=args.gamma, epsilon=epsilons[i]) for i in range(num_envs)]
    replay_memory = Experiment_Replay(capacity=args.capacity, td_epsilon=args.td_epsilon)
    learner = Learner.remote(num_actions=ACTION, batch_size=args.batch, gamma=args.gamma)
    current_weights = ray.get(learner.define_network.remote())
    if args.save: 
        torch.save(current_weights, f'{model_path}/model_step_{0:03}.pth')
    current_weights_ray = ray.put(current_weights)
    tester = Tester.remote()
    if args.graph: 
        writer = SummaryWriter(log_dir="./logs/run_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    num_update = 0

    wip_env = [env.rollout.remote(current_weights_ray) for env in envs]
    for _ in tqdm(range(30)):
        finish_env, wip_env = ray.wait(wip_env, num_returns=1)
        td_error, transition, pid = ray.get(finish_env[0])
        replay_memory.push(td_error, transition)
        wip_env.extend([envs[pid].rollout.remote(current_weights_ray)])

    minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(16)]
    wip_learner = learner.update.remote(minibatch)
    minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(16)]
    wip_tester = tester.test_play.remote(current_weights_ray, num_update)
    num_update += 1
    actor_cycle = 0
    sum = 0
    with tqdm(total=args.update) as pbar:
        while num_update < args.update:  
            
            actor_cycle += 1
            finished_env, wip_env = ray.wait(wip_env, num_returns=1)
            td_error, transition, pid = ray.get(finished_env[0])
            replay_memory.push(td_error, transition)
            wip_env.extend([envs[pid].rollout.remote(current_weights_ray)])

            finished_learner, _ = ray.wait([wip_learner], timeout=0)
            if finished_learner:
                current_weights, index, td_error = ray.get(finished_learner[0])
                current_weights_ray = ray.put(current_weights)
                wip_learner = learner.update.remote(minibatch)
                if num_update % 5 == 0: learner.update_target_q_network.remote()
                replay_memory.update_priority(index, td_error)
                minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(16)]

                print(f"Actorが遷移をReplayに渡した回数:{actor_cycle}")

                pbar.update(1)
                num_update += 1
                sum += actor_cycle
                actor_cycle = 0

                #test is faster than interval
                if num_update % args.interval == 0:# and not test:
                    test_score, step = ray.get(wip_tester)
                    if args.graph:
                        writer.add_scalar(f"Ape-X_CartPole.png", test_score, step)
                    if args.save: torch.save(current_weights, f'{model_path}/model_step_{num_update//args.interval:03}.pth')
                    wip_tester = tester.test_play.remote(current_weights_ray, num_update)
    
    ray.get(wip_env)
    test_score, step = ray.get(wip_tester)
    if args.graph:
        writer.add_scalar(f"Ape-X_CartPole.png", test_score, step)
        writer.close()
    ray.shutdown()
    print(f"actor_sum: {sum} ")
    print("END")
    
if __name__ == '__main__':
    main(num_envs=20)