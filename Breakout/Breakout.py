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
from model import Net
from utils import initial_state, input_state
import argparse
from tqdm import tqdm
import os
import copy

ENV = "BreakoutDeterministic-v4"

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True # pytorch speed up

def arg_get() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="myenv setting")

    parser.add_argument("--graph", action="store_true", help="show Graph")
    parser.add_argument("--save", action="store_true",help="save parameter")
    parser.add_argument("--gamma", default=0.99, type=float, help="learning rate")
    parser.add_argument("--batch", default=512, type=int, help="batch size")
    parser.add_argument("--capacity", default=2 ** 18, type=int, help="Replay memory size (2 ** x)")
    parser.add_argument("--epsilon", default=0.5, type=float, help="exploration rate")
    parser.add_argument("--eps_alpha", default=7, type=int, help="epsilon alpha")
    parser.add_argument("--advanced", default=3, type=int, help="number of advanced step")
    parser.add_argument("--td_epsilon", default=0.001, type=float, help="td error epsilon")
    parser.add_argument("--interval", default=10, type=int, help="Test interval")
    parser.add_argument("--update", default=5000, type=int, help="number of update")
    parser.add_argument("--target_update", default=2400, type=int, help="target q network update interval")
    parser.add_argument("--min_replay", default=50000, type=int, help="min experience replay data")
    parser.add_argument("--local_cycle", default=100, type=int, help="number of cycle in Local Environment")
    parser.add_argument("--num_minibatch", default=16, type=int, help="number of minibatch for 1 update")

    # 結果を受ける
    args = parser.parse_args()

    return(args)

class Agent:
    def __init__(self, advanced_step: int):
        self.multi_step = advanced_step
        self.state = []
        self.reward = []
        self.store_state = []
    
    def state_storage(self, state: torch.Tensor) -> None:
        self.store_state.append(state)
        if len(self.store_state) > self.multi_step + 1:
            del self.store_state[0]
        self.state = copy.deepcopy(self.store_state)
        del self.state[1:-1]
        assert len(self.state) <= 2

    def reward_storage(self, reward: torch.Tensor) -> None:
        self.reward.append(reward) 
        if len(self.reward) > self.multi_step:
            del self.reward[0]

    def data_full(self) -> bool:
        if len(self.reward) == self.multi_step and len(self.store_state) == self.multi_step + 1:
            return True
        
        return False
    
    def reset(self):
        self.state = []
        self.reward = []
        self.store_state = []


@ray.remote
class Environment:
    def __init__(self, pid: int, gamma: float, advanced_step: int, epsilon: float, local_cycle: int):
        self.pid = pid
        self.env_name = ENV
        self.env = gym.make(ENV)
        self.action_space = self.env.action_space.n
        self.q_network = Net(action_space=self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.advanced_step = advanced_step
        state = self.env.reset()
        self.agent = Agent(advanced_step=self.advanced_step)
        self.agent.state_storage(initial_state(state))
        self.episode_reward = 0
        self.local_cycle = local_cycle
    
    def rollout(self, weights: parameter) -> list:

        #print("Start:{}-Environment".format(self.pid))

        self.q_network.load_state_dict(weights) 
        buffer = []

        for _ in range(self.local_cycle):

            action = self.q_network.get_action(self.agent.state[-1], epsilon=self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.state_storage(input_state(next_state, self.agent.state[-1]))
            self.episode_reward += reward
            self.agent.reward_storage(reward)
            if self.agent.data_full():
                transition = Transition(self.agent.state[0],
                                        torch.LongTensor([[action]]),
                                        self.agent.state[1], 
                                        torch.FloatTensor([self.agent.reward]),
                                        torch.BoolTensor([done])
                                        )
                buffer.append(transition)
            if done:
               self.agent.reset()
               state = self.env.reset()
               self.agent.state_storage(initial_state(state))
               self.episode_reward = 0
                
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
        
        qvalue = qnet_script(state)
        action_onehot = torch.eye(self.action_space)[action.squeeze()]
        Q = torch.sum(qvalue * action_onehot, dim=1, keepdim=True).squeeze()
        next_qvalue = qnet_script(next_state)
        next_action = torch.argmax(next_qvalue, dim=1)# argmaxQ
        next_action_onehot = torch.eye(self.action_space)[next_action]
        next_maxQ = torch.sum(next_qvalue * next_action_onehot, dim=1, keepdim=True)
        reward_sum = torch.zeros_like(reward[:, 0].unsqueeze(1))
        for r in range(self.advanced_step):
            reward_sum += self.gamma ** r * reward[:, r].unsqueeze(1)
        TQ = (reward_sum + self.gamma ** self.advanced_step * (1 - done.int().unsqueeze(1)) * next_maxQ).squeeze()
        td_error = torch.square(Q - TQ)
        td_errors = td_error.cpu().detach().numpy().flatten()

        return td_errors, transition
    
    def get_action_space(self) -> int:
        
        return self.action_space

class Experiment_Replay:

    def __init__(self, capacity: int, td_epsilon: float):
        self.capacity = capacity
        self.priority = Tree(self.capacity)
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
            priority = (abs(td_error) + 0.001) ** self.alpha
            self.priority[idx] = priority

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

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, action_space: int, batch_size: int, gamma: float, advanced_step: int, target_update: int):
        self.action_space = action_space
        self.main_q_network = Net(self.action_space).to(device)
        self.target_q_network = Net(self.action_space).to(device)
        #self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.001)
        self.optimizer = optim.RMSprop(self.main_q_network.parameters(), lr=0.0025/4, alpha=0.9, momentum=0.0, eps=1.5e-07, centered=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 1000, gamma=0.8)
        self.batch_size = batch_size
        self.gamma = gamma
        self.advanced_step = advanced_step
        self.target_update = target_update
        self.update_count = 0
    
    def define_network(self) -> parameter:
        current_weights = self.main_q_network.to('cpu').state_dict()
        self.main_q_network.to(device)
        self.update_target_q_network()  
        return current_weights

    def update(self, minibatch: Tuple) -> list:
        
        index_all = []
        td_error_all = []
        loss_list = []

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
#            with torch.no_grad():
            next_qvalue = self.main_q_network(next_state_batch)
            next_action = torch.argmax(next_qvalue, dim=1)# argmaxQ
            next_action_onehot = torch.eye(self.action_space)[next_action].to(device)
            target_qvalue = self.target_q_network(next_state_batch)
            next_maxQ = torch.sum(target_qvalue * next_action_onehot, dim=1, keepdim=True)
            reward_sum = torch.zeros_like(reward_batch[:, 0].unsqueeze(1))
            for r in range(self.advanced_step):
                reward_sum += self.gamma ** r * reward_batch[:, r].unsqueeze(1)
            TQ = (reward_sum + self.gamma ** self.advanced_step * (1 - done_batch.int().unsqueeze(1)) * next_maxQ).squeeze()
            #Q(s,a)-value
            qvalue = self.main_q_network(state_batch)
            action_onehot = torch.eye(self.action_space)[action_batch.squeeze()].to(device)
            Q = torch.sum(qvalue * action_onehot, dim=1, keepdim=True).squeeze()
            td_error = torch.square(Q - TQ)
            td_errors = td_error.cpu().detach().numpy().flatten()

            self.main_q_network.train()
            loss = torch.mean(weights * td_error)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            utils.clip_grad_norm_(self.main_q_network.parameters(), max_norm=40.0, norm_type=2.0)
            self.optimizer.step()

            index_all += index
            td_error_all += td_errors.tolist()
            self.update_count += 1

            if self.update_count % self.target_update == 0:
                self.update_target_q_network()
        
        #self.scheduler.step()
        current_weight = self.main_q_network.to('cpu').state_dict()
        self.main_q_network.to(device)

        return current_weight, index_all, td_error_all

    def update_target_q_network(self) -> None:
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

@ray.remote
class Tester:

    def __init__(self, action_space: int):
        self.q_network = Net(action_space)
        self.epsilon = 0.01
    
    def test_play(self, current_weights: parameter, step: int) -> list:
        self.q_network.load_state_dict(current_weights)
        env = gym.make(ENV)
        state = env.reset()
        state = initial_state(state)
        total_reward = 0
        done = False
        while not done:
            action = self.q_network.get_action(state=state, epsilon=self.epsilon)
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = input_state(new_state, state)
        
        return total_reward, step
    
def main(num_envs: int) -> None:

    args = arg_get()
    if args.save:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%-M")
        os.makedirs('/home/mukai/params/run_Ape-X_Breakout_{}'.format(date, exist_ok=True))
        model_path = '/home/mukai/params/run_Ape-X_Breakout_{}'.format(date)

    ray.init(num_gpus=1)
    print("START")
    #epsilons = np.linspace(0.01, args.epsilon, num_envs)
    epsilons = [args.epsilon ** (1 + args.eps_alpha * i / (num_envs - 1)) for i in range(num_envs)]
    epsilons = [max(0.01, eps) for eps in epsilons]
    envs = [Environment.remote(pid=i, gamma=args.gamma, advanced_step=args.advanced, epsilon=epsilons[i], local_cycle=args.local_cycle) for i in range(num_envs)]
    replay_memory = Experiment_Replay(capacity=args.capacity, td_epsilon=args.td_epsilon)
    action_space = envs[0].get_action_space.remote()
    learner = Learner.remote(action_space=action_space, batch_size=args.batch, gamma=args.gamma, advanced_step=args.advanced, target_update=args.target_update)
    current_weights = ray.get(learner.define_network.remote())
    if args.save: torch.save(current_weights, f'{model_path}/model_step_{0:03}.pth')
    current_weights_ray = ray.put(current_weights)
    tester = Tester.remote(action_space=action_space)
    if args.graph: writer = SummaryWriter(log_dir="./logs2/run_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    num_update = 0

    wip_env = [env.rollout.remote(current_weights_ray) for env in envs]
    for _ in tqdm(range(args.min_replay // args.local_cycle)):
        finish_env, wip_env = ray.wait(wip_env, num_returns=1)
        td_error, transition, pid = ray.get(finish_env[0])
        replay_memory.push(td_error, transition)
        wip_env.extend([envs[pid].rollout.remote(current_weights_ray)])

    minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(args.num_minibatch)]
    wip_learner = learner.update.remote(minibatch)
    minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(args.num_minibatch)]
    wip_tester = tester.test_play.remote(current_weights_ray, num_update)
    actor_cycle = 0
    sum = actor_cycle
    print('-'*80)
    print(f"Test Start : {num_update//args.interval} | Number of Updates : {num_update} | Number of Push : {sum}")
    print('-'*80)
    
    with tqdm(total=args.update) as pbar:
        while num_update < args.update:  

            actor_cycle += 1
            finished_env, wip_env = ray.wait(wip_env, num_returns=1)
            td_error, transition, pid = ray.get(finished_env[0])
            replay_memory.push(td_error, transition)
            wip_env.extend([envs[pid].rollout.remote(current_weights_ray)])

            finished_learner, _ = ray.wait([wip_learner], timeout=0)
            if finished_learner and actor_cycle >= 100:
                current_weights, index, td_error = ray.get(finished_learner[0])
                current_weights_ray = ray.put(current_weights)
                wip_learner = learner.update.remote(minibatch)
                replay_memory.update_priority(index, td_error)
                minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(16)]
                
                #rint(f"Actorが遷移をReplayに渡した回数:{actor_cycle}")

                pbar.update(1)
                num_update += 1
                sum += actor_cycle
                actor_cycle = 0

               #test is faster than interval
                if num_update % args.interval == 0:
                    test_score, step = ray.get(wip_tester)
                    if args.graph:
                        writer.add_scalar(f"Ape-X_Breakout.png", test_score, step)
                    if args.save: torch.save(current_weights, f'{model_path}/model_step_{num_update//args.interval:03}.pth')
                    print('\n' + '-' * 80)
                    print(f"Test End : {num_update//args.interval - 1} | Number of Updates : {num_update} | Test Score : {test_score}")
                    #Test End↑　Test Start↓
                    wip_tester = tester.test_play.remote(current_weights_ray, num_update)
                    print(f"Test Start : {num_update//args.interval} | Number of Updates : {num_update} | Number of Push : {sum}")
                    print('-'*80)
    
    ray.get(wip_env)
    test_score, step = ray.get(wip_tester)
    print('\n' + '-' * 80)
    print(f"Test End : {num_update//args.interval} | Number of Updates : {num_update} | Test Score : {test_score}")
    print('-' * 80)
    if args.graph:
        writer.add_scalar(f"Ape-X_Breakout.png", test_score, step)
        writer.close()
    ray.shutdown()
    print("END")
    
if __name__ == '__main__':
    main(num_envs=20)