import os
import datetime
import numpy as np
import ray
import argparse
from tqdm import tqdm
from typing import Tuple

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import parameter, utils

from ReplayMemory import Experiment_Replay
from Actor import Environment, Transition, Tester
from model import Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    parser.add_argument("--n_frame", default=4, type=int, help="state frame")

    # 結果を受ける
    args = parser.parse_args()

    return(args)

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:

    def __init__(self, action_space: int, gamma: float, advanced_step: int, target_update: int):
    
        self.action_space = action_space
        self.gamma = gamma
        self.advanced_step = advanced_step
        self.main_q_network = Net(self.action_space).cuda()
        self.target_q_network = Net(self.action_space).cuda()
        self.target_update = target_update
        
        #self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.001)
        self.optimizer = optim.RMSprop(self.main_q_network.parameters(), lr=0.0025/4, alpha=0.95, momentum=0.0, eps=1.5e-07, centered=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, 1000, gamma=0.8)
        
        self.update_count = 0
    
    def define_network(self) -> parameter:
        current_weights = self.main_q_network.cpu().state_dict()
        self.main_q_network.cuda()
        self.update_target_q_network()  
        return current_weights

    def update(self, minibatch: Tuple) -> list:
        
        index_all = []
        td_error_all = []
        loss_list = []

        for (index, weight, transition) in minibatch:

            batch = Transition(*zip(*transition))
            state_batch = torch.cat(batch.state).cuda()
            action_batch = torch.cat(batch.action).cuda()
            reward_batch = torch.cat(batch.reward).cuda()
            next_state_batch = torch.cat(batch.next_state).cuda()
            done_batch = torch.cat(batch.done).cuda()
            weights = torch.from_numpy(np.atleast_2d(weight).astype(np.float32)).clone().cuda()

            self.main_q_network.eval()
            self.target_q_network.eval()

            #TDerror
            next_qvalue = self.main_q_network(next_state_batch)
            next_action = torch.argmax(next_qvalue, dim=1)# argmaxQ
            next_action_onehot = torch.eye(self.action_space)[next_action].cuda()
            target_qvalue = self.target_q_network(next_state_batch)
            next_maxQ = torch.sum(target_qvalue * next_action_onehot, dim=1, keepdim=True)
            TQ = (reward_batch + self.gamma ** self.advanced_step * (1 - done_batch.int().unsqueeze(1)) * next_maxQ).squeeze()
            #Q(s,a)-value
            qvalue = self.main_q_network(state_batch)
            action_onehot = torch.eye(self.action_space)[action_batch].cuda()
            Q = torch.sum(qvalue * action_onehot, dim=1, keepdim=True).squeeze()
            td_error = torch.square(Q - TQ)

            self.main_q_network.train()
            loss = torch.mean(weights * td_error)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            utils.clip_grad_norm_(self.main_q_network.parameters(), max_norm=40.0, norm_type=2.0)
            self.optimizer.step()

            index_all += index
            td_error_all += td_error.cpu().detach().numpy().flatten().tolist()
            loss_list.append(loss.cpu().detach().numpy())
            self.update_count += 1

            if self.update_count % self.target_update == 0:
                self.update_target_q_network()
        
        #self.scheduler.step()
        current_weight = self.main_q_network.cpu().state_dict()
        self.main_q_network.cuda()
        loss_mean = np.array(loss_list).mean()

        return current_weight, index_all, td_error_all, loss_mean

    def update_target_q_network(self) -> None:
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    
def main(num_envs: int) -> None:

    args = arg_get()
    if args.save:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%-M")
        os.makedirs('/home/mukai/params/run_Ape-X_Breakout_{}'.format(date, exist_ok=True))
        model_path = '/home/mukai/params/run_Ape-X_Breakout_{}'.format(date)

    ray.init(local_mode=False)
    print("START")
    #epsilons = np.linspace(0.01, args.epsilon, num_envs)
    epsilons = [args.epsilon ** (1 + args.eps_alpha * i / (num_envs - 1)) for i in range(num_envs)]
    epsilons = [max(0.01, eps) for eps in epsilons]
    envs = [Environment.remote(pid=i, gamma=args.gamma, advanced_step=args.advanced, epsilon=epsilons[i], local_cycle=args.local_cycle, n_frame=args.n_frame) for i in range(num_envs)]
    action_space = envs[0].get_action_space.remote()

    replay_memory = Experiment_Replay(capacity=args.capacity, td_epsilon=args.td_epsilon)

    learner = Learner.remote(action_space=action_space, gamma=args.gamma, advanced_step=args.advanced, target_update=args.target_update)
    
    current_weights = ray.get(learner.define_network.remote())
    if args.save:
        torch.save(current_weights, f'{model_path}/model_step_{0:03}.pth')
    current_weights = ray.put(current_weights)
    tester = Tester.remote(action_space=action_space, n_frame=args.n_frame)
    num_update = 0
    wip_env = [env.rollout.remote(current_weights) for env in envs]

    for _ in tqdm(range(args.min_replay // args.local_cycle)):
        finish_env, wip_env = ray.wait(wip_env, num_returns=1)
        td_error, transition, pid = ray.get(finish_env[0])
        replay_memory.push(td_error, transition)
        wip_env.extend([envs[pid].rollout.remote(current_weights)])
    
    if args.graph:
        writer = SummaryWriter(log_dir="./logs2/run_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
        writer.add_scalar(f"Replay_Memory", len(replay_memory), num_update)   
    
    minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(args.num_minibatch)]
    wip_learner = learner.update.remote(minibatch)
    minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(args.num_minibatch)]
    wip_tester = tester.test_play.remote(current_weights, num_update)

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
            wip_env.extend([envs[pid].rollout.remote(current_weights)])

            finished_learner, _ = ray.wait([wip_learner], timeout=0)

            if finished_learner and actor_cycle >= 150:
                current_weights, index, td_error, loss_mean = ray.get(finished_learner[0])
                if args.save and num_update % 100 == 0: 
                    torch.save(current_weights, f'{model_path}/model_step_{num_update//args.interval:03}.pth')
                if args.graph:
                    writer.add_scalar(f"Learner_Loss", loss_mean, num_update)
                current_weights = ray.put(current_weights)
                wip_learner = learner.update.remote(minibatch)
                replay_memory.update_priority(index, td_error)
                minibatch = [replay_memory.sample(batch_size=args.batch) for _ in range(args.num_minibatch)]
                
                #print(f"Actorが遷移をReplayに渡した回数:{actor_cycle}")

                pbar.update(1)
                num_update += 1
                sum += actor_cycle
                actor_cycle = 0

               #test is faster than interval
                if num_update % args.interval == 0:
                    test_score, episode, step = ray.get(wip_tester)
                    if args.graph:
                        writer.add_scalar(f"Ape-X_Breakout", test_score, step)
                        writer.add_scalar(f"Test_Episode", episode, step)
                        writer.add_scalar(f"Replay_Memory", len(replay_memory), num_update)
                    print('\n' + '-' * 80)
                    print(f"Test End   : {num_update//args.interval - 1} | Test Episode : {episode} | Test Score : {test_score}")
                    #Test End↑　Test Start↓
                    wip_tester = tester.test_play.remote(current_weights, num_update)
                    print(f"Test Start : {num_update//args.interval} | Number of Updates : {num_update} | Number of Push : {sum}")
                    print('-'*80)
    
    ray.get(wip_env)
    test_score,episode, step = ray.get(wip_tester)
    print('\n' + '-' * 80)
    print(f"Test End   : {num_update//args.interval} | Test Episode : {episode} | Test Score : {test_score}")
    print('-' * 80)
    if args.graph:
        writer.add_scalar(f"Ape-X_Breakout", test_score, step)
        writer.add_scalar(f"Test_Episode", episode, step)
        writer.close()
    ray.shutdown()
    print("END")
    
if __name__ == '__main__':
    main(num_envs=20)