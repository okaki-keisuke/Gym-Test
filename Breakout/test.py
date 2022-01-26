import numpy as np
import gym
import torch
from model import Net
from Actor import ENV
from utils import preproccess
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import collections


model_path = "/home/mukai/params/run_Ape-X_Breakout_2022-01-26_12-34"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description="Test parameter")
parser.add_argument("--random", action="store_true", help="action randam select")
parser.add_argument("--model" , type=str, default="360", help="model number")
args = parser.parse_args()

env = gym.make(ENV)
Model = Net(env.action_space.n).to(device)
if not args.random:
    Model.load_state_dict(torch.load(f"{model_path}/model_step_{args.model}.pth"))


def decide_action(state: torch.Tensor) -> int:
    
    Model.eval()
    state.to(device)
    if args.random:
        output = env.action_space.sample()
    else:
        output = Model(state).max(1)[1].view(1, 1).to("cpu").item()
    
    return output

if __name__=="__main__":

    plt.style.use("ggplot")
    sns.set_palette('Set2')
    warnings.filterwarnings('ignore')
    score = []  
    frames = collections.deque(maxlen=4)
    for episode in tqdm(range(10)):
        total_reward = 0
        frame = preproccess(env.reset())
        for _ in range(4):
            frames.append(frame)
        state = torch.FloatTensor(np.stack(frames, axis=0)[np.newaxis, ...]).cuda()
        done = False
        step = 0
        while not done:
            env.render()
            action = decide_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            #print(step, done)
            if done or step > 1000:
                score.append(total_reward)
                break
            else:
                frames.append(preproccess(next_state))
                state = torch.FloatTensor(np.stack(frames, axis=0)[np.newaxis, ...]).cuda()
                step += 1
    env.close()
    plt.figure(figsize=(10,6))
    plt.boxplot(score)
    plt.savefig("test.png")