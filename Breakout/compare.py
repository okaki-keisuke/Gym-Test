import numpy as np
import gym
import torch
from model import Net
from Breakout import ENV
from utils import preproccess
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torchvision      
from tqdm import tqdm
import collections

model_path = "/home/mukai/params/run_Ape-X_Breakout_2022-01-08_17-53"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(description="Test parameter")
parser.add_argument("--random", action="store_true", help="action randam select")
parser.add_argument("--model" , type=str, default="000", help="model number")
args = parser.parse_args()

if not args.random:
    Model = Net().to(device)
    Model.load_state_dict(torch.load(f"{model_path}/model_step_{args.model}.pth"))

env = gym.make(ENV)

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
    score_model = []
    score_random = []
    for episode in tqdm(range(100)):
        for type in ["random", "model"]:
            total_reward = 0
            frame = collections.deque(maxlen=4)
            state = preproccess(env.reset())
            for _ in range(4):
                frame.append(state)
            state = torch.FloatTensor(np.stack(frame, axis=0)[np.newaxis, ...])
            while True:
                #env.render()
                action = decide_action(state) if type == "model" else env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                if done and type == "random":
                    score_random.append(total_reward)
                    break
                elif done and type == "model":
                    score_model.append(total_reward)
                    break
                frame.append(preproccess(next_state))
                state = torch.FloatTensor(np.stack(frame, axis=0)[np.newaxis, ...])
                
    #env.close()
    plt.figure(figsize=(10,6))
    plt.boxplot((score_random, score_model))
    plt.xlabel("policy")
    plt.ylabel("score")
    ax = plt.gca()
    plt.setp(ax,xticklabels = ['random','model'])
    plt.savefig("compare.png")