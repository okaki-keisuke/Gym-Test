import numpy as np
import gym
import torch
from model import Net
from Cartpole import ENV
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

model_path = "/home/mukai/params/run_Ape-X_CartPole_2022-01-18_19-8"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



parser = argparse.ArgumentParser(description="Test parameter")
parser.add_argument("--random", action="store_true", help="action randam select")
parser.add_argument("--model" , type=str, default="040", help="model number")
args = parser.parse_args()

Model = Net()
if not args.random:
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
    for _ in tqdm(range(100)):
          for type in ["random", "model"]:
                total_reward = 0
                state = env.reset()
                state = torch.from_numpy(np.atleast_2d(state).astype(np.float32)).clone()
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
                    
                    state = torch.from_numpy(np.atleast_2d(next_state).astype(np.float32)).clone()
               
    #env.close()
    plt.figure(figsize=(10,6))
    plt.boxplot((score_random, score_model))
    plt.xlabel("policy")
    plt.ylabel("score")
    ax = plt.gca()
    plt.setp(ax,xticklabels = ['random','model'])
    plt.savefig("compare.png")