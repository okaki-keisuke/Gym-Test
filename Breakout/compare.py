import numpy as np
import gym
import torch
from model import Net
from .Breakout import ACTION, ENV
from utils import get_initial_state, input_image
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torchvision      
from tqdm import tqdm

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


class Agent:
    def __init__(self, observation: np) -> None:
        observation = torchvision.transforms.functional.to_tensor(observation).to(device)
        self.state = get_initial_state(observation)
    
    def update_state(self, observation: np) -> None:
        observation = torchvision.transforms.functional.to_tensor(observation).to(device)
        self.state = input_image(observation=observation, state=self.state)

if __name__=="__main__":

    
    plt.style.use("ggplot")
    sns.set_palette('Set2')
    warnings.filterwarnings('ignore')
    score_model = []
    score_random = []
    for episode in tqdm(range(100)):
          for type in ["random", "model"]:
               total_reward = 0
               observation = env.reset()
               agent = Agent(observation)
               while True:
                    #env.render()
                    action = decide_action(agent.state) if type == "model" else env.action_space.sample()
                    observation, reward, done, info = env.step(action)
                    total_reward += reward
                    if done and type == "random":
                         score_random.append(total_reward)
                         break
                    elif done and type == "model":
                         score_model.append(total_reward)
                         break
                    
                    agent.update_state(observation)
               
    #env.close()
    plt.figure(figsize=(10,6))
    plt.boxplot((score_random, score_model))
    plt.xlabel("policy")
    plt.ylabel("score")
    ax = plt.gca()
    plt.setp(ax,xticklabels = ['random','model'])
    plt.savefig("compare.png")