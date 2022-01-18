import torch
import torch
import torchvision      
from torchvision import transforms
import numpy as np
from model import FRAME_HEIGHT, FRAME_WIDTH, INPUT_HEIGHT, INPUT_WIDTH

def get_initial_state(observation: np) -> torch.Tensor:
        observation = torchvision.transforms.functional.to_tensor(observation)
        observation = observation.reshape(1, 3, FRAME_HEIGHT, FRAME_WIDTH)
        transform = transforms.Compose([transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH), transforms.InterpolationMode.BICUBIC),
                                        torchvision.transforms.Grayscale(num_output_channels=1)
                                        ])
        observation = transform(observation)
        state = torch.cat((observation, observation, observation, observation), 1)
        return state

def input_image(observation: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        observation = torchvision.transforms.functional.to_tensor(observation)
        observation = observation.reshape(1, 3, FRAME_HEIGHT, FRAME_WIDTH)
        transform = transforms.Compose([transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH), transforms.InterpolationMode.BICUBIC),
                                        torchvision.transforms.Grayscale(num_output_channels=1)
                                        ])
        observation = transform(observation)
        state = torch.cat((observation, state[:, :3, :, :]), 1)
        return state