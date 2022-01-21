import torch
import torch
import torchvision      
from torchvision import transforms
import numpy as np
from model import FRAME_HEIGHT, FRAME_WIDTH, INPUT_HEIGHT, INPUT_WIDTH
from PIL import Image

def get_initial_state(observation: np) -> torch.Tensor:
        observation = torchvision.transforms.functional.to_tensor(observation)
        observation = observation.reshape(1, 3, FRAME_HEIGHT, FRAME_WIDTH)
        observation_crop = torchvision.transforms.functional.crop(observation, 34, 0, 160, 160)
        transform = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH), transforms.InterpolationMode.BICUBIC)
                                        ])
        observation_trans = transform(observation_crop)
        observation_scale = torch.div(observation_trans, 255)
        state = torch.cat((observation_scale, observation_scale, observation_scale, observation_scale), 1)
        return state

def input_image(observation: np, state: torch.Tensor) -> torch.Tensor:
        observation = torchvision.transforms.functional.to_tensor(observation)
        observation = observation.reshape(1, 3, FRAME_HEIGHT, FRAME_WIDTH)
        observation_crop = torchvision.transforms.functional.crop(observation, 34, 0, 160, 160)
        transform = transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH), transforms.InterpolationMode.BICUBIC)
                                        ])
        observation_trans = transform(observation_crop)
        observation_scale = torch.div(observation_trans, 255)
        state = torch.cat((observation_scale, state[:, :3, :, :]), 1)
        return state

def initial_state(state: np) -> torch.Tensor:
        state_pil = Image.fromarray(state)
        state_pil = state_pil.crop((34, 0, 160, 160)).convert("L").resize((INPUT_WIDTH, INPUT_HEIGHT))
        state_torch = torchvision.transforms.functional.to_tensor(state_pil)
        state_torch = state_torch.reshape(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
        state_torch = torch.div(state_torch, 255)
        state = torch.cat((state_torch, state_torch, state_torch, state_torch), 1)
        return state

def input_state(state: np, pre_state: torch.Tensor) -> torch.Tensor:
        state_pil = Image.fromarray(state)
        state_pil = state_pil.crop((34, 0, 160, 160)).convert("L").resize((INPUT_WIDTH, INPUT_HEIGHT))
        state_torch = torchvision.transforms.functional.to_tensor(state_pil)
        state_torch = state_torch.reshape(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
        state_torch = torch.div(state_torch, 255)
        state = torch.cat((state_torch, pre_state[:, :3, :, :]), 1)

        return state