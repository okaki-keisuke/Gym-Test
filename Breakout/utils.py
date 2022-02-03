import torch
import numpy as np
from model import INPUT_HEIGHT, INPUT_WIDTH
from PIL import Image
import random
import time

class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        fin = time.time() - self.start
        print(self.name, fin)


class Tree:
    #初期設定
    def __init__(self, capacity : int):
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.values = [0 for _ in range(2 * capacity)]
    
    #print 文字列にしてくれる
    def __str__(self):
        return str(self.values[self.capacity:])
    
    #pri_tree[idx] = val のかたちで代入できる
    #木の左から順に代入する
    def __setitem__(self, idx, val):
        idx = idx + self.capacity
        self.values[idx] = val 

        current_idx = idx // 2
        while current_idx >= 1:
            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1
            self.values[current_idx] = self.values[idx_lchild] + self.values[idx_rchild]
            current_idx //= 2

    def __getitem__(self, idx):
        idx = idx + self.capacity
        return self.values[idx]
    
    def sum(self):
        return self.values[1]
    
    def sample(self):
        z = random.uniform(0, self.sum())
        assert 0 <= z <= self.sum(), z 

        current_idx = 1
        while current_idx < self.capacity:
            idx_lchild = 2 * current_idx
            idx_rchild = 2 * current_idx + 1

            if z > self.values[idx_lchild]:
                current_idx = idx_rchild
                z = z - self.values[idx_lchild]
            else:
                current_idx = idx_lchild
        
        idx = current_idx - self.capacity
        return idx

def preproccess(state: np) ->  np:
    
    state_pil = Image.fromarray(state)
    state_pil = state_pil.convert("L").crop((0, 34, 160, 200)).resize((INPUT_WIDTH, INPUT_HEIGHT))
    state_np = np.array(state_pil) / 255.0

    return state_np.astype(np.float32)

def huber_loss(target_q, q, d=1.0):
    """
    See https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/losses.py#L1098-L1162
    """
    td_error = target_q - q
    is_smaller_than_d = torch.abs(td_error) < d
    squared_loss = 0.5 * torch.square(td_error)
    linear_loss = 0.5 * d ** 2 + d * (torch.abs(td_error) - d)
    loss = torch.where(is_smaller_than_d, squared_loss, linear_loss)
    return loss
        
if __name__ == "__main__":
    sumtree = Tree(capacity=4)
    sumtree[0] = 4
    sumtree[1] = 1
    sumtree[2] = 2
    sumtree[3] = 3
    samples = [sumtree.sample() for _ in range(1000)]

    print(samples.count(0))
    print(samples.count(1))
    print(samples.count(2))
    print(samples.count(3))