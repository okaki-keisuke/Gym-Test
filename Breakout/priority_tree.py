import random
import numpy as np
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

if __name__=="__main__":
    pass