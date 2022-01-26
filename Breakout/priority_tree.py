import numpy as np
from utils import Tree

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
            priority = (np.abs(td_error) + self.td_epsilon) ** self.alpha
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

        return len(self.memory) if self.is_full else self.index

if __name__=="__main__":
    pass