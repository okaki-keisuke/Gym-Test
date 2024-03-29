import random

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