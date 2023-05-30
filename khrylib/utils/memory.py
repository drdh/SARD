import random
import numpy as np

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a tuple."""
        self.memory.append([*args])

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch

    def sample_stride(self, batch_size, stride=0):
        idx = np.random.randint(0, len(self.memory)-stride, batch_size)
        random_batch = []
        for s in range(stride+1):
            random_batch.append([self.memory[i] for i in idx+s])
        return random_batch

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

