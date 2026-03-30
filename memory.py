import collections
import random
import pickle

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = collections.deque(maxlen=max_size)

    def save(self, memory):
        self.buffer.extend(memory)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def save_to_disk(self, filename="buffer.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)