import collections
import random
import pickle


# In memory.py
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []

    def add(self, experience):
        """Adds a single (state, policy, value) triplet."""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove oldest
        self.buffer.append(experience)

    def sample(self, batch_size):
        import random

        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)
