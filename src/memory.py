import random
import torch
import numpy as np
from src.data_storage import StoreState
from collections import deque


class DQNBuffer:
    def __init__(self, max_n, pos):
        self._buffer = deque(maxlen=max_n)
        self._buffer_positive = deque(maxlen=max_n)
        self.data = StoreState
        self.pos = pos

    def add(self, sample):
        self._buffer.append(sample)

    def add_positive(self, sample):
        self._buffer_positive.append(sample)

    def sample(self, n):
        mini_batch = random.sample(self._buffer, int(np.ceil(n * (1-self.pos)))) + random.sample(self._buffer_positive, int(np.floor(n * self.pos)))
        trans = StoreState(*zip(*mini_batch))  # unzipping

        state, state_prime = trans.concat()
        a = torch.tensor(trans.action).unsqueeze(1)
        r = torch.tensor(trans.reward).unsqueeze(1)
        done_mask = torch.tensor(trans.done_mask).unsqueeze(1)

        return state, a, r, state_prime, done_mask

    def refill(self):
        self._buffer.clear()

    @property
    def size(self):
        return len(self._buffer)
