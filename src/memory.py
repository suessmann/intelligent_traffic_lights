import random
import torch
from data_storage import StoreState
from collections import deque, namedtuple


class DQNBuffer:
    def __init__(self, max_n):
        self._buffer = deque(maxlen=max_n)
        self.data = StoreState

    def add(self, sample):
        self._buffer.append(sample)

    def sample(self, n):
        mini_batch = random.sample(self._buffer, n)
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
