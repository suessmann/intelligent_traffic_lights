import random
import torch
from collections import deque


class DQNBuffer:
    def __init__(self, max_n):
        self._buffer = deque(maxlen=max_n)

    def add(self, sample):
        self._buffer.append(sample)

    def sample(self, n):
        # return s_prime, r, done, info
        mini_batch = random.sample(self._buffer, n)
        p_m_lst, v_m_lst, tl_m_lst, p_m_lst_prime, v_m_lst_prime, tl_m_lst_prime,\
        a_lst, r_lst, s_prime_lst = [], [], [], [], [], [], [], [], []

        for batch in mini_batch:
            s, a, r, s_prime = batch

            p_m, v_m, tl_m = s
            p_m_prime, v_m_prime, tl_m_prime = s_prime

            p_m_lst.append(p_m)
            v_m_lst.append(v_m)
            tl_m_lst.append(tl_m)

            p_m_lst_prime.append(p_m_prime)
            v_m_lst_prime.append(v_m_prime)
            tl_m_lst_prime.append(tl_m_prime)

            a_lst.append([a])
            r_lst.append([r])

        p_m_lst = torch.tensor(p_m_lst, dtype=torch.float32).reshape(-1, 4, 4, 15)
        v_m_lst = torch.tensor(v_m_lst, dtype=torch.float32).reshape(-1, 4, 4, 15)
        tl_m_lst = torch.tensor(tl_m_lst, dtype=torch.float32)
        p_m_lst_prime = torch.tensor(p_m_lst_prime, dtype=torch.float32).reshape(-1, 4, 4, 15)
        v_m_lst_prime = torch.tensor(v_m_lst_prime, dtype=torch.float32).reshape(-1, 4, 4, 15)
        tl_m_lst_prime = torch.tensor(tl_m_lst_prime, dtype=torch.float32)

            # done_mask_lst.append([done_mask])
        return (p_m_lst, v_m_lst, tl_m_lst), \
               torch.tensor(a_lst), \
               torch.tensor(r_lst), \
               (p_m_lst_prime, v_m_lst_prime, tl_m_lst_prime)
                # torch.tensor(done_mask_lst)

    def refill(self):
        self._buffer.clear()


    @property
    def size(self):
        return len(self._buffer)
