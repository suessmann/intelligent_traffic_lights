from dqn import DQNetwork
from env import SumoIntersection
from memory import DQNBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

LEARNING_RATE = 0.0005
GAMMA = 0.98
BUFFER_LIMIT = 50000
BATCH_SIZE = 32
SIM_LEN = 4500
MEM_REFILL = 250
C = 50
random.seed(42)
EPOCHS = 100

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime = memory.sample(BATCH_SIZE)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + GAMMA * max_q_prime
        criterion = nn.MSELoss()
        loss = criterion(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def unsqueeze(state):
    state = (torch.from_numpy(np.array(state[0], dtype=np.float32)).unsqueeze(0).reshape(1, 4, 4, 15),
             torch.from_numpy(np.array(state[1], dtype=np.float32)).unsqueeze(0).reshape(1, 4, 4, 15),
             torch.from_numpy(np.array(state[2], dtype=np.float32)).unsqueeze(0)
             )
    return state


if __name__ == '__main__':
    q = DQNetwork()
    q_target = DQNetwork()
    q_target.load_state_dict(q.state_dict())

    memory = DQNBuffer(BUFFER_LIMIT)
    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN)

    optimizer = torch.optim.RMSprop(q.parameters(), lr=LEARNING_RATE)

    # number of different simulations to perform


    for n_epi in np.arange(EPOCHS):
        eps = max(0.01, 0.08 - 0.01*(n_epi/200))

        if n_epi != 0:
            env.reset()

        s, _, _, _ = env.step(0)
        done = False
        while not done:
            a = q.predict(unsqueeze(s), eps)
            s_prime, r, done, info = env.step(a)

            #  TODO: сделать вот тут по-человечески
            info = (r[0], r[1])
            r = r[1]
            memory.add((s, a, r, s_prime))
            s = s_prime

            if memory.size > 8000:
                train(q, q_target, memory, optimizer)

            if env.time % C == 0:
                q_target.load_state_dict(q.state_dict())
                print(f"EPOCH: {n_epi}, step: {env.time}, queue: {info[0]}, total time waiting: {info[1]}")



        #  clear memory each MEM_REFILL epoch
        if n_epi%MEM_REFILL == 0 and n_epi != 0:
            memory.refill()



