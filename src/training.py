import sys, os
sys.path.append(os.path.join('/home/thelak-dev/sumo/tools'))

from datetime import datetime
from dqn import DQNetwork, FCQNetwork
from env import SumoIntersection
from memory import DQNBuffer
from data_storage import StoreState
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from tqdm import tqdm

LEARNING_RATE = 0.00001
GAMMA = 0.95
BUFFER_LIMIT = 500000
BATCH_SIZE = 128
SIM_LEN = 4500
MEM_REFILL = 200
C = 100
EPOCHS = 1600
PRINT = 10
N_CARS= 1000
WEIGHTS_PATH = ''

sumoBinary = "/home/thelak-dev/sumo/bin/sumo"
sumoCmd = "/home/thelak-dev/tldqn/exp2/TLC-DQN/src/cfg/sumo_config.sumocfg"

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(),
                                                       local_model.parameters()):
                    target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

def train_net(q, q_target, memory, optimizer):
    state, a, r, state_prime, done_mask = memory.sample(BATCH_SIZE)

    q_out = q(state)
    q_a = q_out.gather(1, a)
    max_q_prime = q_target(state_prime).max(1)[0].unsqueeze(1)
    target = r + GAMMA * max_q_prime * done_mask
    criterion = nn.MSELoss()
    loss = criterion(q_a, target)
    mse = loss.item()

    optimizer.zero_grad()
    loss.backward()
    for param in q.parameters():
       param.grad.data.clamp_(-1, 1)
    optimizer.step()

if __name__ == '__main__':
    q = DQNetwork()

    try:
        q.load_state_dict(torch.load(WEIGHTS_PATH))
    except FileNotFoundError:
        q.apply(weights_init)
        print('No model weights found, initializing xavier_uniform')

    q_target = DQNetwork()
    q_target.load_state_dict(q.state_dict())

    memory = DQNBuffer(BUFFER_LIMIT, 0.)

    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN, N_CARS)
    optimizer = torch.optim.RMSprop(q.parameters(), lr=LEARNING_RATE)

    state = StoreState()
    min_wait = float('inf')
    mse_mean = 0
    total_steps=0
    pbar = tqdm(total=EPOCHS)

    for epoch in np.arange(EPOCHS):
        eps = max(0.01, 0.07 - 0.01*(epoch/200))
        step = 0

        if epoch != 0:
            env.reset()

        state, _, _, _ = env.step(0)
        done = False

        done_mask = 1.0
        pbar.set_description(f'EPOCH: {epoch}, mean reward: {info[6]}, mean waiting time: {info[2]}')
        mse = 0
        while not done:
            with torch.no_grad():
                a = q.predict(state.as_tuple, eps)
            state, r, done, info = env.step(a) # storing prime state

            if done:
                done_mask = 0.0

            if r != 0 or step > 60:
                memory.add((state.position, state.speed,
                    state.tl, state.p_position,
                    state.p_speed, state.p_tl,
                    a, r, done_mask))

            state.swap()  # state = state_prime

            if memory.size > BATCH_SIZE:
                train_net(q, q_target, memory, optimizer)
           
            step += 1
            total_steps += 1
        
        #  clear memory each MEM_REFILL epoch
        if epoch%MEM_REFILL == 0 and epoch != 0:
            memory.refill()
        pbar.update(1)
        
        # soft update weights at the end of epoch
        soft_update(q, q_target, 0.01) )

        if info[4]/info[5] < min_wait and epoch > 50:
            torch.save(q.state_dict(), f'../model/dqn_{epoch}.pt')
            min_wait = info[4]/info[5]
        if epoch == 1599:
            torch.save(q.state_dict(), f'../model/dqn_{epoch}_final.pt')

    print('finished training')

