import sys, os

from src.dqn import DQNetwork, FCQNetwork
from src.env import SumoIntersection
from src.memory import DQNBuffer
from src.data_storage import StoreState

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

def train_net(q, q_target, memory, optimizer, batch_size, gamma):
    state, a, r, state_prime, done_mask = memory.sample(batch_size)

    q_out = q(state)
    q_a = q_out.gather(1, a)
    max_q_prime = q_target(state_prime).max(1)[0].unsqueeze(1)
    target = r + gamma * max_q_prime * done_mask
    criterion = nn.MSELoss()
    loss = criterion(q_a, target)
    mse = loss.item()

    optimizer.zero_grad()
    loss.backward()
    for param in q.parameters():
       param.grad.data.clamp_(-1, 1)
    optimizer.step()

def training(config):
    learning_rate = config['learning_rate']
    gamma = config['gamma']
    buffer_limit = config['buffer_limit']
    batch_size = config['batch_size']
    sim_len = config['sim_len']
    mem_refill = config['mem_refill']
    epochs = config['epochs']
    n_cars = config['n_cars']
    weights_path = config['weights_path']

    sumoBinary = config['sumoBinary']
    sumoCmd = config['sumoCmd']
    sys.path.append(os.path.join(config['sumoTools']))

    q = DQNetwork()

    try:
        q.load_state_dict(torch.load(weights_path))
    except FileNotFoundError:
        q.apply(weights_init)
        print('No model weights found, initializing xavier_uniform')

    q_target = DQNetwork()
    q_target.load_state_dict(q.state_dict())

    memory = DQNBuffer(buffer_limit, 0.)

    env = SumoIntersection(sumoBinary, sumoCmd, sim_len, n_cars)
    optimizer = torch.optim.RMSprop(q.parameters(), lr=learning_rate)

    state = StoreState()
    min_wait = float('inf')
    total_steps=0
    pbar = tqdm(total=epochs)

    for epoch in np.arange(epochs):
        eps = max(0.01, 0.07 - 0.01*(epoch/200))
        step = 0

        if epoch != 0:
            env.reset()
            pbar.set_description(f'EPOCH: {epoch}, mean reward: {info[6]}, mean waiting time: {info[2]}')

        state, _, _, _ = env.step(0)
        done = False

        done_mask = 1.0
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

            if memory.size > batch_size:
                train_net(q, q_target, memory, optimizer, batch_size, gamma)
           
            step += 1
            total_steps += 1
        
        #  clear memory each mem_refill epoch
        if epoch%mem_refill == 0 and epoch != 0:
            memory.refill()
        pbar.update(1)
        
        # soft update weights at the end of epoch
        soft_update(q, q_target, 0.01)

        if info[4]/info[5] < min_wait and epoch > 50:
            torch.save(q.state_dict(), f'../model/dqn_{epoch}.pt')
            min_wait = info[4]/info[5]
        if epoch == 1599:
            torch.save(q.state_dict(), f'../model/dqn_{epoch}_final.pt')

    print('finished training')
