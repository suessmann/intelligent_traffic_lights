import sys, os
sys.path.append(os.path.join('/home/thelak-dev/sumo/tools'))

from datetime import datetime
from dqn import DQNetwork
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
BATCH_SIZE = 16
SIM_LEN = 4500
MEM_REFILL = 200
C = 10
EPOCHS = 1600
PRINT = 10
N_CARS= 1000
WEIGHTS_PATH = ''

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo-gui"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"

def weights_init(m):
    if isinstance(m, DQNetwork):
        return
    if not isinstance(m, nn.ReLU) and not isinstance(m, nn.Sequential):
        torch.nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def train(q, q_target, memory, optimizer):
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

    return mse

def log(epoch, total_steps, info, mse, writer):
    # to_file
    #print(f"{epoch},{total_steps},{info[0]},{info[1]},{info[2]},{info[3]},{info[4]}", file=f)
    #f.flush()

    # tensorboard
    # writer.add_scalar('env_metrics/mean_waiting', info[2], total_steps)
    # writer.add_scalar('env_metrics/mean_queue', info[3], total_steps)
    # writer.add_scalar('env_metrics/mean_reward', info[6], total_steps)
    writer.add_scalar('loss/loss', mse, total_steps) # 4 5
    #writer.flush()


if __name__ == '__main__':
    q = DQNetwork()

    try:
        q.load_state_dict(torch.load(WEIGHTS_PATH))
    except FileNotFoundError:
        q.apply(weights_init)
        print('No model weights found, initializing xavier_uniform')

    q_target = DQNetwork()
    q_target.load_state_dict(q.state_dict())

    memory = DQNBuffer(BUFFER_LIMIT, 0.1)

    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN, N_CARS)
    optimizer = torch.optim.RMSprop(q.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(logdir=f'logs/{int(datetime.now().timestamp())}', flush_secs=1)


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

        start_time = time.time()
        done_mask = 1.0
        pbar.set_description(f'EPOCH: {epoch}')
        mse = 0
        while not done:
            with torch.no_grad():
                a = q.predict(state.as_tuple, eps)
            state_prime, r, done, info = env.step(a)

            if done:
                done_mask = 0.0

            if r != 0:
                memory.add((state.position, state.speed,
                            state.tl, state_prime.position,
                            state_prime.speed, state_prime.tl,
                            a, r, done_mask))
                if epoch < 16:
                    memory.add_positive((state.position, state.speed,
                            state.tl, state_prime.position,
                            state_prime.speed, state_prime.tl,
                            a, r, done_mask))


            state = state_prime

            if memory.size > BATCH_SIZE:
                mse = train(q, q_target, memory, optimizer)

            #if total_steps % C == 0 and total_steps != 0:
            #    q_target.load_state_dict(q.state_dict())

            #if step % PRINT == 0:
            log(epoch, total_steps, info, mse, writer)
           
            step += 1
            total_steps += 1
        
        #  clear memory each MEM_REFILL epoch
        if epoch%MEM_REFILL == 0 and epoch != 0:
            memory.refill()
        pbar.update(1)
        
        if epoch % C == 0 and total_steps != 0:
            q_target.load_state_dict(q.state_dict())

        writer.add_scalar('env_metrics/mean_waiting', info[2], epoch)
        writer.add_scalar('env_metrics/mean_queue', info[3], epoch)
        writer.add_scalar('env_metrics/mean_reward', info[6], epoch)

        writer.add_scalar('env_metrics/cum_waiting', info[4], epoch)
        writer.add_scalar('env_metrics/cum_queue', info[5], epoch)
        writer.add_scalar('env_metrics/cum_reward', info[7], epoch)
        writer.add_scalar('env_status/n_cars', info[-1], epoch)
        writer.add_scalar('env_status/randomess', eps, epoch) 

        torch.save(q.state_dict(), '/Users/suess_mann/wd/tcqdrl/tca/saved_model/dqn.pt')

    print('finished training')
