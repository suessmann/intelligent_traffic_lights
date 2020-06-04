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

LEARNING_RATE = 0.005
GAMMA = 0.95
BUFFER_LIMIT = 50000
BATCH_SIZE = 32
SIM_LEN = 4500
MEM_REFILL = 250
C = 100000
EPOCHS = 1600
PRINT = 100
N_CARS=1000
WEIGHTS_PATH = ''

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo-gui"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"


def train(q, q_target, memory, optimizer, mse_mean):
    # for i in range(10):
    state, a, r, state_prime, done_mask = memory.sample(BATCH_SIZE)

    q_out = q(state)
    q_a = q_out.gather(1, a)
    max_q_prime = q_target(state_prime).max(1)[0].unsqueeze(1)
    target = r + GAMMA * max_q_prime * done_mask
    criterion = nn.MSELoss()
    loss = criterion(q_a, target)
    mse_mean.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    for param in q.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def log(epoch, total_steps, info, mse_mean, writer):
    # to_file
    print(f"{epoch},{total_steps},{info[0]},{info[1]},{info[2]},{info[3]},{info[4]}", file=f)
    f.flush()

    # tensorboard
    writer.add_scalar('env_metrics/mean_waiting', info[2], total_steps)
    writer.add_scalar('env_metrics/mean_queue', info[3], total_steps)
    writer.add_scalar('env_metrics/mean_reward', info[4], total_steps)
    writer.add_scalar('loss/loss', np.mean(mse_mean), total_steps)
    writer.flush()


if __name__ == '__main__':
    q = DQNetwork()

    try:
        q.load_state_dict(torch.load(WEIGHTS_PATH))
    except FileNotFoundError:
        print('No model weights found, initializing random')

    q_target = DQNetwork()
    q_target.load_state_dict(q.state_dict())

    memory = DQNBuffer(BUFFER_LIMIT)
    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN, N_CARS)
    optimizer = torch.optim.Adam(q.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter()
    total_time = time.time()
    f = open('/Users/suess_mann/wd/tcqdrl/tca/tests/run_avg.csv', 'w')
    print("epoch,step,mov_wait,mov_que,mean_w,mean_que,r", file=f)


    mse_mean = [0]
    total_steps=0
    for epoch in np.arange(EPOCHS):
        eps = max(0.01, 0.08 - 0.01*(epoch/200))
        step = 0

        if epoch != 0:
            env.reset()

        state, _, _, _ = env.step(0)
        done = False

        start_time = time.time()
        done_mask = 1.0
        pbar = tqdm(total=4500)
        pbar.set_description(f'EPOCH: {epoch}')
        while not done:
            a = q.predict(state.as_tuple, eps)
            state_prime, r, done, info = env.step(a)

            if done:
                done_mask = 0.0

            if step > 60:
                memory.add((state.position, state.speed,
                            state.tl, state_prime.position,
                            state_prime.speed, state_prime.tl,
                            a, r, done_mask))

            state = state_prime

            if memory.size > BATCH_SIZE:
                train(q, q_target, memory, optimizer, mse_mean)

            if total_steps % C == 0 and total_steps != 0:
                q_target.load_state_dict(q.state_dict())

            if step % PRINT == 0:
                log(epoch, total_steps, info, mse_mean, writer)

            step += 1
            total_steps += 1
            pbar.n = info[-1]
            pbar.update()


        #  clear memory each MEM_REFILL epoch
        if epoch%MEM_REFILL == 0 and epoch != 0:
            memory.refill()

        torch.save(q.state_dict(), '/Users/suess_mann/wd/tcqdrl/tca/saved_model/dqn.pt')

    f.close()
    print('finished training')
