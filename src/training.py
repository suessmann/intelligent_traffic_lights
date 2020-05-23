from dqn import DQNetwork
from env import SumoIntersection
from memory import DQNBuffer
from data_storage import StoreState
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time

LEARNING_RATE = 0.0005
GAMMA = 0.95
BUFFER_LIMIT = 50000
BATCH_SIZE = 32
SIM_LEN = 4500
MEM_REFILL = 250
C = 150
EPOCHS = 1600
PRINT = 100
N_CARS=2250
WEIGHTS_PATH = ''

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"

def train(q, q_target, memory, optimizer):
    # for i in range(10):
    state, a, r, state_prime, done_mask = memory.sample(BATCH_SIZE)

    q_out = q(state)
    q_a = q_out.gather(1, a)
    max_q_prime = q_target(state_prime).max(1)[0].unsqueeze(1)
    target = r + GAMMA * max_q_prime * done_mask
    criterion = nn.MSELoss()
    loss = criterion(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
    optimizer = torch.optim.RMSprop(q.parameters(), lr=LEARNING_RATE)

    # number of different simulations to perform
    total_time = time.time()
    f = open('/Users/suess_mann/wd/tcqdrl/tca/tests/run_avg.csv', 'w')
    print("epoch,step,mov_wait,mov_que,mean_w,mean_que,r", file=f)

    i=0
    for epoch in np.arange(EPOCHS):
        eps = max(0.01, 0.08 - 0.01*(epoch/200))
        step = 0

        if epoch != 0:
            env.reset()

        state, _, _, _ = env.step(0)
        done = False

        start_time = time.time()
        done_mask = 1.0

        while not done:
            a = q.predict(state.as_tuple, eps)
            state_prime, r, done, info = env.step(a)

            if done:
                done_mask = 0.0

            memory.add((state.position, state.speed,
                        state.tl, state_prime.position,
                        state_prime.speed, state_prime.tl,
                        a, r, done_mask))

            state = state_prime

            if memory.size > BATCH_SIZE:
                train(q, q_target, memory, optimizer)

            if step % C == 0:
                q_target.load_state_dict(q.state_dict())

            if step % PRINT == 0:
                print(f"EPOCH: {epoch}, step: {step}, "
                      f"moving waiting: {info[0]}, moving queue: {info[1]}, "
                      f"mean waiting: {info[2]}, mean queue: {info[3]}, mean reward {info[4]}")
                print(f"{epoch},{i},{info[0]},{info[1]},{info[2]},{info[3]},{info[4]}", file=f)
                f.flush()

            step += 1
            i += 1


        #  clear memory each MEM_REFILL epoch
        if epoch%MEM_REFILL == 0 and epoch != 0:
            memory.refill()

        torch.save(q.state_dict(), '/Users/suess_mann/wd/tcqdrl/tca/saved_model/dqn.pt')

        print(f"EPOCH {epoch} finished in {(time.time() - start_time)} sec.,"
              f"TOTAL time: {(time.time() - total_time) / 3600} hours")

    f.close()
    print('finished training')



