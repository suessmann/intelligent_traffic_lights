from dqn import DQNetwork
from env import SumoIntersection
from memory import DQNBuffer
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

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo-gui"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"

def train(q, q_target, memory, optimizer):
    # for i in range(10):
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

    f = open('/Users/suess_mann/wd/tcqdrl/tca/tests/run_avg.txt', 'w')
    print("step, av_time_wait, av_que_len", file=f)

    for epoch in np.arange(EPOCHS):
        eps = max(0.01, 0.08 - 0.01*(epoch/200))
        step = 0

        if epoch != 0:
            env.reset()

        s, _, _, _ = env.step(0)
        done = False

        start_time = time.time()
        while not done:
            a = q.predict(unsqueeze(s), eps)
            s_prime, r, done, info = env.step(a)

            memory.add((s, a, r, s_prime))
            s = s_prime

            if memory.size > BATCH_SIZE:
                train(q, q_target, memory, optimizer)

            if step % C == 0:
                q_target.load_state_dict(q.state_dict())

            if step % PRINT == 0:
                print(f"EPOCH: {epoch}, step: {step}, "
                      f"average waiting: {info[0]}, average queue size: {info[1]}")
                print(f"{step}, {info[0]}, {info[1]}", file=f)

            step += 1
        


        #  clear memory each MEM_REFILL epoch
        if epoch%MEM_REFILL == 0 and epoch != 0:
            memory.refill()

        torch.save(q.state_dict(), '/Users/suess_mann/wd/tcqdrl/tca/saved_model/dqn.pt')

        print(f"EPOCH {epoch} finished in {start_time - time.time()}")

    f.close()
    print('finished training')



