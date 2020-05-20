from dqn import DQNetwork
from env import SumoIntersection
import torch
from traning import unsqueeze
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo-gui"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"
EPOCHS = 100
SIM_LEN = 4500
C = 50


if __name__ == '__main__':
    q = DQNetwork()
    q.load_state_dict(torch.load('/Users/suess_mann/wd/tcqdrl/tca/saved_model/dqn.pt'))

    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN)

    # f = open('run_avg.txt', 'w')

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
            # info = (r[0], r[1])
            r = r[1]
            s = s_prime

            # if memory.size > 8000:
            #     train(q, q_target, memory, optimizer)

            if env.time % C == 0:
                print(f"EPOCH: {n_epi}, step: {env.time}, total time waiting: {r},"
                      f"running average of 100: {info}")
                # print(info, sep=',', file=f)