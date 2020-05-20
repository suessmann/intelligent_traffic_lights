from dqn import DQNetwork
from env import SumoIntersection
from memory import DQNBuffer
import torch
import numpy as np
import time
from training import unsqueeze

BATCH_SIZE = 32
SIM_LEN = 4500
WEIGHTS_PATH = '/Users/suess_mann/wd/tcqdrl/tca/saved_model/dqn.pt'

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo-gui"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"


if __name__ == '__main__':
    q = DQNetwork()

    try:
        q.load_state_dict(torch.load(WEIGHTS_PATH))
    except FileNotFoundError:
        print('No model weights found')

    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN)

    s, _, _, _ = env.step(0)
    done = False

    while not done:
        a = q.predict(unsqueeze(s), 0)
        s_prime, r, done, info = env.step(a)

        s = s_prime









