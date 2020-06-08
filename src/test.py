from dqn import DQNetwork
from env import SumoIntersection
import torch

BATCH_SIZE = 32
SIM_LEN = 4500
WEIGHTS_PATH = '/Users/suess_mann/wd/tcqdrl/tca/saved_model/dqn_1246.pt'

sumoBinary = "/usr/local/opt/sumo/share/sumo/bin/sumo-gui"
sumoCmd = "/Users/suess_mann/wd/tcqdrl/tca/src/cfg/sumo_config.sumocfg"


if __name__ == '__main__':
    q = DQNetwork()
    q.eval()
    try:
        q.load_state_dict(torch.load(WEIGHTS_PATH))
    except FileNotFoundError:
        print('No model weights found')

    env = SumoIntersection(sumoBinary, sumoCmd, SIM_LEN, 1600)

    state, _, _, _ = env.step(0)
    done = False

    while not done:
        a = q.predict(state.as_tuple, 0.00)

        s_prime, r, done, info = env.step(a)
        print(r)
        s = s_prime









