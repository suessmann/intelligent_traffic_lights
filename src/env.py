import numpy as np
import sys, os
from collections import deque
from src.generator import generate
from src.data_storage import StoreState

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7


class SumoIntersection:
    def __init__(self, path_bin, path_cfg, max_steps, n_cars):
        self.n_cars_out = generate(max_steps, n_cars)
        self.sumoCmd = [path_bin, "-c", path_cfg, '--waiting-time-memory', "4500", '--no-step-log', 'true', '-W', 'true']
        traci.start(self.sumoCmd)
        self.traci_api = traci.getConnection()

        self.path_cfg = path_cfg
        self.max_steps = max_steps
        self.time = 0
        self.old_phase = 0
        self.waiting_time = 0
        self.queue = 0

        self.done = False
        self.r_w = 0
        self.r_q = 0
        self.n_cars = n_cars

        self.r_epoch = 0
        self.w_epoch = 0
        self.q_epoch = 0


        self.waiting_time_float_av = deque(maxlen=max_steps)
        self.queue_float_av = deque(maxlen=max_steps)
        self.waiting_time_av = []
        self.queue_av = []
        self.r_av = []

        self.data = StoreState()

    def _get_info(self):
        # self._get_time()
        self._get_length()

        self.waiting_time_float_av.append(self.waiting_time)
        self.queue_float_av.append(self.queue)
        # self.waiting_time_av.append(self.waiting_time)
        # self.queue_av.append(self.queue)
        
        self.w_epoch += self.waiting_time
        self.q_epoch += self.queue

        return [np.mean(self.waiting_time_float_av), \
               np.mean(self.queue_float_av), \
               self.w_epoch/self.time, \
               self.q_epoch/self.time,
               self.w_epoch,
               self.q_epoch]

    def _tl_control(self, phase):
        if (self.time + 5) >= (self.max_steps - 10):
            self.done = True
            return

        #self.data.tl = torch.zeros((1, 1, 4))

        if phase != self.old_phase:
            yellow_phase_code = self.old_phase * 2 + 1
            self.traci_api.trafficlight.setPhase("TL", yellow_phase_code)

            if self.traci_api.simulation.getMinExpectedNumber() == 0:
                self.done = True
                return

            self.traci_api.simulationStep(self.time + 5.)
            self.time += 5

        self.data.p_tl[0, 0, phase] = 1

        if phase == 0:
            self.traci_api.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif phase == 1:
            self.traci_api.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif phase == 2:
            self.traci_api.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif phase == 3:
            self.traci_api.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

        self.traci_api.simulationStep(self.time + 2.)
        if self.traci_api.simulation.getMinExpectedNumber() == 0:
            self.done = True
            return

        self.time += 2

    def _get_length(self):
        halt_N = self.traci_api.edge.getLastStepHaltingNumber("N2TL")
        halt_S = self.traci_api.edge.getLastStepHaltingNumber("S2TL")
        halt_E = self.traci_api.edge.getLastStepHaltingNumber("E2TL")
        halt_W = self.traci_api.edge.getLastStepHaltingNumber("W2TL")
        self.queue = halt_N + halt_S + halt_E + halt_W

    def _get_time(self, car_id):
        self.waiting_time += self.traci_api.vehicle.getAccumulatedWaitingTime(car_id)

    def _get_state(self):
        car_list = self.traci_api.vehicle.getIDList()
        lanes_list = np.array(['W2TL_0', 'W2TL_1', 'W2TL_2', 'W2TL_3',
                               'E2TL_0', 'E2TL_1', 'E2TL_2', 'E2TL_3',
                               'N2TL_0', 'N2TL_1', 'N2TL_2', 'N2TL_3',
                               'S2TL_0', 'S2TL_1', 'S2TL_2', 'S2TL_3'])

        for car_id in car_list:
            lane_id = self.traci_api.vehicle.getLaneID(car_id)


            if lane_id not in lanes_list:
                continue

            self._get_time(car_id)

            lane_pos = self.traci_api.vehicle.getLanePosition(car_id)
            car_speed = self.traci_api.vehicle.getSpeed(car_id) / 13.89
            lane_pos = 750 - lane_pos

            if lane_pos > 112:
                continue

            # distance in meters from the traffic light -> mapping into cells
            for i, dist in enumerate(np.arange(7, 113, 7)):
                if lane_pos < dist:
                    lane_cell = i
                    break

            pos = np.where(np.array(lanes_list) == lane_id)[0][0]

            r = int(lane_id[-1])
            c = lane_cell

            for i, clown in enumerate(np.arange(4, 17, 4)):
                if pos < clown:
                    d = i
                    break

            self.data.p_position[0, d, r, c] = 1  # depth, row, col
            self.data.p_speed[0, d, r, c] = car_speed

    def _get_reward(self):
        if self.time < 4:
            return 0

        self.r_w = 0.85 * self.waiting_time_float_av[-2] - self.waiting_time_float_av[-1]

        # if self.r_w < 0:
        #     self.r_w *= 1.5
        #
        # self.r_av.append(self.r_w)

        return self.r_w

    def reset(self):
        self.n_cars_out = generate(self.max_steps, self.n_cars)

        self.time = 0
        self.old_phase = 0
        self.done = False
        self.r_epoch = 0
        self.w_epoch = 0
        self.q_epoch = 0

        self.traci_api.load(['-c', self.path_cfg, '--waiting-time-memory', "4500", '--no-step-log', 'true', '-W', 'true'])
        self.traci_api.simulationStep()


    def step(self, a):
        self._tl_control(a)  # self.traci_api.steps are here

        self.old_phase = a
        self.waiting_time = 0

        self._get_state()
        info = self._get_info()  # [0] av time, [1]: av queue

        self.r = self._get_reward()
        
        self.r_av.append(self.r)
        self.r_epoch += self.r

        info.append(self.r_epoch/self.time)
        info.append(self.r_epoch)
        info.append(self.time)
        if self.done:
            info.append(self.n_cars_out)

        return self.data, self.r, self.done, info

