import traci
import numpy as np
import random
import torch
from collections import deque
from generator import generate
from data_storage import StoreState

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
        self.sumoCmd = [path_bin, "-c", path_cfg]
        traci.start(self.sumoCmd)
        self.traci_api = traci.getConnection()

        self.path_cfg = path_cfg
        self.max_steps = max_steps
        self.time = 0
        self.old_phase = 0
        self.waiting_time = 0
        self.queue = 0

        self.n_cars = n_cars

        self.done = False
        self.r_w = 0
        self.r_q = 0

        generate(self.max_steps, n_cars)

        self.waiting_time_float_av = deque(maxlen=max_steps)
        self.queue_float_av = deque(maxlen=max_steps)
        self.waiting_time_av = []
        self.queue_av = []

        self.data = StoreState()

    def _get_info(self):
        self._get_time()
        self._get_length()

        self.waiting_time_float_av.append(self.waiting_time)
        self.queue_float_av.append(self.queue)
        self.waiting_time_av.append(self.waiting_time)
        self.queue_av.append(self.queue)
        return np.round(np.mean(self.waiting_time_float_av), 2), \
               np.round(np.mean(self.queue_float_av), 2), \
               np.round(np.mean(self.waiting_time_av)), \
               np.round(np.mean(self.queue_av))

    def _tl_control(self, phase):
        if (self.time + 7) >= self.max_steps:
            self.done = True
            return

        self.data.s_tl = torch.zeros((1, 1, 4))
        if phase != self.old_phase:
            yellow_phase_code = self.old_phase * 2 + 1
            self.traci_api.trafficlight.setPhase("TL", yellow_phase_code)
            for i in range(5):  # 5 is a yellow duration
                self.traci_api.simulationStep()
                self.time += 1

        self.data.s_tl[0, 0, phase] = 1

        if phase == 0:
            self.traci_api.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif phase == 1:
            self.traci_api.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif phase == 2:
            self.traci_api.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif phase == 3:
            self.traci_api.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

        for i in range(2):  # 2 is a duration of green light
            self.traci_api.simulationStep()
            self.time += 1

    def _get_length(self):
        halt_N = self.traci_api.edge.getLastStepHaltingNumber("N2TL")
        halt_S = self.traci_api.edge.getLastStepHaltingNumber("S2TL")
        halt_E = self.traci_api.edge.getLastStepHaltingNumber("E2TL")
        halt_W = self.traci_api.edge.getLastStepHaltingNumber("W2TL")
        self.queue = halt_N + halt_S + halt_E + halt_W

    def _get_time(self):
        wait_N = self.traci_api.edge.getWaitingTime("N2TL")
        wait_S = self.traci_api.edge.getWaitingTime("S2TL")
        wait_E = self.traci_api.edge.getWaitingTime("E2TL")
        wait_W = self.traci_api.edge.getWaitingTime("W2TL")
        self.waiting_time = wait_N + wait_S + wait_E + wait_W

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

            lane_pos = self.traci_api.vehicle.getLanePosition(car_id)
            car_speed = self.traci_api.vehicle.getSpeed(car_id) / 13.89
            lane_pos = 750 - lane_pos

            if lane_pos > 98:
                continue

            # distance in meters from the traffic light -> mapping into cells
            for i, dist in enumerate(np.arange(0, 99, 7)):
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

            self.data.s_position[0, d, r, c] = 1
            self.data.s_speed[0, d, r, c] = car_speed

    def _get_reward(self):
        if self.time < 3:
            return 0

        self.r_q = self.queue_av[-2] - self.queue_av[-1]

        if self.r_q < 0:
            self.r_q *= 1.5
        if self.r_q == 0:
            self.r_q = -10

        return self.r_q

    def reset(self):
        generate(self.max_steps, self.n_cars)
        self.time = 0
        self.old_phase = 0
        self.done = False
        self.traci_api.load(['-c', self.path_cfg])
        self.traci_api.simulationStep()


    def step(self, a):
        self._tl_control(a)  # self.traci_api.steps are here

        self.data.s_position = torch.zeros((1, 4, 4, 15))  # depth, rows, cols
        self.data.s_speed = torch.zeros((1, 4, 4, 15))
        self.old_phase = a
        info = self._get_info()  # [0] av time, [1]: av queue



        self._get_state()
        self.r = self._get_reward()

        return self.data, self.r, self.done, info
        # return self.pos_matrix, self.vel_matrix, self.tl_state, \
        #         \
        #        self.done, \
        #        info  # s_prime, r, done, info
