import traci
import numpy as np
import random
import torch
from generator import generate

PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6
PHASE_EWL_YELLOW = 7


class SumoIntersection:
    def __init__(self, path_bin, path_cfg, max_steps):
        self.sumoCmd = [path_bin, "-c", path_cfg]
        traci.start(self.sumoCmd)

        self.path_cfg = path_cfg
        self.max_steps = max_steps
        self.cum_queue = 0
        self.time = 0
        self.old_phase = 0
        generate(self.max_steps, self.max_steps // 3)

    def _get_info(self):
        pass

    def _tl_control(self, phase):
        self.tl_state = np.zeros((4))
        if phase != self.old_phase:
            yellow_phase_code = self.old_phase * 2 + 1
            traci.trafficlight.setPhase("TL", yellow_phase_code)
            for i in range(0, 3):  # 3 is a yellow duration
                traci.simulationStep()
                self.time += 1
        self.tl_state[phase] = 1

        if phase == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif phase == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif phase == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif phase == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_length(self):
        self.queue = self.pos_matrix.sum()

    def _get_time(self, car_id):
        self.waiting_time += traci.vehicle.getWaitingTime(car_id)

    def _get_state(self):
        car_list = traci.vehicle.getIDList()
        lanes_list = np.array(['W2TL_0', 'W2TL_1', 'W2TL_2', 'W2TL_3',
                               'E2TL_0', 'E2TL_1', 'E2TL_2', 'E2TL_3',
                               'N2TL_0', 'N2TL_1', 'N2TL_2', 'N2TL_3',
                               'S2TL_0', 'S2TL_1', 'S2TL_2', 'S2TL_3'])
        self.waiting_time = 0

        for car_id in car_list:
            lane_id = traci.vehicle.getLaneID(car_id)



            if lane_id not in lanes_list:
                continue

            self._get_time(car_id)

            lane_pos = traci.vehicle.getLanePosition(car_id)
            car_speed = traci.vehicle.getSpeed(car_id) / 13.89
            lane_pos = 750 - lane_pos

            if lane_pos > 98:
                continue
            # print(f'lane pos: {lane_pos}')
            # distance in meters from the traffic light -> mapping into cells
            for i, dist in enumerate(np.arange(0, 99, 7)):
                if lane_pos < dist:
                    lane_cell = i
                    break

            pos = np.where(np.array(lanes_list) == lane_id)[0][0]

            r = int(lane_id[-1])
            c = lane_cell

            for i, clown in enumerate(np.arange(4, 17, 4)):
                if pos < clown :
                    d = i
                    break

            self.pos_matrix[r, c, d] = 1
            self.vel_matrix[r, c, d] = car_speed

    def reset(self):
        generate(self.max_steps, self.max_steps // 3)
        self.cum_queue = 0
        self.time = 0
        self.old_phase = 0
        traci.load(['-c', self.path_cfg])
        traci.simulationStep()


    def step(self, a): # обнулить все
        self.time += 1
        traci.simulationStep()

        self.waiting_time = 0
        self.queue = 0
        self.pos_matrix = np.zeros((4, 15, 4))  # rows, cols, depth
        self.vel_matrix = np.zeros((4, 15, 4))

        self._get_state()
        self._get_length()
        self._tl_control(a)

        self.old_phase = a

        s_prime = (self.pos_matrix, self.vel_matrix, self.tl_state)

        r = (-self.queue, -self.waiting_time)
        done = False
        # info = self._get_info()
        info = 0

        if self.time >= self.max_steps:
            done = True

        return s_prime, r, done, info
