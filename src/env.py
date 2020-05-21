import traci
import numpy as np
import random
from collections import deque
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
    def __init__(self, ip, port, max_steps, path_bin=0, path_cfg=0):
        # self.sumoCmd = [path_bin, "-c", path_cfg]
        # traci.start(self.sumoCmd, label='sim')
        self.traci_api = traci.connect(port=port, host=ip)

        self.done = False
        self.path_cfg = path_cfg
        self.max_steps = max_steps
        self.time = 0
        self.old_phase = 0

        self.r_w = 0
        self.r_q = 0

        generate(self.max_steps, self.max_steps//2)

        self.waiting_time_float_av = deque(maxlen=1000)
        self.queue_av = deque(maxlen=1000)

    def _get_info(self):
        self.waiting_time_float_av.append(self.waiting_time)
        self.queue_av.append(self.queue)
        return np.round(np.mean(self.waiting_time_float_av), 2), \
               np.round(np.mean(self.queue_av), 2)

    def _tl_control(self, phase):
        self.tl_state = np.zeros((4))
        if phase != self.old_phase:
            yellow_phase_code = self.old_phase * 2 + 1
            self.traci_api.trafficlight.setPhase("TL", yellow_phase_code)
            for i in range(5):  # 5 is a yellow duration
                self.traci_api.simulationStep()
                self.time += 1

        self.tl_state[phase] = 1

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
                if pos < clown:
                    d = i
                    break

            self.pos_matrix[r, c, d] = 1
            self.vel_matrix[r, c, d] = car_speed

    def _get_reward(self):
        if self.r_q == self.queue:
            self.r_q = 0
        else:
            self.r_q = self.queue - self.r_q

        if self.r_w == self.waiting_time:
            self.r_w = 0
        else:
            self.r_w = self.waiting_time - self.r_w

        #return #self.r_w * 0.5 + self.r_q * 0.5
        return self.r_w

    def reset(self):
        generate(self.max_steps, self.max_steps // 3)
        self.time = 0
        self.old_phase = 0
        self.done = False
        self.traci_api.load(['-c', self.path_cfg])
        self.traci_api.simulationStep()


    def step(self, a): # обнулить все
        self._tl_control(a)  # self.traci_api.steps are here
        # self.traci_api.simulationStep()

        self.pos_matrix = np.zeros((4, 15, 4))  # rows, cols, depth
        self.vel_matrix = np.zeros((4, 15, 4))
        self.old_phase = a

        self._get_length()
        self._get_time()
        self._get_state()
        self.r = self._get_reward()

        info = self._get_info() # [0] av time, [1]: av queue

        if self.time >= self.max_steps:
            self.done = True

        return (self.pos_matrix, self.vel_matrix, self.tl_state), \
               -self.r, \
               self.done, \
               info  # s_prime, r, done, info
