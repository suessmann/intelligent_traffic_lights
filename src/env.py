import traci
import numpy as np
import random
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

        self.done = False
        self.path_cfg = path_cfg
        self.max_steps = max_steps
        self.time = 0
        self.old_phase = 0

        self.r_w = 0
        self.r_q = 0

        generate(self.max_steps, self.max_steps//2)

        self.waiting_time_float_av = []
        self.queue_av = []

    def _get_info(self):
        self.waiting_time_float_av.append(self.waiting_time)
        self.queue_av.append(self.queue)
        return np.round(np.average(self.waiting_time_float_av), 2), \
               np.round(np.average(self.queue_av), 2)

    def _tl_control(self, phase):
        self.tl_state = np.zeros((4))
        if phase != self.old_phase:
            yellow_phase_code = self.old_phase * 2 + 1
            traci.trafficlight.setPhase("TL", yellow_phase_code)
            for i in range(5):  # 5 is a yellow duration
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

        for i in range(2):  # 2 is a duration of green light
            traci.simulationStep()
            self.time += 1

    def _get_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        self.queue = halt_N + halt_S + halt_E + halt_W

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

    def _get_reward(self):
        if self.r_q == self.queue:
            self.r_q = 0
        else:
            self.r_q = self.queue - self.r_q

        if self.r_w == self.waiting_time:
            self.r_w = 0
        else:
            self.r_w = self.waiting_time - self.r

        return self.r_w * 0.5 + self.r_q * 0.5

    def reset(self):
        generate(self.max_steps, self.max_steps // 3)
        self.time = 0
        self.old_phase = 0
        self.done = False
        traci.load(['-c', self.path_cfg])
        traci.simulationStep()


    def step(self, a): # обнулить все
        self._tl_control(a)  # traci.steps are here
        # traci.simulationStep()

        self.pos_matrix = np.zeros((4, 15, 4))  # rows, cols, depth
        self.vel_matrix = np.zeros((4, 15, 4))
        self.waiting_time = 0
        self.queue = 0
        self.old_phase = a

        self._get_length()
        self._get_state()
        self.r = self._get_reward()

        info = self._get_info() # [0] av time, [1]: av queue

        if self.time >= self.max_steps:
            self.done = True

        return (self.pos_matrix, self.vel_matrix, self.tl_state), \
               -self.r, \
               self.done, \
               info  # s_prime, r, done, info
