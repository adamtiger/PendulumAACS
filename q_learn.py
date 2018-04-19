'''
This module provides functions to solve
the Pendulum problem for a fixed goal.
The result is a table containing
Q-values for s, a pairs.
'''

import math
import numpy as np
import random
from logger import Mode, Logger
import gym
from gym.wrappers import Monitor

# CONSTANTS of matrix size
ROWS = 360 + 1  # th
COLS = 200 + 1  # thdot
DPTS = 1 + 1   # action

PI = math.pi


def wrapper_parallel_execution(init_params):
    max_ep = init_params[0]
    folder = init_params[1]
    constraints = init_params[2]
    goal = init_params[3]

    q = Q(max_ep, folder, constraints, goal)

    q.run()


class Q:

    def __init__(self, max_ep, folder, constraints, goal):  # constraints: the boundaries for s in Q^g(s, a)
        self.max_episode = max_ep
        self.log = Logger(folder)
        self.env = gym.make('PendulumGoal-v0')
        self.constraints = constraints
        self.start = None
        self.goal = goal

        self.q = np.zeros((ROWS, COLS, DPTS))
        self.alpha_rec = np.ones((ROWS, COLS, DPTS))
        self.lr = 1.0
        self.gamma = 0.8

        self.epsilon = 0.4

    def __env_init_fn(self):
        self.env.reset()
        th = random.uniform(self.constraints[0][0], self.constraints[0][1])
        thdot = random.uniform(self.constraints[1][0], self.constraints[1][1])
        self.env.setup([np.array([th, thdot]), self.goal])
        return np.array([math.cos(th), math.sin(th), thdot])

    def __indexer(self, state, action=None):  # state=(cos(th), sin(th), thdot)
        c, s, dot = state[0], state[1], state[2]
        theta = math.acos(math.fabs(c))
        if c >= 0 and s >= 0:
            theta = theta
        elif c < 0 and s > 0:
            theta = PI - theta
        elif c < 0 and s < 0:
            theta = PI + theta
        else:
            theta = 2.0 * PI - theta

        row = int(round(math.degrees(theta)))

        col = int(round((dot + 8.0) / 16.0 * (COLS - 1)))

        if action is None:
            return (row, col)
        else:
            dph = 1 if action == 1.0 else 0 #int(round((action + 1.0)))#/ 4.0 * (DPTS -1)))
            return (row, col, dph)

    def __argmin(self, state):
        idx = self.__indexer(state)
        return np.argmin(self.q[idx[0], idx[1], :])

    def __min(self, state):
        idx = self.__indexer(state)
        return np.min(self.q[idx[0], idx[1], :])

    def __update(self, state, action, cost, next_state, done, next_action=None):
        idx = self.__indexer(state, action)
        idx_next = self.__indexer(next_state, next_action)
        self.q[idx] = ((1 - self.lr) * self.q[idx]
                       +
                       self.lr * (cost + 0.0 if done else self.gamma * self.q[idx_next]))
        self.alpha_rec[idx] += 1

    def __select_act(self, state, explorefree=False):
        best_act = self.__argmin(state)
        dice = random.randint(1, 1000)
        if dice > self.epsilon * 1000 or explorefree:
            return best_act * 2.0 - 1.0
        else:
            idcs = [idx for idx in range(0, DPTS)]
            idcs.remove(best_act)
            return random.choice(idcs) * 2.0 - 1.0

    def __decrease_eps(self):
        self.epsilon = max(self.epsilon - 0.005, 0.1)

    def __decrease_lr(self):
        self.lr = max(self.lr - 0.005, 0.001)

    def run(self):

        self.log.log(Mode.STDOUT, 'Learning started.')

        total_cost = 0
        episode = 0
        cntr = 0
        done = True
        state = None
        min_cost = 0
        action = [0.0]

        while episode < self.max_episode:

            if done or cntr % 50 == 0:
                episode += 1
                state = self.__env_init_fn()
                action = self.__select_act(state)
                self.log.log(Mode.TRAIN_RET_F, [cntr, episode, total_cost])
                total_cost = 0
                self.__decrease_eps()
                self.__decrease_lr()

                if episode % 10 == 0:
                    rtn, scs = self.evaluate()
                    self.log.log(Mode.STD_LOG, str(episode) + ': ' + str(rtn) + ' cost: ' + str(min_cost) + ' scs: ' + str(scs))
                    self.log.log(Mode.RET_F, [cntr, episode, rtn])

            next_state, cost, done, inf = self.env.step([action])
            next_action = self.__select_act(next_state)
            min_cost = inf['min_cost']
            self.__update(state, action, cost, next_state, done, next_action)
            action = next_action

            cntr += 1
            total_cost += cost
            state = next_state

        self.log.log(Mode.NUMPY, self.q)
        self.log.log(Mode.STDOUT, 'Learning finished. matrix was saved.')

    def evaluate(self, video=False):

        total_cost = 0
        cntr = 0
        episode = 0
        done = True
        success = -1
        state = None

        orig_env = self.env
        if video:
            self.env = Monitor(orig_env, self.log.video_folder())

        while episode < 20:

            if done:
                success += 1

            if done or cntr > 2000:
                if video:
                    print(str(episode) + ' ' + str(total_cost))
                episode += 1
                state = self.__env_init_fn()
                cntr = 0

            action = self.__select_act(state, explorefree=False)
            state, cost, done, _ = self.env.step([action])
            #self.env.render()
            total_cost += cost
            cntr += 1

        self.env = orig_env

        return total_cost / 20.0, success

    def load_from_file(self):
        self.q = self.log.deserialize_numpy()

    def __del__(self):
        self.env.close()
