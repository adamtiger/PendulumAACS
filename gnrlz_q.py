'''
This module creates a regression over the
Q-tables by a simple neural network.
'''

from q_learn import wrapper_parallel_execution
import math
import os
from logger import deserialize_numpy
from keras.models import Sequential
from keras.layers import Dense, Activation


def assemble_dataset():
    '''
    the goals are arranged on the circle
    starts at 60 degree then repeated with
    15 degrees frequency
    '''

    init_params = []

    for num in range(17):
        goal = [math.pi/3.0 + num * math.pi/12.0, 0.0]
        if goal[0] < math.pi:
            start = goal[0] - math.pi/6.0
            end = 2.0 * math.pi - start
        else:
            end = goal[0] + math.pi / 6.0
            start = 2.0 * math.pi - end

        init_params.append(
            [150, 'files/params' + str(num), ((start, end), (-1.0, 1.0)), goal]
        )

    return init_params


def run_training_batch():

    params = assemble_dataset()

    for p in params:
        wrapper_parallel_execution(p)


class Regression:

    def __init__(self, folder):
        # first read the tables

        dirs = os.listdir(folder)
        self.matrices = []
        for d in dirs:
            self.matrices.append(deserialize_numpy(d))

    def __create_model(self):

        net = Sequential()
        net.add(Dense(5, input_shape=(4)))
        net.add(Activation('relu'))
        net.add(Dense(2))
        net.add(Activation('softmax'))

        net.compile(optimizer='sgd', loss='binary_crossentropy')
        return net


