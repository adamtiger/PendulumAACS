'''
This module creates a regression over the
Q-tables by a simple neural network.
'''

from q_learn import ROWS, COLS, indexer
import math
import itertools
import numpy as np
import logger
from logger import deserialize_numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback


class LogLosses(Callback):

    def on_train_begin(self, logs={}):
        self.log = logger.Logger('files')
        self.cntr = 0

    def on_epoch_end(self, epoch, logs={}):

        msg = [self.cntr, logs.get('loss'), logs.get('val_loss'), logs.get('acc'), logs.get('val_acc')]
        self.log.log(logger.Mode.LOSS_F, msg)
        self.cntr += 1

class Regression:

    def __init__(self, fd_num):
        # first read the tables
        print('Start reading data.')
        samples_x = []
        samples_y = []
        self.train_data = {}
        for d in fd_num:
            x, y = self.__create_samples(d[0], d[1])
            samples_x.append(x)
            samples_y.append(y)

        print('Reading from file was finished. Concatenating to numpy matrix.')
        if len(fd_num) > 1:
            self.train_data['x'] = np.concatenate(samples_x, axis=0)
            self.train_data['y'] = np.concatenate(samples_y, axis=0)
        else:
            self.train_data['x'] = samples_x[0]
            self.train_data['y'] = samples_y[0]

        print('Creating network.')
        self.net = self.__create_model()
        self.log_losses = LogLosses()

    def __create_model(self):

        net = Sequential()
        net.add(Dense(5, input_shape=(4,)))
        net.add(Activation('relu'))
        net.add(Dense(2))
        net.add(Activation('softmax'))

        net.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return net

    def __create_samples(self, folder, num):
        # in case of files with json extension
        goal = [math.pi / 3.0 + num * math.pi / 12.0, 0.0]

        # generating states
        thetas = [th * math.pi / (ROWS - 1) for th in range(ROWS)]
        thdots = [thdot * 16.0 / (COLS - 1) - 8.0 for thdot in range(COLS)]
        thxthdots = itertools.product(thetas, thdots)

        # read the matrix from the file
        mtx = deserialize_numpy(folder)

        train_x = []
        train_y = []
        for thxthdot in thxthdots:
            idx = indexer(thxthdot)
            act_vals = (np.array([1, 0]) if np.argmax(mtx[idx[0], idx[1], :]) == 0 else np.array([0, 1]))
            train_x.append(np.array([thxthdot[0], thxthdot[1], goal[0], goal[1]]))
            train_y.append(act_vals)

        return np.vstack(train_x), np.vstack(train_y)

    def regression(self, batch_size, epochs):
        print('Start regression.')
        history = self.net.fit(x=self.train_data['x'], y=self.train_data['y'],
                     batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[self.log_losses])
        self.net.save_weights('files/gen_weights.h5')
        return history
