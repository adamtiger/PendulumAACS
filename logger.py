'''
This module is responsible for
logging the most important information
into a file.
The file names are fixed only the folder
name is necessary
'''

from enum import Enum
import csv
import json
import pandas as pd
import numpy as np

log_name = 'logs.log'
train_ret_name = 'tret.csv'
return_name = 'return.csv'
loss_name = 'loss.csv'
video_name = 'videos'
numpy_mtx = 'matx.json'


def join(folder, name):
    return folder + '/' + name


def deserialize_numpy(foldername):
    numpy_obj_read = open(join(foldername, numpy_mtx), 'r')
    mtx = json.load(numpy_obj_read)
    numpy_obj_read.close()
    return np.array(mtx['weight'])


class Mode(Enum):

    STDOUT = 1
    LOG_F = 2   # log file
    TRAIN_RET_F = 3  # return during training
    RET_F = 4   # return csv
    STD_LOG = 5  # both STDOUT and LOG file
    LOSS_F = 6  # loss file
    NUMPY = 7  # numpy matrix


class Logger:

    def __init__(self, foldername):
        self.foldername = foldername

        self.log_file = open(join(foldername, log_name), 'a', buffering=1)
        self.train_ret_csv = open(join(foldername, train_ret_name), 'a', newline='', buffering=1)
        self.train_ret_csv_obj = csv.writer(self.train_ret_csv)
        self.return_csv = open(join(foldername, return_name), 'a', newline='', buffering=1)
        self.return_csv_obj = csv.writer(self.return_csv)
        self.loss_csv = open(join(foldername, loss_name), 'a', newline='', buffering=1)
        self.loss_csv_obj = csv.writer(self.loss_csv)
        self.numpy_obj = open(join(foldername, numpy_mtx), 'a')

        self.log_funcs = [self.__log_STDOUT,
                          self.__log_LOG_F,
                          self.__log_TRAIN_RET_F,
                          self.__log_RET_F,
                          self.__log_STD_LOG,
                          self.__log_LOSS_F,
                          self.__log_NUMPY
                          ]

    def log(self, mode, msg):
        success = False
        for func in self.log_funcs:
            success = success or func(mode, msg)

        if not success:
            raise AttributeError('Unknown mode ' + str(mode) + ' in logger!')

    def dataframes(self):
        df_trainret = pd.read_csv(join(self.foldername, train_ret_name), names=['iteration', 'episode', 'return'])
        df_ret = pd.read_csv(join(self.foldername, return_name), names=['iteration', 'episode', 'return'])
        df_loss = pd.read_csv(join(self.foldername, return_name), names=['iteration', 'episode', 'loss'])
        return df_trainret, df_ret, df_loss

    def video_folder(self):
        return join(self.foldername, video_name)

    def deserialize_numpy(self):
        numpy_obj_read = open(join(self.foldername, numpy_mtx), 'r')
        mtx = json.load(numpy_obj_read)
        numpy_obj_read.close()
        return np.array(mtx['weight'])

    def __del__(self):
        self.log_file.flush()
        self.train_ret_csv.flush()
        self.return_csv.flush()
        self.numpy_obj.flush()

        self.log_file.close()
        self.train_ret_csv.close()
        self.return_csv.close()
        self.numpy_obj.close()

    # -----------------------------------
    # Private functions.

    def __log_STDOUT(self, mode, msg):

        if mode == Mode.STDOUT:
            print(msg)
            return True

        return False

    def __log_LOG_F(self, mode, msg):

        if mode == Mode.LOG_F:
            self.log_file.write(msg + '\n')
            return True

        return False

    def __log_TRAIN_RET_F(self, mode, msg):

        if mode == Mode.TRAIN_RET_F:  # msg: [iteration, episode, loss]
            self.train_ret_csv_obj.writerow(msg)
            return True

        return False

    def __log_RET_F(self, mode, msg):

        if mode == Mode.RET_F:  # msg: [iteration, episode, return]
            self.return_csv_obj.writerow(msg)
            return True

        return False

    def __log_STD_LOG(self, mode, msg):

        if mode == Mode.STD_LOG:
            print(msg)
            self.log_file.write(msg + '\n')
            return True

        return False

    def __log_LOSS_F(self, mode, msg):

        if mode == Mode.LOSS_F:  # msg: [iteration, episode, loss]
            self.loss_csv_obj.writerow(msg)
            return True

        return False

    def __log_NUMPY(self, mode, matrix):

        if mode == Mode.NUMPY:  # msg: [iteration, episode, loss]
            json.dump({'weight': matrix.tolist()}, self.numpy_obj)
            return True

        return False