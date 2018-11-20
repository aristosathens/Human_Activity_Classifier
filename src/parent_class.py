# Zach Blum, Navjot Singh, Aristos Athens

"""
    Defines enumerated types, DataLoader parent class and various subclasses like RegressionLearner, etc.
"""

import numpy as np
import pandas as pd

import util
from enum_types import *


# ------------------------------------- Class ------------------------------------- #

class DataLoader:
    """
        Use to load data and store it as an object.
        m - number of data points
        n - number of input features
        k - number of classes
    """

    def __init__(self,
                 file_name,
                 output_folder,
                 model_folder,
                 percent_validation=0.15,
                 batch_size=None,
                 learning_rate=0.1,
                 epsilon=1e-2,
                 epochs=None,

                 architecture=None,
                 activation=None,
                 optimizer=None,
                 metric=None,
                 loss=None,
                 ):
        '''
            Initialize DataLoader

            file_name - path to data file
            output_folder - path to output directory
            percent_validation - fraction of data to use for validation
            batch_size - size of each batch for training. If None, don't use batches. Can be overwritten in self.train()
            learning_rate - learning rate for update rule
            epsilon - criterion for convergence
            epochs - number of rounds of training

            architecture - This will set up a keras neural net with a predefined architecure. Use this OR the following
                            paramgs. Don't need to use both.
            activation - activate type. Should be of type ActivationType. Currently only used for DeepLearner
            optimizer - Which Keras optimizer to use. Should be of type OptimizerType. Currently only used for DeepLearner
            metric - Which keras metric to use. Should be of type AccuracyType. Currently only used for DeepLearner
            loss - Which keras loss to use. Should be of type LossType. Currently only used for DeepLearner
        '''
        print("Initializing DataLoader object with file: {}".format(file_name))

        self.output_folder = output_folder
        self.model_folder = model_folder

        self.batch_size = batch_size
        self.epochs = epochs
        self.alpha = learning_rate
        self.eps = epsilon

        self.architecture = architecture
        self.activation = activation
        self.optimizer = optimizer
        self.metric = metric
        self.loss = loss

        self.read_data(file_name, percent_validation)

        self.child_init()

        print("Finish Initializing DataLoader object.")

    def read_data(self, file_name, percent_validation=0.15):
        '''
            Read data from file_name, store in DataLoader class
        '''

        person1_data_numpy = np.loadtxt(file_name, delimiter=" ")
        nrows, _ = person1_data_numpy.shape

        a = 1  # a == 1 excludes the first column from the raw_data matrix (time stamp column)
        self.raw_data = person1_data_numpy[:, a:]
        self.assign_data_indices(a + 1)  # a+1 b/c will be using data from training/test sets, which won't have label column
        self.split_data(percent_validation)

    def split_data(self, percent_validation):
        '''
            Splits data into training and validation sets.
            Requires self.raw_data
        '''

        n = self.raw_data.shape[0]
        num_validation = int(percent_validation * n)
        self.test_data = self.raw_data[:num_validation, 1:]
        self.test_labels = self.raw_data[:num_validation, 0]

        self.train_data = self.raw_data[num_validation:, 1:]
        self.train_labels = self.raw_data[num_validation:, 0]
        self.m, self.n = self.train_data.shape

    def assign_data_indices(self, a):
        '''
            Requires self.raw_data.
            a is the offset of the indices. If using the complete dataset (i.e. includes timestamp and
            activity_ID) then a = 0. If excluding timestamp and activity ID, a = 2. If excluding timestamp,
            activity_ID, and heart_rate, a = 3

            Example usage:
                hand_data = train_data[:, self.index[self.BodyPart.hand]]
                hand_accel_data = hand_data[:, self.index[SensorType.accel]]
                plot(hand_accel_data)

        '''
        self.index = {}

        # These indices are relative to the raw_data matrix

        self.index["heart_rate"] = slice(2 - a)
        self.index[BodyPart.heart_rate] = slice(2 - a)

        self.index["hand"] = slice(3 - a, 19 - a, 1)
        self.index["chest"] = slice(20 - a, 36 - a, 1)
        self.index["ankle"] = slice(37 - a, 53 - a, 1)

        self.index[BodyPart.hand] = slice(3 - a, 19 - a, 1)
        self.index[BodyPart.chest] = slice(20 - a, 36 - a, 1)
        self.index[BodyPart.ankle] = slice(37 - a, 53 - a, 1)

        # These indices are offset relative to the BodyPart indices

        self.index["temp"] = slice(0)
        self.index["accel"] = slice(1, 3, 1)
        self.index["gyro"] = slice(7, 9, 1)
        self.index["magnet"] = slice(10, 12, 1)

        self.index[SensorType.temp] = slice(0)
        self.index[SensorType.accel] = slice(1, 3, 1)
        self.index[SensorType.gyro] = slice(7, 9, 1)
        self.index[SensorType.magnet] = slice(10, 12, 1)


    # ------------------------------------- SubClass Methods ------------------------------------- #

    '''
        Subclasses that inherit from DataLoader should overwrite these methods.
    '''

    def child_init(self):
        pass

    def train(self, batch_size):
        raise Exception("DataLoader does not implement self.train(). Child class must implement it.")

    def loss(self):
        raise Exception("DataLoader does not implement self.loss(). Child class must implement it.")

    def predict(self, input_data):
        raise Exception("DataLoader does not implement self.predict(). Child class must implement it.")

    def accuracy(self):
        raise Exception("DataLoader does not implement self.accuracy(). Child class must implement it.")

    def save(self):
        raise Exception("DataLoader does not implement self.save(). Child class must implement it.")

    def load(self):
        raise Exception("DataLoader does not implement self.load(). Child class must implement it.")



