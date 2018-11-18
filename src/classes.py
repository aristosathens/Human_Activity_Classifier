# Zach Blum, Navjot Singh, Aristos Athens

'''
    Defines enumerated types, DataLoader parent class and various subclasses like RegressionLearner, etc.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util
from enum_types import *


# ------------------------------------- Class ------------------------------------- #

class DataLoader():
    '''
        Use to load data and store it as an object.
        m - number of data points
        n - number of input features
        k - number of classes
    '''

    def __init__(self,
                file_name,
                percent_validation=0.15,
                learning_rate = 0.1,
                epsilon = 1e-2
                ):
        '''
            Initialize DataLoader
        '''
        print("Initializing DataLoader object with file: {}".format(file_name))

        self.alpha = learning_rate
        self.eps = epsilon

        self.read_data(file_name, percent_validation)

        self.child_init()


    def read_data(self, file_name, percent_validation=0.15):
        '''
            Read data from file_name, store in DataLoader
        '''
        person1_data_file = file_name
        person1_data = pd.read_table(person1_data_file)
        person1_data_numpy = person1_data.values
        nrows,_ = person1_data_numpy.shape
        ncols = 54

        print(person1_data_numpy.shape)

        #convert the string of data for each row into array
        person1_data_matrix = np.zeros((nrows, ncols))

        #person1_data_list = list(list())
        # for row_ind in range(nrows):
        for i, row in enumerate(person1_data_numpy):
            row_list = row[0].split()
            row_array = np.asarray(row_list)
            row_array_floats = row_array.astype(np.float)
            person1_data_matrix[i, :] = row_array_floats


        # discard data that includes activityID = 0
        activity_ID = person1_data_matrix[:, 1]
        good_data_count = 0
        for i in range(nrows):
            if activity_ID[i] != 0:
                good_data_count += 1

        person1_data_matrix_fixed = np.zeros((good_data_count, ncols))
        count = 0
        for i in range(nrows):
            if activity_ID[i] != 0:
                person1_data_matrix_fixed[count, :] = person1_data_matrix[i, :]
                count += 1

        # Remove all rows with Nan
        person1_data_matrix_fixed = person1_data_matrix_fixed[~np.any(np.isnan(person1_data_matrix_fixed), axis=1)]

        #extract data
        self.timestamp = person1_data_matrix_fixed[:,0]
        self.activity_ID = person1_data_matrix_fixed[:,1]

        '''
        self.heart_rate = person1_data_matrix_fixed[:,2]

        IMU_hand = person1_data_matrix_fixed[:,3:19]
        self.hand_temp = IMU_hand[:,0]
        self.hand_accel = IMU_hand[:,1:3] #only use the +-16g sensor, as noted in the readme
        self.hand_gyro = IMU_hand[:,7:9]
        self.hand_magnet = IMU_hand[:,10:12]

        IMU_chest = person1_data_matrix_fixed[:, 20:36]
        self.chest_temp = IMU_chest[:, 0]
        self.chest_accel = IMU_chest[:, 1:3]  # only use the +-16g sensor, as noted in the readme
        self.chest_gyro = IMU_chest[:, 7:9]
        self.chest_magnet = IMU_chest[:, 10:12]

        IMU_ankle = person1_data_matrix_fixed[:,37:53]
        self.ankle_temp = IMU_ankle[:, 0]
        self.ankle_accel = IMU_ankle[:, 1:3]  # only use the +-16g sensor, as noted in the readme
        self.ankle_gyro = IMU_ankle[:, 7:9]
        self.ankle_magnet = IMU_ankle[:, 10:12]
        '''

        # TODO: Split data into train/validation set
        a = 2 # a == 2 excludes the first 2 columns from the raw_data matrix
        self.raw_data = person1_data_matrix_fixed[a:, :]
        self.assign_data_indices(a)
        self.m, self.n = self.raw_data.shape

        self.labels = np.unique(activity_ID)
        self.k = int(np.max(self.labels))

        self.split_data(person1_data_matrix_fixed, percent_validation)

    def split_data(self, data, percent_validation):
        '''
            Splits data into training and validation sets.
            Requires self.raw_data
        '''
        n = data.shape[0]
        num_validation = int(percent_validation * n)
        self.validation_data = self.raw_data[:num_validation, :]
        self.training_data = self.raw_data[num_validation:, :]

    def assign_data_indices(self, a):
        '''
            Requires self.raw_data.
            a is the offset of the indices. If using the complete dataset (i.e. includes timestamp and
            activity_ID) then a = 0. If excluding timestamp and activity ID, a = 2. If excluding timestamp,
            activity_ID, and heart_rate, a = 3

            Example usage:
                hand_data = train_data[:, self.index[self.BodyPart.hand]]
                hand_accel_data = hand_data[self.index[SensorType.accel]]
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

    def train(self, batch_size):
        raise Exception("DataLoader does not implement self.train(). Child class must implement it.")

    def predict(self):
        raise Exception("DataLoader does not implement self.predict(). Child class must implement it.")

    def child_init(self):
        pass



# ------------------------------------- SubClass ------------------------------------- #

class RegressionLearner(DataLoader):
    '''
        Inherits __init__() from DataLoader.
    '''

    def child_init(self):
        '''
            Init data specific to RegressionLearner
        '''
        self.theta = np.random.rand(self.n, self.k)

    def predict(self, input_data):
        '''
            Return predicted class for input_data
        '''
        return self.h(input_data)

    def train(self, batch_size=50):
        '''
            Train RegressionLearner 
        '''
        self.stochastic_train(batch_size)

    def h(self, x):
        '''
            The hypothesis function. Sigmoid in this case
        '''
        return util.sigmoid(self.theta.T.dot(x))

    def stochastic_train(self, batch_size=50):
        '''
            Trains RegressionLearner on self.train_data
        '''
        print("Beginning training...")
        train_data = self.training_data

        delta = 1e25

        while (delta > self.eps):

            theta_previous = np.copy(self.theta)
            for _ in range(batch_size):
                i = np.random.randint(0, self.m)
                row = self.raw_data[i, :]

                for j in range(self.n):
                    self.theta[j] += self.alpha * (self.activity_ID[i] - self.h(row)) * row[j]


            delta = np.abs(np.linalg.norm(self.theta - theta_previous))
            print(delta)

