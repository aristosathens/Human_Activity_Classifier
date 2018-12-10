# Zach Blum, Navjot Singh, Aristos Athens

'''
    Defines enumerated types, DataLoader parent class and various subclasses like RegressionLearner, etc.
'''

import os

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
                 use_lib=False,
                 model='log_reg'
                 ):
        '''
            Initialize DataLoader

            file_name - path to data file
            output_folder - path to output directory
            model_folder - path to directory for saving/loading models
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
        self.use_lib = use_lib
        self.model = model

        self.architecture = architecture
        self.activation = activation
        self.optimizer = optimizer
        self.metric = metric
        self.loss = loss

        self.init_data(file_name, percent_validation)
        self.child_init()

        print("Finished Initializing DataLoader object.")


    def init_data(self, file_name, percent_validation=0.15):
        '''
            Read data from file_name, store in DataLoader
        '''
        # Get data from .dat and/or .csv files
        person1_data_matrix_fixed = self.convert_data(file_name)

        # Remove all rows with Nan
        person1_data_matrix_fixed = person1_data_matrix_fixed[~np.any(np.isnan(person1_data_matrix_fixed), axis=1)]

        #extract data
        self.timestamp = person1_data_matrix_fixed[:, 0]
        self.activity_ID = person1_data_matrix_fixed[:, 1]

        a = 2  # a == 2 excludes the first 2 columns from the raw_data matrix
        self.raw_data = person1_data_matrix_fixed[:, a:]
        # self.clean_data()
        self.assign_data_indices(a)
        self.m, self.n = self.raw_data.shape

        self.labels = self.activity_ID
        self.k = int(np.max(np.unique(self.activity_ID))) + 1

        self.split_data(person1_data_matrix_fixed, percent_validation)

        import collections
        print("Label frequency: ", collections.Counter(self.labels))


    def convert_data(self, data_folder):
        '''
            Convert data from .dat to .csv (or use .csv if available)
        '''
        all_subjects_data_matrix = None
        directory = os.fsencode(data_folder)
        data_file_path = data_folder + "cleanData.csv"

        # check if cleanData.csv file already exists, otherwise create it
        if os.path.isfile(data_file_path):
            print("Reading from {}".format(data_file_path))
            all_subjects_data_matrix = np.loadtxt(data_file_path)
        else:
            # Iterate through all files in data directory
            for file in os.listdir(directory):
                file_name = os.fsdecode(file)

                # Check if file is data file
                if file_name.endswith(".dat"):
                    print(file_name[:-4] + ".csv")

                    # Check if csv version of file already exists
                    if os.path.isfile(file_name[:-4] + ".csv"):
                        dat = np.genfromtxt(file_name[:-4] + ".csv", delimiter=" ")
                    else:
                        dat = self.read_data(data_folder + file_name)

                    # Add data to matrix
                    if all_subjects_data_matrix is None:
                        all_subjects_data_matrix = dat
                    else:
                        all_subjects_data_matrix = np.append(all_subjects_data_matrix, dat, axis=0)

            np.random.shuffle(all_subjects_data_matrix)
            np.savetxt(data_file_path, all_subjects_data_matrix, fmt='%.5f', delimiter=" ")

        return all_subjects_data_matrix


    def read_data(self, data_file_name):
        """
            Read data from file_name, store in DataLoader
        """
        # Read raw data
        person1_data = pd.read_table(data_file_name)
        person1_data_numpy = person1_data.values
        nrows, _ = person1_data_numpy.shape
        ncols = 54

        # Convert the string of data for each row into array
        person1_data_matrix = np.zeros((nrows, ncols))

        # Person1_data_list = list(list())
        for i, row in enumerate(person1_data_numpy):
            row_list = row[0].split()
            row_array = np.asarray(row_list)
            row_array_floats = row_array.astype(np.float)
            person1_data_matrix[i, :] = row_array_floats

        # Discard data that includes activityID = 0
        activity_ID = person1_data_matrix[:, 1]
        good_data_count = 0
        for i in range(nrows):
            if activity_ID[i] != 0:
                good_data_count += 1

        # Remove data with label 0
        person1_data_matrix_fixed = np.zeros((good_data_count, ncols))
        count = 0
        for i in range(nrows):
            if activity_ID[i] != 0:
                person1_data_matrix_fixed[count, :] = person1_data_matrix[i, :]
                count += 1

        prev_heart_rate = np.nan
        # Fill in heart rate values with previous time-stamp values
        for i in range(np.alen(person1_data_matrix_fixed)):
            if not np.isnan(person1_data_matrix_fixed[i, 2]):
                prev_heart_rate = person1_data_matrix_fixed[i, 2]
                continue
            if np.isnan(person1_data_matrix_fixed[i, 2]) and not np.isnan(prev_heart_rate):
                person1_data_matrix_fixed[i, 2] = prev_heart_rate

        # Remove all rows with Nan
        person1_data_matrix_fixed = person1_data_matrix_fixed[~np.any(np.isnan(person1_data_matrix_fixed), axis=1)]

        return person1_data_matrix_fixed


    def split_data(self, data, percent_validation):
        '''
            Splits data into training and validation sets.
            Requires self.raw_data
        '''
        n = data.shape[0]
        num_validation = int(percent_validation * n)

        self.test_data = self.raw_data[:num_validation, :]
        self.test_labels = self.labels[:num_validation]

        self.train_data = self.raw_data[num_validation:, :]
        self.train_labels = self.labels[num_validation:]


    def clean_data(self):
        '''
            Does data preprocessing.
            Requires self.raw_data.

            -- > Currently does not work as expected < --
        '''
        # 0 center the data
        self.raw_data -= np.mean(self.raw_data, axis=0)

        # Scale the data
        self.raw_data /= np.std(self.raw_data, axis=0)


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
        # Dict for selecting specific IMU's with heart rate sensor
        self.feature_indices = {
            'hand': [0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
            'chest': [0, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31],
            'ankle': [0, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48],
            'hand_ankle': [0, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48,
                           2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
            'hand_ankle_chest': [0, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48,
                                 2, 3, 4, 5, 9, 10, 11, 12, 13, 14,
                                 19, 20, 21, 22, 26, 27, 28, 29, 30, 31]
        }
        self.feature_indices['hand_ankle'] = self.feature_indices['hand'] + self.feature_indices['ankle'][1:]
        self.feature_indices['hand_ankle_chest'] = self.feature_indices['hand_ankle'] + self.feature_indices['chest'][1:]
        self.feature_indices['hand_HR'] = self.feature_indices['hand'] + [1]
        self.feature_indices['chest_HR'] = self.feature_indices['chest'] + [1]
        self.feature_indices['ankle_HR'] = self.feature_indices['ankle'] + [1]
        self.feature_indices['hand_ankle_HR'] = self.feature_indices['hand_ankle'] + [1]
        self.feature_indices['hand_ankle_chest_HR'] = self.feature_indices['hand_ankle_chest'] + [1]

        # # Indices for this dict are relative to the raw_data matrix
        # self.index = {}
        # self.index["heart_rate"] = slice(2 - a)
        # self.index[BodyPart.heart_rate] = slice(2 - a)
        #
        # self.index["hand"] = slice(3 - a, 19 - a, 1)
        # self.index["chest"] = slice(20 - a, 36 - a, 1)
        # self.index["ankle"] = slice(37 - a, 53 - a, 1)
        #
        # self.index[BodyPart.hand] = slice(3 - a, 19 - a, 1)
        # self.index[BodyPart.chest] = slice(20 - a, 36 - a, 1)
        # self.index[BodyPart.ankle] = slice(37 - a, 53 - a, 1)
        #
        # # These indices are offset relative to the BodyPart indices
        #
        # self.index["temp"] = slice(0)
        # self.index["accel"] = slice(1, 3, 1)
        # self.index["gyro"] = slice(7, 9, 1)
        # self.index["magnet"] = slice(10, 12, 1)
        #
        # self.index[SensorType.temp] = slice(0)
        # self.index[SensorType.accel] = slice(1, 3, 1)
        # self.index[SensorType.gyro] = slice(7, 9, 1)
        # self.index[SensorType.magnet] = slice(10, 12, 1)


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