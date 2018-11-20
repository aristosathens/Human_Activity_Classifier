# Zach Blum, Navjot Singh, Aristos Athens

'''
    Main file for running project.
'''

import os
import sys

import logistic_regression
import neural_net
import util

from enum_types import *

# ------------------------------------- Main ------------------------------------- #

def main():
    '''
        Create Learner object, train it, get predictions.
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_folder_name = "./../output/"
    models_folder_name = "./../models/"
    data_file_name = './../data/cleanData.csv'

    # ---------------------------- Neural Net Model -------------------------------------
    # Create DeepLearner object, train it
    # learner = neural_net.DeepLearner(data_file_name,
    #                                     output_folder_name,
    #                                     models_folder_name,
    #                                     batch_size = 200,
    #                                     architecture = ArchitectureType.MLP_multiclass
    #                                     )
    # learner.train(epochs = 100)
    # accuracy = learner.accuracy()
    # print(accuracy)

    # ----------------------- Logistic Regression model ---------------------------------
    learner = logistic_regression.RegressionLearner(data_file_name, output_folder_name, models_folder_name,
                                                    epsilon=25.0, learning_rate=1e-2)
    learner.train(batch_size=500)
    learner.predict()
    # Use plot() from our util.py package
    # util.plot([learner.hand_accel], show=True, title="Hand Accel vs Time")
    # util.plot([learner.chest_accel, learner.chest_gyro], show=True, title="Chest Accel & Chest Gyro vs Time")
    # util.plot([learner.activity_ID], show=True, title="Activity ID v Time")


if __name__ == "__main__":

    if not hasattr(sys, 'real_prefix'):
        print("\n --- WARNING: Not running in virtual environment --- \n")

    main()
