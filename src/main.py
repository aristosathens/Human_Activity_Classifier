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
    data_file_name = "./../data/subject101.dat"
    # data_file_name = './PAMAP2_Dataset/Protocol/subject101.dat'

    # Create DeepLearner object, train it
    learner = neural_net.DeepLearner(data_file_name,
                                        output_folder_name,
                                        models_folder_name,
                                        batch_size = 200,
                                        architecture = ArchitectureType.MLP_multiclass
                                        )
    learner.train(epochs = 100)
    accuracy = learner.history.history["acc"]
    loss = util.normalize(learner.history.history["loss"])
    util.plot(data = [accuracy, loss],
                title = "Accuracy, Loss v Time",
                x_label = "Epochs",
                labels = ["Training Accuracy", "Normalized Loss"],
                file_name = output_folder_name + "Accuracy, Loss v Time.png"
                )
    accuracy = learner.accuracy()
    print(accuracy)


    # Create object, train it
    # learner = logistic_regression.RegressionLearner(data_file_name, output_folder_name, models_folder_name, learning_rate=1e-2)
    # learner.train(batch_size=500)
    # Use plot() from our util.py package
    # util.plot([learner.hand_accel], show=True, title="Hand Accel vs Time")
    # util.plot([learner.chest_accel, learner.chest_gyro], show=True, title="Chest Accel & Chest Gyro vs Time")
    # util.plot([learner.activity_ID], show=True, title="Activity ID v Time")


if __name__ == "__main__":

    if not hasattr(sys, 'real_prefix'):
        print("\n --- WARNING: Not running in virtual environment --- \n")

    main()