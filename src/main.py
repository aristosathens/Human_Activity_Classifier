# Zach Blum, Navjot Singh, Aristos Athens

'''
    Main file for running project.
'''

import os
import sys
import datetime

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
    learner.train(epochs = 200)

    # Plot learner info
    accuracy = learner.history.history["acc"]
    loss = util.normalize(learner.history.history["loss"])

    time_string = str(datetime.datetime.now().isoformat(' ', 'minutes'))
    info = learner.info_string()
    util.plot(data = [accuracy, loss],
                title = "Accuracy, Loss v Epochs",
                x_label = "Epochs",
                labels = ["Training Accuracy", "Normalized Loss"],
                file_name = output_folder_name + "Accuracy, Loss v Time " + time_string +".png",
                fig_text = info
                )
    accuracy = learner.accuracy()
    print(accuracy)


if __name__ == "__main__":

    if not hasattr(sys, 'real_prefix'):
        print("\n --- WARNING: Not running in virtual environment --- \n")

    main()