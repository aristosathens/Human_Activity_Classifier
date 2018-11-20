# Zach Blum, Navjot Singh, Aristos Athens

'''
    Main file for running project.
'''

import sys
import datetime

import logistic_regression
import neural_net
import util

import makeCleanData

from enum_types import *

# ------------------------------------- Main ------------------------------------- #

def main():
    '''
        Create Learner object, train it, get predictions.
    '''

    output_folder_name = "./../output/"
    models_folder_name = "./../models/"
    # data_file_name = './../data/cleanData.csv'
    data_file_name = './../data/'


    # makeCleanData.convert_data("./../data/")
    # raise Exception("here")

    # ---------------------------- Neural Net Model -------------------------------------
    # Create DeepLearner object, train it
    learner = neural_net.DeepLearner(data_file_name,
                                        output_folder_name,
                                        models_folder_name,
                                        batch_size = 200,
                                        architecture = ArchitectureType.MLP_multiclass
                                        )
    learner.train(epochs = 10)

    # Plot learner info
    accuracy = learner.history.history["acc"]
    loss = util.normalize(learner.history.history["loss"])

    time_string = str(datetime.datetime.now().isoformat(' ', 'minutes'))
    info = learner.info_string()
    util.plot(data = [accuracy, loss],
                title = "Accuracy, Loss v Epochs",
                x_label = "Epochs",
                labels = ["Training Accuracy", "Normalized Loss"],
                file_name = output_folder_name + "Accuracy, Loss v Time " + time_string + ".png",
                fig_text = info
                )
    accuracy = learner.accuracy()
    print(accuracy)


    # ----------------------- Logistic Regression model ---------------------------------
    learner = logistic_regression.RegressionLearner(data_file_name, output_folder_name, models_folder_name,
                                                    epsilon=25.0, learning_rate=1e-2)
    learner.train(batch_size=500)
    learner.predict()


if __name__ == "__main__":

    if not hasattr(sys, 'real_prefix'):
        print("\n --- WARNING: Not running in virtual environment --- \n")

    main()
