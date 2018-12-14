# Zach Blum, Navjot Singh, Aristos Athens

'''
    Main file for running project.
'''

import sys
import datetime

import discriminative_models
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
    data_folder_name = './../data/'


    # makeCleanData.convert_data("./../data/")
    # raise Exception("here")

    # ---------------------------- Neural Net Model -------------------------------------
    # Create DeepLearner object, train it
    # learner = neural_net.DeepLearner(data_file_name,
    #                                     output_folder_name,
    #                                     models_folder_name,
    #                                     batch_size = 200,
    #                                     architecture = ArchitectureType.MLP_multiclass
    #                                     )
    # learner.train(epochs = 10)
    #
    # # Plot learner info
    # accuracy = learner.history.history["acc"]
    # loss = util.normalize(learner.history.history["loss"])
    #
    # time_string = str(datetime.datetime.now().isoformat(' ', 'minutes'))
    # info = learner.info_string()
    # util.plot(data = [accuracy, loss],
    #             title = "Accuracy, Loss v Epochs",
    #             x_label = "Epochs",
    #             labels = ["Training Accuracy", "Normalized Loss"],
    #             file_name = output_folder_name + "Accuracy, Loss v Time " + time_string + ".png",
    #             fig_text = info
    #             )
    # accuracy = learner.accuracy()
    # print(accuracy)

    # ----------------------- Logistic Regression model ---------------------------------
    learner = discriminative_models.DiscriminativeLearner(data_folder_name, output_folder_name, models_folder_name,
                                                          percent_validation=0.3, epsilon=25.0, learning_rate=1e-2,
                                                          use_lib=True, model='svm')  # model can be 'log_reg' or 'svm'
    # learner.tune_hyperparamter()
    # learner.train(None)
    learner.predict(None)

    # -----------------------------------------------------------------------------------
    # Use plot() from our util.py package
    # util.plot([learner.hand_accel], show=True, title="Hand Accel vs Time")
    # util.plot([learner.chest_accel, learner.chest_gyro], show=True, title="Chest Accel & Chest Gyro vs Time")
    # util.plot([learner.activity_ID], show=True, title="Activity ID v Time")


if __name__ == "__main__":
    main()
