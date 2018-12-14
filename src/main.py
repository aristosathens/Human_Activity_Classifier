# Zach Blum, Navjot Singh, Aristos Athens

"""
    Main file for running project.
"""

import datetime

import discriminative_models
import neural_net
import util
import decision_trees

import makeCleanData
from enum_types import *

# ------------------------------------- Main ------------------------------------- #


def main():
    """
        Create Learner object, train it, get predictions.
    """

    output_folder_name = "./../output/"
    models_folder_name = "./../models/"
    data_folder_name = './../data/'

    # makeCleanData.convert_data("./../data/")
    # raise Exception("here")

    # ---------------------------- Neural Net Model -------------------------------------
    # Create DeepLearner object, train it
    learner = neural_net.DeepLearner(data_folder_name,
                                     output_folder_name,
                                     models_folder_name,
                                     batch_size=200,
                                     architecture=ArchitectureType.MLP_multiclass
                                     )
    learner.train(epochs=10)

    # Plot learner info
    accuracy = learner.history.history["acc"]
    loss = util.normalize(learner.history.history["loss"])

    time_string = str(datetime.datetime.now().isoformat(' ', 'minutes'))
    info = learner.info_string()
    util.plot(data=[accuracy, loss],
              title="Accuracy, Loss v Epochs",
              x_label="Epochs",
              labels=["Training Accuracy", "Normalized Loss"],
              file_name=output_folder_name + "Accuracy, Loss v Time " + time_string + ".png",
              fig_text=info
              )
    accuracy = learner.accuracy()
    print(accuracy)

    # ----------------------- Logistic Regression model ---------------------------------
    learner = discriminative_models.DiscriminativeLearner(data_folder_name, output_folder_name, models_folder_name,
                                                          percent_validation=0.3, epsilon=25.0, learning_rate=1e-2,
                                                          use_lib=True, model='svm')  # model can be 'log_reg' or 'svm'
    learner.tune_hyperparamter()
    learner.train(None)
    learner.predict(None)

    # ----------------------- Decision Trees ---------------------------------------------
    learner = decision_trees.DecisionTreeLearner(data_folder_name, output_folder_name, models_folder_name,
                                                 percent_validation=0.15)
    learner.normal_trees()
    learner.test_trees()
    learner.random_forest()
    learner.test_RF()
    learner.boosted_trees()
    learner.test_boosted()


if __name__ == "__main__":
    main()
