# Zach Blum, Navjot Singh, Aristos Athens

'''
    Main file for running project.
'''

import logistic_regression
import util

# ------------------------------------- Main ------------------------------------- #

def main():
    '''
        Create Learner object, train it, get predictions.
    '''

    output_folder_name = "./../output/"
    data_file_name = "./../data/subject101.dat"
    # file_name = './PAMAP2_Dataset/Protocol/subject101.dat'

    # Create object, train it
    learner = logistic_regression.RegressionLearner(data_file_name, output_folder_name, learning_rate=1e-2)
    learner.train(batch_size=500)

    # Use plot() from our util.py package
    util.plot([learner.hand_accel], show=True, title="Hand Accel vs Time")
    util.plot([learner.chest_accel, learner.chest_gyro], show=True, title="Chest Accel & Chest Gyro vs Time")
    util.plot([learner.activity_ID], show=True, title="Activity ID v Time")


if __name__ == "__main__":
    main()