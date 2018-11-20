import numpy as np
import pandas as pd
import os


def read_data(data_file_name):
    """
        Read data from file_name, store in DataLoader
    """

    person1_data = pd.read_table(data_file_name)
    person1_data_numpy = person1_data.values
    nrows, _ = person1_data_numpy.shape
    ncols = 54

    # convert the string of data for each row into array
    person1_data_matrix = np.zeros((nrows, ncols))

    # person1_data_list = list(list())
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
    return person1_data_matrix_fixed


if __name__ == "__main__":

    # *****************************Variables to change**********************************
    # Navjot's paths - comment these two out and write your own
    output_folder_path = "./../data/"                     # CHANGE TO DESIRED LOCATION
    proto_folder_path = '../../PAMAP2_Dataset/Protocol/'  # CHANGE CORRECT PATH TO THE PROTOCAL FOLDER
    NUM_FILES_TO_READ = 9
    # **********************************************************************************

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    all_subjects_data_matrix = None
    for i in range(1, NUM_FILES_TO_READ+1):
        data_file_name = proto_folder_path + 'subject10{}.dat'.format(i)
        if all_subjects_data_matrix is None:
            all_subjects_data_matrix = read_data(data_file_name)
        else:
            all_subjects_data_matrix = np.append(all_subjects_data_matrix, read_data(data_file_name), axis=0)
        print("Files read: {}".format(i))
        print("Current data matrix size: {}".format(all_subjects_data_matrix.shape))

    print("Shuffling data matrix and writing to file...")
    np.random.shuffle(all_subjects_data_matrix)
    np.savetxt(output_folder_path + 'cleanData.csv', all_subjects_data_matrix, fmt='%.5f', delimiter=" ")
    print("All done dawg")
