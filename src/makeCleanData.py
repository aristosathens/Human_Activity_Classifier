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

    # prev_heart_rate = np.nan
    # # Fill in heart rate values with previous time-stamp values
    # for i in range(np.alen(person1_data_matrix_fixed)):
    #     if not np.isnan(person1_data_matrix_fixed[i, 2]):
    #         prev_heart_rate = person1_data_matrix_fixed[i, 2]
    #         continue
    #     if np.isnan(person1_data_matrix_fixed[i, 2]) and not np.isnan(prev_heart_rate):
    #         person1_data_matrix_fixed[i, 2] = prev_heart_rate

    # Remove all rows with Nan
    person1_data_matrix_fixed = person1_data_matrix_fixed[~np.any(np.isnan(person1_data_matrix_fixed), axis=1)]

    return person1_data_matrix_fixed


def convert_data(data_folder):

    all_subjects_data_matrix = None
    directory = os.fsencode(data_folder)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)

        # Check if file is data file
        if file_name.endswith(".dat"):
            print(file_name[:-4] + ".csv")

            # Check if csv version of file already exists
            if os.path.isfile(file_name[:-4] + ".csv"):
                dat = np.genfromtxt (file_name[:-4] + ".csv", delimiter=",")
            else:
                dat = read_data(data_folder + file_name)
            
            # Add data to matrix
            if all_subjects_data_matrix is None:
                all_subjects_data_matrix = dat
            else:
                all_subjects_data_matrix = np.append(all_subjects_data_matrix, dat, axis=0)

    np.random.shuffle(all_subjects_data_matrix)
    np.savetxt(data_folder + 'cleanData.csv', all_subjects_data_matrix, fmt='%.5f', delimiter=" ")

    return all_subjects_data_matrix




if __name__ == "__main__":

    # *****************************Variables to change**********************************
    # Navjot's paths - comment these two out and write your own
    output_folder_path = "./../data/"                     # CHANGE TO DESIRED LOCATION
    proto_folder_path = './../../PAMAP2_Dataset/Protocol/'  # CHANGE CORRECT PATH TO THE PROTOCAL FOLDER
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
