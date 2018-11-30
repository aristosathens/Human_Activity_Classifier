# Zach Blum, Navjot Singh, Aristos Athens

'''
    RegressionLearner class.
'''

from parent_class import *
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import os
import time


# ------------------------------------- SubClass ------------------------------------- #

class RegressionLearner(DataLoader):
    '''
        Inherits __init__() from DataLoader.
    '''

    def child_init(self):
        '''
            Init data specific to RegressionLearner
        '''

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        if self.use_lib and self.model == 'svm':
            self.raw_data = self.raw_data[:175000, :]  # reduce data size for svm

        import collections
        print(collections.Counter(self.raw_data[:, 0]))

        # Scale all features to make iterative algorithm more robust
        scaler = StandardScaler(copy=False)
        if self.use_lib:
            scaler.fit(self.train_data)
            scaler.transform(self.train_data)
            scaler.transform(self.test_data)
        else:
            scaler.transform(self.raw_data)

        # Dict for selecting specific IMU's with heart rate sensor
        self.feature_indices = {
                            # 'hand': [0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
                            # 'chest': [0, 19, 20, 21, 22, 26, 27, 28, 29, 30, 31],
                            # 'ankle': [0, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48],
                            # 'hand_ankle': [0, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48,
                            #                2, 3, 4, 5, 9, 10, 11, 12, 13, 14],
                            'hand_ankle_chest_HR': [0, 1, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48,  # TODO: change back
                                                 2, 3, 4, 5, 9, 10, 11, 12, 13, 14,
                                                 19, 20, 21, 22, 26, 27, 28, 29, 30, 31]
                            }
        # self.feature_indices['hand_ankle'] = self.feature_indices['hand'] + self.feature_indices['ankle'][1:]
        # self.feature_indices['hand_ankle_chest'] = self.feature_indices['hand_ankle'] + self.feature_indices['chest'][1:]
        # self.feature_indices['hand_HR'] = self.feature_indices['hand'] + [1]
        # self.feature_indices['chest_HR'] = self.feature_indices['chest'] + [1]
        # self.feature_indices['ankle_HR'] = self.feature_indices['ankle'] + [1]
        # self.feature_indices['hand_ankle_HR'] = self.feature_indices['hand_ankle'] + [1]
        # self.feature_indices['hand_ankle_chest_HR'] = self.feature_indices['hand_ankle_chest'] + [1]

        # Add intercept term (add column of 1's to x matrix) if using custom log reg function
        # scilearn already has built-in functionality
        if not self.use_lib:
            self.train_data = self.add_intercept(self.train_data)
            self.test_data = self.add_intercept(self.test_data)

        self.select_activities()

        self.log_train_data = self.train_data
        self.log_train_labels = self.train_labels
        self.log_test_data = self.test_data
        self.log_test_labels = self.test_labels

        # self.change_labels()

        self.m, self.n = np.shape(self.log_train_data)
        if self.use_lib:
            if self.model == 'svm':
                self.scilearn_model = svm.SVC(kernel='rbf', gamma='auto', shrinking=True)
            else:
                self.scilearn_model = linear_model.LogisticRegression(solver='sag', multi_class='multinomial',
                                                                      max_iter=5000)
            self.theta = None
        else:
            self.theta = np.zeros(self.n)

    def select_activities(self):
        """
        Which activities to select
        """
        bool_idxs = (self.train_labels == 1) | (self.train_labels == 2) | (self.train_labels == 3) | \
                    (self.train_labels == 4) | (self.train_labels == 5) | (self.train_labels == 6) | \
                    (self.train_labels == 7) | (self.train_labels == 24)
        bool_idxs_test = (self.test_labels == 1) | (self.test_labels == 2) | (self.test_labels == 3) | \
                         (self.test_labels == 4) | (self.test_labels == 5) | (self.test_labels == 6) | \
                         (self.test_labels == 7) | (self.test_labels == 24)

        self.log_train_data = self.train_data[bool_idxs]
        self.log_train_labels = self.train_labels[bool_idxs]
        self.log_test_data = self.test_data[bool_idxs_test]
        self.log_test_labels = self.test_labels[bool_idxs_test]

    def change_labels(self):
        """
        replace labels 1, 2, 3 with 0 and 4, 5, 6, 7, 24 with 1
        """
        nonactive_idxs = (self.log_train_labels == 1) | (self.log_train_labels == 2) | (self.log_train_labels == 3)
        active_idxs = (self.log_train_labels == 4) | (self.log_train_labels == 5) | (self.log_train_labels == 6) | \
                      (self.log_train_labels == 7) | (self.log_train_labels == 24)
        nonactive_idxs_test = (self.log_test_labels == 1) | (self.log_test_labels == 2) | (self.log_test_labels == 3)
        active_idxs_test = (self.log_test_labels == 4) | (self.log_test_labels == 5) | (self.log_test_labels == 6) | \
                           (self.log_test_labels == 7) | (self.log_test_labels == 24)
        self.log_train_labels[nonactive_idxs] = 0
        self.log_test_labels[nonactive_idxs_test] = 0
        self.log_train_labels[active_idxs] = 1
        self.log_test_labels[active_idxs_test] = 1

    def train(self):
        '''
            Train RegressionLearner
        '''
        if self.use_lib:
            print("Training model with scikitlearn {}...".format(self.scilearn_model.__class__.__name__))
            # c_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]  # for log reg
            # c_range = [0.1 1.0, 10.0, 100.0, 1000.0, 10000.0]  # for SVM

            print("Starting validation curve")
            print(time.time())
            for key in self.feature_indices.keys():
                filtered_data = self.raw_data[:, self.feature_indices[key]]
                train_scores, test_scores = validation_curve(self.scilearn_model, filtered_data[:, 1:],
                                                             filtered_data[:, 0], param_name="C", param_range=c_range,
                                                             cv=3, scoring="accuracy", n_jobs=-1)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)

                print(time.time())
                print("For key: {}".format(key))
                print("Train scores mean: {} with std: {}".format(train_scores_mean, train_scores_std))
                print("Test scores mean: {} with std: {}".format(test_scores_mean, test_scores_std))

                lw = 2
                # plt.semilogx(c_range, train_scores_mean, label="Training", lw=lw)
                # plt.fill_between(c_range, train_scores_mean - train_scores_std,
                #                  train_scores_mean + train_scores_std, alpha=0.2, lw=lw)
                plt.semilogx(c_range, test_scores_mean, label=key, lw=lw)
                # plt.fill_between(c_range, test_scores_mean - test_scores_std,
                #                  test_scores_mean + test_scores_std, alpha=0.2, lw=lw)

            plt.title("Validation Curves")
            plt.xlabel("C")
            plt.ylabel("Accuracy")
            plt.ylim(0.2, 1.0)
            plt.legend(loc="best")
            plt.savefig("./../output/mix_svm2.png")

        else:
            self.stochastic_train()
            # self.batch_train()

    def predict(self):
        '''
            Return predicted class for input_data
        '''
        # return util.sigmoid(self.theta.T.dot(input_data))
        theta_pred = None
        predictions = None
        accur = None
        if self.use_lib:
            theta_pred = (self.scilearn_model.intercept_, self.scilearn_model.coef_)
            predictions = None
            accur = self.scilearn_model.score(self.log_test_data, self.log_test_labels)
        else:
            theta_pred = self.theta
            predictions = self.h(self.log_test_data)
            accur = self.accuracy(predictions)

        print("Thetas: {}".format(theta_pred))
        print("Model accuracy: {}".format(accur))

        return predictions

    def add_intercept(self, x):
        """Add intercept to matrix x.

        Args:
            x: 2D NumPy array.

        Returns:
            New matrix same as x with 1's in the 0th column.
        """
        new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = x

        return new_x

    def accuracy(self, predictions):
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        acc_sum = 0
        for pred_i, t_label_i in zip(predictions, self.log_test_labels):
            if pred_i == t_label_i:
                acc_sum += 1

        accuracy = acc_sum / np.alen(self.log_test_labels)
        return accuracy

    def h(self, x):
        """
        :param x: theta:
        :return hypothesis. Sigmoid in this case:
        """
        return util.sigmoid(x @ self.theta)  # The hypothesis function. Sigmoid in this case

    def batch_train(self):
        """
            Trains RegressionLearner on self.train_data
        """
        print("Beginning batch grad descent training...")

        delta = np.inf
        iter = 0
        while delta > self.eps and iter < 50:

            theta_previous = np.copy(self.theta)
            for j in range(self.n):
                self.theta[j] += self.alpha * ((self.log_train_labels - self.h(self.log_train_data)) @
                                 self.log_train_data[:, j])

            delta = np.linalg.norm(theta_previous - self.theta)
            print(delta)
            iter += 1

    def stochastic_train(self):
        '''
            Trains RegressionLearner on self.train_data
        '''
        print("Beginning stochastic gradient descent training...")

        delta = np.inf
        iter = 0
        while delta > self.eps and iter < 50:

            theta_previous = np.copy(self.theta)
            for i in range(self.m):
                row = self.log_train_data[i, :]
                self.theta += self.alpha * (self.log_train_labels[i] - self.h(row)) * row
                # for j in range(self.n):
                #     self.theta[j] += self.alpha * (self.log_train_labels[i] - self.h(row)) * row[j]

            self.accuracy(self.h(self.log_test_data))
            delta = np.linalg.norm(theta_previous - self.theta)
            print("Delta: {}".format(delta))
            iter += 1
