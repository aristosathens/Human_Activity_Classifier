# Zach Blum, Navjot Singh, Aristos Athens

"""
Discriminative learner class that uses logistic regression and svm models to train and predict on data from the
PAMAP2 Dataset.
"""


from parent_class import *
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from joblib import dump, load
import os
import time


# ------------------------------------- SubClass ------------------------------------- #

class DiscriminativeLearner(DataLoader):
    def __init__(self,
                 file_name,
                 output_folder,
                 model_folder,
                 percent_validation=0.15,
                 batch_size=None,
                 learning_rate=0.1,
                 epsilon=1e-2,
                 epochs=None,

                 architecture=None,
                 activation=None,
                 optimizer=None,
                 metric=None,
                 loss=None,
                 use_lib=True,
                 model='svm'
                 ):
        '''
            Init data specific to RegressionLearner
        '''
        super().__init__(file_name, output_folder, model_folder, percent_validation, batch_size, learning_rate, epsilon,
                         epochs, architecture, activation, optimizer, metric, loss)

        self.use_lib = use_lib
        self.model_name = model
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        if self.use_lib and self.model_name == 'svm' and self.train_data.shape[0] > 175000:
            self.train_data = self.train_data[:175000, :]  # reduce data size for svm
            self.train_labels = self.train_labels[:175000]

        # Scale all features to make iterative algorithm more robust
        if not self.use_lib:
            my_scaler = StandardScaler(copy=False)
            my_scaler.fit(self.train_data)
            my_scaler.transform(self.train_data)
            my_scaler.transform(self.test_data)

        # Add intercept term (add column of 1's to x matrix) if using custom log reg function
        # scilearn already has built-in functionality
        if not self.use_lib:
            self.train_data = self.add_intercept(self.train_data)
            self.test_data = self.add_intercept(self.test_data)

        self.m, self.n = np.shape(self.train_data)

        if self.use_lib:
            if self.model_name == 'svm':
                self.estimator = Pipeline([('scl', StandardScaler()),
                                           (self.model_name, svm.SVC(kernel='rbf', gamma='auto', shrinking=False))])
            else:
                self.estimator = Pipeline([('scl', StandardScaler()),
                                           (self.model_name, linear_model.LogisticRegression(solver='sag',
                                                                                             multi_class='multinomial',
                                                                                             max_iter=5000))])
            self.theta = None
        else:
            self.theta = np.zeros(self.n)

    def train(self, batch_size):
        print(time.time())
        a = 100000
        self.raw_data = self.raw_data[:, self.feature_indices['hand_HR']]
        self.train_data = self.raw_data[:a, :]  # reduce data size for svm
        self.train_labels = self.labels[:a]

        self.test_data = self.raw_data[a:, :]
        self.test_labels = self.labels[a:]
        print("train/test: {} {}".format(self.train_labels.shape[0]/self.raw_data.shape[0],
                                         self.test_labels.shape[0] / self.raw_data.shape[0]))

        my_scaler = StandardScaler(copy=False)
        my_scaler.fit(self.train_data)
        my_scaler.transform(self.train_data)
        my_scaler.transform(self.test_data)

        sklearn_svm_model = svm.SVC(C=10, kernel='rbf', gamma='auto', shrinking=False)
        sklearn_svm_model.fit(self.train_data, self.train_labels)
        print("done fitting at {}".format(time.time()))
        print("Score: {}".format(sklearn_svm_model.score(self.test_data, self.test_labels)))
        print("Num of support vecs: {}".format(sklearn_svm_model.n_support_))
        dump(sklearn_svm_model, self.model_folder + 'svm_c10_rbf')
        print(time.time())

    def tune_hyperparamter(self):
        '''
            Train DiscriminativeLearner
        '''
        if self.use_lib:
            print("Training model with scikitlearn {}...")  # .format(self.scilearn_model.__class__.__name__))

            if self.model_name == 'svm':
                c_range = [.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # for SVM
            else:
                c_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10]  # for log reg

            print("Number of training points {}".format(self.train_data.shape[0]))
            print("{} with c_range: {}".format(self.model_name, c_range))
            for key in self.feature_indices.keys():
                print("For key: {}".format(key))

                filtered_features_data = self.train_data[:, self.feature_indices[key]]

                train_scores, test_scores = validation_curve(self.estimator, filtered_features_data,
                                                             self.train_labels, param_name=self.model_name+"__C",
                                                             param_range=c_range, cv=5, scoring="accuracy", n_jobs=-1)

                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)

                print(time.time())
                print("For key: {}".format(key))
                print("Train scores mean: {} with std: {}".format(train_scores_mean, train_scores_std))
                print("Validation scores mean: {} with std: {}".format(test_scores_mean, test_scores_std))

        else:
            self.stochastic_train()
            # self.batch_train()

    def predict(self, input_data):
        '''
            Return predicted class for input_data
        '''
        # return util.sigmoid(self.theta.T.dot(input_data))
        print("Test data len: {} {}".format(float(self.test_labels.shape[0]) / self.raw_data.shape[0], self.test_labels.shape[0]))
        if self.use_lib:
            self.test_data = self.test_data[:, self.feature_indices['hand_HR']]
            init_time = time.time()
            scilearn_model = load(self.model_folder + 'svm_c10_rbf')
            predictions = scilearn_model.predict(self.test_data)
            accur = scilearn_model.score(self.test_data, self.test_labels)
            print("Predictions made in {} secs".format(time.time() - init_time))
        else:
            predictions = self.h(self.test_data)
            accur = self.accuracy(predictions)

        print("Model accuracy: {}".format(accur))

        return predictions

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

        self.train_data = self.train_data[bool_idxs]
        self.train_labels = self.train_labels[bool_idxs]
        self.test_data = self.test_data[bool_idxs_test]
        self.test_labels = self.test_labels[bool_idxs_test]

    def change_labels(self):
        """
        replace labels 1, 2, 3 with 0 and 4, 5, 6, 7, 24 with 1
        Used for binary classification
        """
        nonactive_idxs = (self.train_labels == 1) | (self.train_labels == 2) | (self.train_labels == 3)
        active_idxs = (self.train_labels == 4) | (self.train_labels == 5) | (self.train_labels == 6) | \
                      (self.train_labels == 7) | (self.train_labels == 24)
        nonactive_idxs_test = (self.test_labels == 1) | (self.test_labels == 2) | (self.test_labels == 3)
        active_idxs_test = (self.test_labels == 4) | (self.test_labels == 5) | (self.test_labels == 6) | \
                           (self.test_labels == 7) | (self.test_labels == 24)
        self.train_labels[nonactive_idxs] = 0
        self.test_labels[nonactive_idxs_test] = 0
        self.train_labels[active_idxs] = 1
        self.test_labels[active_idxs_test] = 1

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
        if not self.use_lib:
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0

        acc_sum = 0
        for pred_i, t_label_i in zip(predictions, self.test_labels):
            if pred_i == t_label_i:
                acc_sum += 1

        accuracy = acc_sum / np.alen(self.test_labels)
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
                self.theta[j] += self.alpha * ((self.train_labels - self.h(self.train_data)) @
                                               self.train_data[:, j])

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
                row = self.train_data[i, :]
                self.theta += self.alpha * (self.train_labels[i] - self.h(row)) * row
                # for j in range(self.n):
                #     self.theta[j] += self.alpha * (self.log_train_labels[i] - self.h(row)) * row[j]

            self.accuracy(self.h(self.test_data))
            delta = np.linalg.norm(theta_previous - self.theta)
            print("Delta: {}".format(delta))
            iter += 1
