# Zach Blum, Navjot Singh, Aristos Athens

'''
    RegressionLearner class.
'''

from parent_class import *


# ------------------------------------- SubClass ------------------------------------- #

class RegressionLearner(DataLoader):
    '''
        Inherits __init__() from DataLoader.
    '''

    def child_init(self):
        '''
            Init data specific to RegressionLearner
        '''

        # Select specific features (e.g. Hand IMU and heart rate only)
        # Note: time column has been taken out so column indices shift over from data documentation
        # self.train_data = self.train_data[:, (2, 4, 5, 6)]
        # self.test_data = self.test_data[:, (2, 4, 5, 6)]

        # Add intercept term (add column of 1's to x mattrix)
        self.train_data = self.add_intercept(self.train_data)
        self.test_data = self.add_intercept(self.test_data)

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

        # replace labels 1, 2, 3 with 0 and 4, 5, 6, 7, 24 with 1
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

        self.m, self.n = np.shape(self.log_train_data)
        # self.theta = np.random.rand(self.n)  # before: (n, k)
        self.theta = np.zeros(self.n)

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

    def predict(self):
        '''
            Return predicted class for input_data
        '''
        # return util.sigmoid(self.theta.T.dot(input_data))
        predictions = self.h(self.log_test_data)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        acc_sum = 0
        for pred_i, t_label_i in zip(predictions, self.log_test_labels):
            if pred_i == t_label_i:
                acc_sum += 1

        accuracy = acc_sum / np.alen(self.log_test_labels)
        print(accuracy)
        print(self.theta)
        return accuracy

    def h(self, x):
        """
        :param x:
        :return hypothesis. Sigmoid in this case:
        """
        return util.sigmoid(x @ self.theta)  # The hypothesis function. Sigmoid in this case

    def train(self, batch_size=50):
        '''
            Train RegressionLearner 
        '''
        # self.stochastic_train()
        self.batch_train()

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

            delta = np.linalg.norm(self.theta - theta_previous) ** 2
            print(np.linalg.norm(self.theta))
            iter += 1

    def stochastic_train(self):
        '''
            Trains RegressionLearner on self.train_data
        '''
        print("Beginning stochastic gradient descent training...")

        delta = np.inf
        while delta > self.eps:

            theta_previous = np.copy(self.theta)
            for i in range(self.m):
                row = self.log_train_data[i, :]
                for j in range(self.n):
                    self.theta[j] += self.alpha * (self.log_train_labels[i] - self.h(row)) * row[j]

            delta = np.linalg.norm(self.theta - theta_previous)**2
            print(delta)
