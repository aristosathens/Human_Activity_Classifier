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
        self.theta = np.random.rand(self.n, self.k)

    def predict(self, input_data):
        '''
            Return predicted class for input_data
        '''
        return self.h(input_data)

    def train(self, batch_size=50):
        '''
            Train RegressionLearner 
        '''
        self.stochastic_train(batch_size)

    def h(self, x):
        '''
            The hypothesis function. Sigmoid in this case
        '''
        return util.sigmoid(self.theta.T.dot(x))

    def stochastic_train(self, batch_size=50):
        '''
            Trains RegressionLearner on self.train_data
        '''
        print("Beginning training...")
        train_data = self.train_data

        delta = 1e25

        while (delta > self.eps):

            theta_previous = np.copy(self.theta)
            for _ in range(batch_size):
                i = np.random.randint(0, self.m)
                row = self.raw_data[i, :]

                for j in range(self.n):
                    self.theta[j] += self.alpha * (self.activity_ID[i] - self.h(row)) * row[j]


            delta = np.abs(np.linalg.norm(self.theta - theta_previous))
            print(delta)

