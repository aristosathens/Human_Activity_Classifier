# Zach Blum, Navjot Singh, Aristos Athens

'''
    Utility functions, including basic activation and loss functions.
'''

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------- Activation ------------------------------------- #

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    if x > 0:
        return 1
    else:
        return 0

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def leaky_relu(x, alpha=0.1):
    if x > 0:
        return x
    else:
        return alpha * x

def softmax(x, k):
    return np.exp(x[k]) / np.sum([np.exp(x[i]) for i in range(len(x))])


# ------------------------------------- Loss ------------------------------------- #


# ------------------------------------- Other ------------------------------------- #

def plot(data,
        title = None,
        show = False,
        file_name = None,
        ):
    '''
        Takes a list of nX2 array data and plots it.
        If show is True, display plot.
        If file_name not None, save plot to filename. file_name must end in .jpg, .png, etc.
    '''
    for i, array in enumerate(data):
        plt.plot(array)

    if file_name != None:
        plt.savefig(file_name)

    if show == True:
        plt.show()