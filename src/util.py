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

def normalize(data):
    '''
        Normalizes data so that max magnitue is 1
    '''
    return data / np.max(np.abs(data))

def plot(data,
        title = None,
        x_label = None,
        y_label = None,
        labels = None,
        fig_text = None,
        show = True,
        file_name = None,
        ):
    '''
        Takes a list of nX2 array data and plots it.
        If show is True, display plot.
        If file_name not None, save plot to filename. file_name must end in .jpg, .png, etc.
    '''
    for i, array in enumerate(data):
        plt.plot(array)

    if labels != None:
        plt.legend(labels)

    if title != None:
        plt.title(title)

    if x_label != None:
        plt.xlabel(x_label)

    if y_label != None:
        plt.ylabel(y_label)

    if fig_text != None:
        plt.text(-0.5, 0, fig_text)

    if file_name != None:
        plt.savefig(file_name)

    if show == True:
        plt.show()