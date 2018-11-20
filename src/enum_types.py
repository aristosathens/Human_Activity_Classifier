# Zach Blum, Navjot Singh, Aristos Athens

'''
    Types for use in other packages.
    Put these in this separate file to avoid circular dependencies.

    Redefine enumerated types for various Keras options. This helps us catch errors at "compile" time. If we just
    used strings for the options, errors would only get caught at run-time, potentially wasting tons of human time.
'''

import enum


# ------------------------------------- Types ------------------------------------- #

'''
    Types specific to the datasets.
'''
class SensorType(enum.Enum): 
    temp = 0
    accel = 1
    gyro = 2
    magnet = 3

class BodyPart(enum.Enum):
    hand = 0
    chest = 1
    ankle = 2
    heart_rate = 3


'''
    Types specific to the DeepLearner class.
'''
class ArchitectureType(enum.Enum):
    MLP_multiclass = 0

    default = MLP_multiclass


'''
    Types specific to the keras+tensorflow framework.
'''
class ActivationType(enum.Enum):
    step = 0
    linear = 1
    relu = 2
    leaky_relu = 3
    softmax = 4
    sigmoid = 5
    tanh = 6

    default = relu

class OptimizerType(enum.Enum):
    '''
        See https://keras.io/optimizers/
    '''
    SGD = 0
    RMSProp = 1
    Adagrad = 2
    Adadelta = 3
    Adam = 4
    Adamax = 5
    Nadam = 6

    default = SGD

class AccuracyType(enum.Enum):
    '''
        See https://keras.io/metrics/
    '''
    accuracy = 0
    binary_accuracy = 1
    categorical_accuracy = 2
    sparse_categorical_accuracy = 3
    top_k_categorical_accuracy = 4
    sparse_top_k_categorical_accuracy = 5

    default = accuracy

class LossType(enum.Enum):
    '''
        See https://keras.io/losses/
    '''
    mean_squared_error = 0
    mean_absolute_error = 1
    mean_absolute_percentage_error = 2
    mean_squared_logarithmic_error = 3
    squared_hinge = 4
    hinge = 5
    categorical_hinge = 6
    logcosh = 7

    #Tensor with one scalar loss entry per sample.
    categorical_crossentropy = 8
    sparse_categorical_crossentropy = 9
    binary_crossentropy = 10
    kullback_leibler_divergence = 11
    poisson = 12
    cosine_proximity = 13

    default = categorical_crossentropy


# ------------------------------------- Dictionaries ------------------------------------- #

'''
    Use these if you want to convert enumType --> string
'''
ArchitectureStrings = {}
ArchitectureStrings[ArchitectureType.MLP_multiclass] = "MLP_multiclass"

OptimizerStrings = {}
OptimizerStrings[OptimizerType] = "SGD"
OptimizerStrings[OptimizerType.SGD] = "SGD"
OptimizerStrings[OptimizerType.RMSProp] = "RMSProp"
OptimizerStrings[OptimizerType.Adagrad] = "Adagrad"
OptimizerStrings[OptimizerType.Adadelta] = "Adadelta"
OptimizerStrings[OptimizerType.Adam] = "Adam"
OptimizerStrings[OptimizerType.Adamax] = "Adamax"
OptimizerStrings[OptimizerType.Nadam] = "Nadam"

AccuracyStrings = {}
AccuracyStrings[AccuracyType.accuracy] = "accuracy"
AccuracyStrings[AccuracyType.binary_accuracy] = "binary_accuracy"
AccuracyStrings[AccuracyType.categorical_accuracy] = "categorical_accuracy"
AccuracyStrings[AccuracyType.sparse_categorical_accuracy] = "sparse_categorical_accuracy"
AccuracyStrings[AccuracyType.top_k_categorical_accuracy] = "top_k_categorical_accuracy"
AccuracyStrings[AccuracyType.sparse_top_k_categorical_accuracy] = "sparse_top_k_categorical_accuracy"

LossStrings = {}
LossStrings[LossType.mean_squared_error] = "mean_squared_error"
LossStrings[LossType.mean_absolute_error] = "mean_absolute_error"
LossStrings[LossType.mean_absolute_percentage_error] = "mean_absolute_percentage_error"
LossStrings[LossType.mean_squared_logarithmic_error] = "mean_squared_logarithmic_error"
LossStrings[LossType.squared_hinge] = "squared_hinge"
LossStrings[LossType.hinge] = "hinge"
LossStrings[LossType.categorical_hinge] = "categorical_hinge"
LossStrings[LossType.logcosh] = "logcosh"
LossStrings[LossType.categorical_crossentropy] = "categorical_crossentropy"
LossStrings[LossType.sparse_categorical_crossentropy] = "sparse_categorical_crossentropy"
LossStrings[LossType.binary_crossentropy] = "binary_crossentropy"
LossStrings[LossType.kullback_leibler_divergence] = "kullback_leibler_divergence"
LossStrings[LossType.poisson] = "poisson"
LossStrings[LossType.cosine_proximity] = "cosine_proximity"

# Master dictionary for all strings
FeatureDictionary = {}
FeatureDictionary.update(ArchitectureStrings)
FeatureDictionary.update(OptimizerStrings)
FeatureDictionary.update(AccuracyStrings)
FeatureDictionary.update(LossStrings) 