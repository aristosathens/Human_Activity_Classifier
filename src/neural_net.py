# Aristos Athens

'''
    DeepLearner class.
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from parent_class import *


# from keras.models import load_model


# ------------------------------------- SubClass ------------------------------------- #

class DeepLearner(DataLoader):
    '''
        Inherits __init__() from DataLoader.
    '''
    
    # Define constants for this class
    default_num_epochs = 10


    def child_init(self):
        '''
            Initialize the neural net.
            Using the Sequential class allows us to easily stack layers sequentially.
            Using the Functional class allows us to use more complex architectures.

            First layer needs input_shape argument. In input_shape, the batch dimension is not included.
            To use batches, give argument batch_size.
            Example:
                If we give input_shape = n and batch_size = 50, every input to the model must be size 50 x n.
        '''
        print("Initializing DeepLearner subclass.")

        # Check if architecture type is set
        if self.architecture == None:
            print("Warning: Architecture type not set. Using default: ", ArchitectureStrings[ArchitectureType.default])
            self.architecture = ArchitectureType.default

        # Call the function to load the model with the correct architecture
        if self.architecture == ArchitectureType.MLP_multiclass:
            self.setup_MLP_multiclass()
        else:
            raise Exception("No function corresponding to architecture: ", ArchitectureStrings[self.architecture])

        print("Finished initializing DeepLearner object.")


    def set_learning_params(self):
        '''
            Set the learning parameters for the Keras model.
        '''
        # If any param isn't set, use default
        if self.optimizer == None:
            print("Warning: Optimizer type not set. Using default: ", OptimizerStrings[OptimizerType.default])
            self.optimizer = OptimizerType.default

        if self.loss == None:
            print("Warning: Loss type not set. Using default: ", LossStrings[LossType.default])
            self.loss = LossType.default

        if self.metric == None:
            print("Warning: Accuracy type not set. Using default: ", AccuracyStrings[AccuracyType.default])
            self.metric = AccuracyType.default

        # Set the training rules for the Keras model
        self.model.compile(optimizer = OptimizerStrings[self.optimizer],
                            loss = LossStrings[self.loss],
                            metrics = [AccuracyStrings[self.metric]]
                            )


    def predict(self, input_data):
        '''
            Predict labels for the input data.
        '''
        return self.model.predict(input_data)

    def accuracy(self):
        '''
            Return validation loss and accuracy.
        '''
        labels = keras.utils.to_categorical(self.test_labels, num_classes = self.k)
        return self.model.evaluate(self.test_data, labels)



    def train(self,
                x = None,
                labels = None,
                batch_size = None,
                epochs = None,
                ):
        '''
            Fit the neural net parameters to the data.

            Note that data (x), labels, and batch_size are set in __init__().
            If x != None and labels != None, it will use these only for training. Will not overrwrite self.data.
            If batch_size != None, it will overwrite self.batch_size. Same with epochs.
        '''
        print("Beginning neural net training.")

        # Set batch size
        if batch_size != None:
            self.batch_size = batch_size

        # Set num epochs
        if epochs != None:
            self.epochs = epochs
        elif self.epochs == None:
            print("Warning: Num epochs not set. Using default: ", DeepLearner.default_num_epochs)
            self.epochs = DeepLearner.default_num_epochs

        # Check if passed in data is valid
        if (x is None) != (labels is None):
            raise Exception("Error in DeepLearner.train(). Both x and labels must be None, or neither must be None.")
        elif (x != None) and (x.shape[0] != len(labels)):
            raise Exception("Error in DeepLearner.train(). Data and labels must contain same number of datapoints.")
        
        # Put data into correct keras format
        if x == None:
            x = self.train_data
            labels = self.train_labels
        labels = keras.utils.to_categorical(labels, num_classes = self.k)

        # Fit model to data
        self.history = self.model.fit(x, labels, epochs = self.epochs, batch_size = self.batch_size)


    def train_on_batch(self, x, labels):
        '''
            Train on only the passed in data
        '''
        if x.shape[0] != len(labels):
            raise Exception("Error in DeepLearner.train_on_batch(). Data and labels must contain same number of datapoints.")

        labels = keras.utils.to_categorical(labels, num_classes = self.k)
        model.train_on_batch(x, labels)


    def save_model(self, name = None, full_path = None, delete = False):
        '''
            Save a keras model so it can be used later.
        '''
        # Check that valid strings passed in
        if (name is None) and (full_path is None):
            raise Exception("Error in DeepLearner.save(). Must provide either file name or full save path.")

        # Save model as an HDF5 file
        if full_path != None:
            print("Saving model to: ", full_path)
            self.model.save(full_path)
        else:
            print("Saving model to: ", self.model_folder + name)
            self.model.save(self.model_folder + name)

        # Delete model, if requested
        if delete == True:
            del self.model
            self.model = None


    def load_model(self, name = None, full_path = None):
        '''
            Load a keras model we previously saved.
        '''
        # Check that valid strings passed in
        if (name is None) and (full_path is None):
            raise Exception("Error in DeepLearner.load(). Must provide either file name or full save path.")

        # Load model
        if full_path != None:
            print("Loading model from: ", full_path)
            self.model = keras.models.load_model(full_path)
        else:
            print("Loading model from: ", self.model_folder + name)
            self.model = keras.models.load_model(self.model_folder + name)


    # ------------------------------------- Architectures ------------------------------------- #

    def setup_MLP_multiclass(self):
        '''
            Multilayer Perceptron (MLP) for multi-class softmax classification:
        '''
        print("Setting up MLP multiclass nueral net architecture.")

        self.model = Sequential()
        self.model.add(Dense(256, activation='relu', input_dim = self.n))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.k, activation='softmax'))

        self.loss = LossType.categorical_crossentropy
        self.optimizer = OptimizerType.SGD
        self.metric = AccuracyType.accuracy
        self.set_learning_params()