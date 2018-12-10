# Zach Blum, Navjot Singh, Aristos Athens

'''
    DecisionTrees class.
'''

from parent_class import *
import os
from sklearn import tree
import matplotlib.pyplot as plt

class DecisionTreeLearner(DataLoader):
    '''
        Inherits __init__() from DataLoader.
    '''
    def child_init(self):
        '''
            Init data specific to DecisionTreeLearner
        '''

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        self.tree_train_data = self.train_data
        self.tree_train_labels = self.train_labels
        self.tree_test_data = self.test_data
        self.tree_test_labels = self.test_labels

       #print(self.tree_train_labels)
    def train(self):
        '''
            Train DecisionTreeLearner
        '''

        #maybe try with different max depths??? and try with boosting!!
        tree_train = tree.DecisionTreeClassifier(max_depth=5)#took out max depth
        tree_train = tree_train.fit(self.tree_train_data,self.tree_train_labels)

        n_nodes = tree_train.tree_.node_count
        children_left = tree_train.tree_.children_left
        children_right = tree_train.tree_.children_right
        feature = tree_train.tree_.feature
        threshold = tree_train.tree_.threshold
        tree.export_graphviz(tree_train,out_file = 'tree_class.dot')
