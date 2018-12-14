# Zach Blum, Navjot Singh, Aristos Athens

'''
    DecisionTrees class.
'''

from parent_class import *
import os
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
#import time as ti
import matplotlib.pyplot as plt

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

        if self.raw_data.shape[0]>17500:
            self.raw_data = self.raw_data[:17500, :]  # reduce data size for svm
            self.labels = self.labels[:17500]

        if self.tree_train_data.shape[0]>17500:
            self.tree_train_data = self.tree_train_data[:17500, :]  # reduce data size for svm
            self.tree_train_labels = self.tree_train_labels[:17500]



       #print(self.tree_train_labels)

    def train_and_test_normal_trees(self,max_depth_val,key):
        tree_train = tree.DecisionTreeClassifier(max_depth=max_depth_val)  # took out max depth
        tree_train = tree_train.fit(self.tree_train_data, self.tree_train_labels)
        y_pred = tree_train.predict(self.tree_test_data)
        #y_pred_train = tree_train.predict(self.tree_train_data)
        #train_accuracy_val = accuracy_score(self.tree_train_labels, y_pred_train)
        #test_accuracy_val = accuracy_score(self.tree_test_labels,y_pred)
        test_accuracy_val = accuracy_score(self.tree_test_labels, y_pred)
        if key=='hand_HR':
            conf_mtrx = confusion_matrix(self.tree_test_labels,y_pred)
            print(conf_mtrx)
        #print(accuracy_val)
        return test_accuracy_val,tree_train

    def test_trees(self):
        for key in self.feature_indices.keys():
            if key == 'hand_HR' or key == 'hand_ankle_chest_HR':
                filtered_features = self.tree_train_data[:, self.feature_indices[key]]
                tree_train = tree.DecisionTreeClassifier(max_depth=15)
                tree_train = tree_train.fit(filtered_features,self.tree_train_labels)
                filtered_test_features = self.tree_test_data[:, self.feature_indices[key]]
                y_pred = tree_train.predict(filtered_test_features)
                test_accuracy_val = accuracy_score(self.tree_test_labels, y_pred)
                print('Test accuracy val for key {} is {}'.format(key,test_accuracy_val))
                if key == 'hand_HR':
                    conf_mtrx = confusion_matrix(self.tree_test_labels, y_pred)
                    print(conf_mtrx)

    def normal_trees(self):

        depth_range = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        for key in self.feature_indices.keys():
            if key=='hand_HR' or key=='hand_ankle_chest_HR':
                #print("For key: {}".format(key))
                #filtered_features = self.raw_data[:, self.feature_indices[key]]
                filtered_features = self.tree_train_data[:, self.feature_indices[key]]
                #tree_train = tree.DecisionTreeClassifier(max_depth=30)
                #tree_train = tree_train.fit(filtered_features,self.labels)
                #y_pred_train = tree_train.predict(filtered_features)
                #y_pred_test = tree_train.predict(self.tree_test_data)
                train_scores,valid_scores = validation_curve(tree.DecisionTreeClassifier(),filtered_features,self.tree_train_labels,param_name='max_depth',param_range=depth_range,cv=5,scoring="accuracy",n_jobs=-1)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                valid_scores_mean = np.mean(valid_scores, axis=1)
                valid_scores_std = np.std(valid_scores, axis=1)

                plt.figure()
                plt.title('Training and Validation Curve for {}'.format(key))
                train_line, = plt.plot(depth_range, train_scores_mean,label='Training Accuracy')
                valid_line, = plt.plot(depth_range, valid_scores_mean,label='Cross-Validation Accuracy')
                plt.xlabel('Max Tree Depth')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()
                plt.savefig('Validation_plot for normal tree with key {}'.format(key))
                #print(time.time())
                print("For key: {}".format(key))
                print("Train scores mean: {} with std: {}".format(train_scores_mean, train_scores_std))
                print("Validation scores mean: {} with std: {}".format(valid_scores_mean, valid_scores_std))

                best_depth = np.argmax(valid_scores_mean)
                print("best_depth is {} for key={}".format(depth_range[best_depth],key))

                test_accuracy_val, best_tree = self.train_and_test_normal_trees(15,key)
                print('Best tree for {} got test score = {}'.format(key,test_accuracy_val))
                #tree.export_graphviz(best_tree, out_file='best_tree_updated_{}.dot'.format(key))
    def boosted_trees(self):
        '''
        Uses Adaboost with default n_estimators = 50, learning_rate =1. Takes a lot of time, maybe try less trees
        '''

        for key in self.feature_indices.keys():
            if key == 'hand_HR' or key == 'hand_ankle_chest_HR':
                #print("For key: {}".format(key))
                filtered_features = self.raw_data[:, self.feature_indices[key]]
                for max_depth_val in range(1,11):
                    train_scores,test_scores = validation_curve(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=max_depth_val)),filtered_features,self.labels,param_name='n_estimators',param_range=[50,100,250,500],cv=5,scoring="accuracy",n_jobs=-1)
                    train_scores_mean = np.mean(train_scores, axis=1)
                    train_scores_std = np.std(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)
                    test_scores_std = np.std(test_scores, axis=1)

                    #print(time.time())
                    print("For key: {}".format(key))
                    print("Train scores mean for boost: {} with std: {} for max_depth_val {}".format(train_scores_mean, train_scores_std,max_depth_val))
                    print("Test scores mean for boost: {} with std: {} for max_depth_val {}".format(test_scores_mean, test_scores_std,max_depth_val))

    def test_boosted(self):
        for key in self.feature_indices.keys():
            if key == 'hand_HR' or key == 'hand_ankle_chest_HR':
                filtered_features = self.tree_train_data[:, self.feature_indices[key]]
                if key == 'hand_HR':
                    tree_train = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10), n_estimators=500)
                    tree_train = tree_train.fit(filtered_features,self.tree_train_labels)
                    filtered_test_features = self.tree_test_data[:, self.feature_indices[key]]
                    y_pred = tree_train.predict(filtered_test_features)
                    test_accuracy_val = accuracy_score(self.tree_test_labels, y_pred)
                    print('Test accuracy val for key {} is {}'.format(key,test_accuracy_val))
                    conf_mtrx = confusion_matrix(self.tree_test_labels, y_pred)
                    print(conf_mtrx)
                else:
                    tree_train = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=9), n_estimators=250)
                    tree_train = tree_train.fit(filtered_features, self.tree_train_labels)
                    filtered_test_features = self.tree_test_data[:, self.feature_indices[key]]
                    y_pred = tree_train.predict(filtered_test_features)
                    test_accuracy_val = accuracy_score(self.tree_test_labels, y_pred)
                    print('Test accuracy val for key {} is {}'.format(key, test_accuracy_val))

    def random_forest(self):
        '''
        Uses default values, except max_depth. Needs to be deeper
        '''
        depth_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

        for key in self.feature_indices.keys():
            if key == 'hand_HR' or key == 'hand_ankle_chest_HR':
                #print("For key: {}".format(key))
                filtered_features = self.raw_data[:, self.feature_indices[key]]
                #tree_train = tree.DecisionTreeClassifier(max_depth=30)
                #tree_train = tree_train.fit(filtered_features,self.labels)
                #y_pred_train = tree_train.predict(filtered_features)
                #y_pred_test = tree_train.predict(self.tree_test_data)
                train_scores,valid_scores = validation_curve(RandomForestClassifier(n_estimators=100),filtered_features,self.labels,param_name='max_depth',param_range=depth_range,cv=5,scoring="accuracy",n_jobs=-1)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                valid_scores_mean = np.mean(valid_scores, axis=1)
                valid_scores_std = np.std(valid_scores, axis=1)

                plt.figure()
                plt.title('Training and Validation Curve for {} Using Random Forest'.format(key))
                train_line, = plt.plot(depth_range, train_scores_mean, label='Training Accuracy')
                valid_line, = plt.plot(depth_range, valid_scores_mean, label='Cross-Validation Accuracy')
                plt.xlabel('Max Tree Depth')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

                #print(time.time())
                print("For key: {}".format(key))
                print("Train scores mean for RF: {} with std: {}".format(train_scores_mean, train_scores_std))
                print("Valid scores mean for RF: {} with std: {}".format(valid_scores_mean, valid_scores_std))


    def test_RF(self):
        for key in self.feature_indices.keys():
            if key == 'hand_HR' or key == 'hand_ankle_chest_HR':
                filtered_features = self.tree_train_data[:, self.feature_indices[key]]
                tree_train = RandomForestClassifier(n_estimators=100,max_depth=20)
                tree_train = tree_train.fit(filtered_features,self.tree_train_labels)
                filtered_test_features = self.tree_test_data[:, self.feature_indices[key]]
                y_pred = tree_train.predict(filtered_test_features)
                test_accuracy_val = accuracy_score(self.tree_test_labels, y_pred)
                print('Test accuracy val for key {} is {} for RF'.format(key,test_accuracy_val))
                if key == 'hand_HR':
                    conf_mtrx = confusion_matrix(self.tree_test_labels, y_pred)
                    print(conf_mtrx)


    def train(self):
        '''
            Train DecisionTreeLearner
        '''

        #maybe try with different max depths??? and try with boosting!!
        tree_train = tree.DecisionTreeClassifier(max_depth=2)#took out max depth
        tree_train = tree_train.fit(self.tree_train_data,self.tree_train_labels)

        n_nodes = tree_train.tree_.node_count
        children_left = tree_train.tree_.children_left
        children_right = tree_train.tree_.children_right
        feature = tree_train.tree_.feature
        threshold = tree_train.tree_.threshold
        tree.export_graphviz(tree_train,out_file = 'tree_class2.dot')
