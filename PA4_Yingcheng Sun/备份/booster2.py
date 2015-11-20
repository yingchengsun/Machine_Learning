"""
An implementation of boosting as a wrapper for a classifier
"""
import numpy as np
import scipy
import random

from dtree import DecisionTree
from ann import ArtificialNeuralNetwork
from nbayes import NaiveBayes
from logistic_regression import LogisticRegression
from hashlib import algorithms
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model


CLASSIFIERS = {
    'dtree'                 : DecisionTree,
    'ann'                   : ArtificialNeuralNetwork,
    'nbayes'                : NaiveBayes,
    'logistic_regression'   : LogisticRegression,
}

class Booster(object):

    def __init__(self, algorithm, iters, **params):
        """
        Boosting wrapper for a classification algorithm

        @param algorithm : Which algorithm to use
                            (dtree, ann, linear_svm, nbayes,
                            or logistic_regression)
        @param iters : How many iterations of boosting to do
        @param params : Parameters for the classification algorithm
        """
        self.algorithm=algorithm
        self.iters=iters
        if self.iters <= 0:
            raise ValueError("the number of iterations  must be greater than zero")
        if self.algorithm=='dtree':
            self.depth = params.pop('depth')
        if self.algorithm=='ann':
            self.gamma = params.pop('gamma')

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.iters, dtype=np.float)    
        self.clf=object
        

    def fit(self, X, y):
        """Build a boosted classifier from the training set (X, y).
            Returns self.
        """
        X_array=np.asarray(X) 
        X_data_int=X_array[:,0:X_array.shape[1]-1]
        X_data= X_data_int/1.0
        
        p=1
        number_of_p=int(X_data.shape[0]*p)
        l = range(0,X_data.shape[0])
        randlist = random.sample(l, number_of_p)  
        
        for i in randlist:
            y[i]=y[i]*(-1)

        sample_weight2 = np.empty(X_data.shape[0], dtype=np.float)
        sample_weight2[:] = 1. / X_data.shape[0]
        self.estimators_ = []
        self.clf = AdaBoostClassifier(linear_model.Perceptron(),n_estimators=30,algorithm='SAMME')
        #self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=18,algorithm='SAMME')
        #self.clf = BaggingClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=self.iters)
        #self.clf = linear_model.Perceptron()
        ##self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=self.iters)
        self.clf.fit(X_data, y)
        '''
        self.estimator_errors_ = np.ones(self.iters, dtype=np.float)
        
        #classfier=CLASSIFIERS[self.algorithm]()
                
        for iboost in range(self.iters):
            # Boosting step
            classfier=DecisionTreeClassifier(max_depth=1)   
            sample_weight3=[]
            sample_weight3=sample_weight2
            classfier.fit(X_data, y,sample_weight3)
            y_predict = classfier.predict(X_data)
            
            # Instances incorrectly classified
            incorrect = y_predict != y

            # Error fraction
            
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight2, axis=0))
            #estimator_error = np.sum(np.average(incorrect, weights=sample_weight2, axis=0))
            
            # Stop if classification is perfect
            if estimator_error <= 0:
                estimator_weight = 1.
                estimator_error = 0.
                
            # Stop if the error is at least as bad as random guessing
            if estimator_error >= 1. - (1. / 2):
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
                break

            #estimator_weight = 1./2*np.log((1. - estimator_error) / estimator_error)
            estimator_weight = np.log((1. - estimator_error) / estimator_error)
            # Only boost the weights if I will fit again
            if not iboost == self.iters - 1:
                # Only boost positive weights
                sample_weight2 *= np.exp(estimator_weight * incorrect * ((sample_weight2 > 0) | (estimator_weight < 0)))

            # Early termination
            if sample_weight2 is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight2)
            

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.iters - 1:
                # Normalize
                sample_weight2 /= sample_weight_sum
                
            self.estimators_.append(classfier)

        return self 
        '''
    def predict(self, X):
        
        X_array=np.asarray(X) 
        X_test=X_array[:,0:X_array.shape[1]-1]
        '''
        classes = np.unique(X_array[:,X_array.shape[1]-1:X_array.shape[1]])
        classes2 = classes [:, np.newaxis]
    
        pred = sum((estimator.predict(X_test) == classes2).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))
        #set_printoptions(threshold='nan')

        return classes.take(np.argmax(pred, axis=1), axis=0).T
        '''
        
        return self.clf.predict(X_test)

    
    def predict_proba(self, X):
        '''
        X_array=np.asarray(X) 
        X_test=X_array[:,0:X_array.shape[1]-1]
        return self.clf.predict_proba(X_test)
        
        proba = sum(estimator.predict_proba(X_test) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))
       
        
        proba /= self.estimator_weights_.sum()
        
        proba = np.exp(1. / proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer
        #print proba
        return proba
        '''
        pass