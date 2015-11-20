"""
An implementation of bagging as a wrapper for a classifier
"""
import numpy as np
import scipy

from dtree import DecisionTree
from ann import ArtificialNeuralNetwork

from nbayes import NaiveBayes
from logistic_regression import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    'dtree'                 : DecisionTree,
    'ann'                   : ArtificialNeuralNetwork,
    'nbayes'                : NaiveBayes,
    'logistic_regression'   : LogisticRegression,
}

class Bagger(object):

    def __init__(self, algorithm, iters, **params):
        """
        Boosting wrapper for a classification algorithm

        @param algorithm : Which algorithm to use
                            (dtree, ann, linear_svm, nbayes,
                            or logistic_regression)
        @param iters : How many iterations of bagging to do
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
        X_array=np.asarray(X) 
        X_data_int=X_array[:,0:X_array.shape[1]-1]
        X_data= X_data_int/1.0
        
        self.estimators_ = []

        n_samples, self.n_features_ = X_data.shape
        
        for iboost in range(self.iters):
            #classfier=CLASSIFIERS[self.algorithm]()
            classfier=DecisionTreeClassifier(max_depth=1) 
            
            rand_selection=np.random.randint(0, n_samples)
            #estimators_samples=[,:]
            estimators_samples=X_array[rand_selection,:]
            print estimators_samples.shape
            for i in range(n_samples-1):
                rand_selection=np.random.randint(0, n_samples)
                estimators_samples=np.append(estimators_samples,X_array[rand_selection], axis=0)
            #classfier.fit(estimators_samples, estimators_samples[:,-1])  
            self.estimators_.append(classfier)
            print estimators_samples.shape
        print estimators_samples
        return self


    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):

       
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

        return proba

        
