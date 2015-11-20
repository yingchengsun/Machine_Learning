"""
An implementation of boosting as a wrapper for a classifier
"""
import numpy as np
import scipy

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
from sklearn.metrics import roc_curve, auc

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
        self.clf=object

        if self.algorithm=='ann':
            self.gamma = params.pop('gamma')


    def fit(self, X, y):
        #classfier=CLASSIFIERS[self.algorithm]()
        #classfier.fit(self, X, y)
        X_array=np.asarray(X) 
        X_data_int=X_array[:,0:X_array.shape[1]-1]
        X_data= X_data_int/1.0

        #Xx, yy = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0) 

        self.clf = DecisionTreeClassifier(max_depth=1)   
        
 
        #self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=20)
        
        self.clf.fit(X_data, y)
        #scores2 = cross_val_score(self.clf, X_data, y) 

        from sklearn import linear_model
      
        #train perceptrons
        perceptron_A = linear_model.Perceptron(n_iter=200)
        perceptron_A.fit(Xa, ya)


        
        # Then, can I initiate an AdaBoostClassifier with existing perceptrons? 
        
        ada_real = AdaBoostClassifier(
            base_estimator='Perceptron', # [perceptron_A, perceptron_B]
            learning_rate=learning_rate,
            n_estimators=2,
            algorithm="SAMME.R")
     

    def predict(self, X,y):
        X_array=np.asarray(X) 
        X_test=X_array[:,0:X_array.shape[1]-1]
        print self.clf.predict(X_test[:1, :]) 
        #scores = cross_val_score(self.clf, X_test, y)  
        #print  scores.mean()  
        
    def predict_proba(self, X):
        pass
