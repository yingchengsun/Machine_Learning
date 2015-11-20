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
        
        '''
        n_samples = x.get_shape()[0]
        for i in xrange(0, self.n_models):
            self.clf.append(self.get_classifier())
            indices = random.sample(xrange(0, n_samples), random.randrange(n_samples / 2, n_samples))
            self.clf[i].fit(x[indices, :], y[indices, :])
            
         '''   
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
       '''
        estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)
         features = sample_without_replacement(n_features,
                                                  max_features,
                                                  random_state=random_state)
         estimators.append(estimator)
        estimators_samples.append(samples)
        estimators_features.append(features)
        
        predictions = estimator.predict(X[:, features])
        
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba
        '''
        
