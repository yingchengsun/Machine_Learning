"""
The Artificial Neural Network
"""
import numpy as np
import scipy

class ArtificialNeuralNetwork(object):


    def __init__(self, gamma, layer_sizes, num_hidden, epsilon=None, max_iters=None):
        """
        Construct an artificial neural network classifier

        @param gamma : weight decay coefficient
        @param layer_sizes:  Number of hidden layers
        @param num_hidden:  Number of hidden units in each hidden layer
        @param epsilon : cutoff for gradient descent
                         (need at least one of [epsilon, max_iters])
        @param max_iters : maximum number of iterations to run
                            gradient descent for
                            (need at least one of [epsilon, max_iters])
        """
        self.gamma=gamma
        self.layer_sizes=layer_sizes
        self.num_hidden=num_hidden
        self.epsilon=epsilon
        self.max_iters=max_iters
        self.syn0=np.array([])
        self.syn1=np.array([])
        self.learning_rate=0.01

    def fit(self, X, y, schema, sample_weight=None):
        """ Fit a neural network of layer_sizes * num_hidden hidden units using X, y """
        X_array=np.asarray(X) 

        X_data_int=X_array[:,0:X_array.shape[1]-1]
        y_data=X_array[:,X_array.shape[1]-1:X_array.shape[1]]
        X_data= X_data_int/1.0
        #max_iterations=len(X_data)*100
        #max_iterations=100
        
        """Data Standardization"""
        for i in range(len(X_data[0])):
            if schema.is_nominal(i):
                nominal_mapping=np.array(schema.nominal_values[i])
                for m in range(len(X_data[:,i])):
                    for n in range(len(nominal_mapping)):
                        if np.float64((nominal_mapping[n]))==X_data[:,i][m]:
                            X_data[:,i][m]=n+1                         
            else:
                X_data[:,i]= (X_data[:,i]-min(X_data[:,i]))/float(max(X_data[:,i])-min(X_data[:,i])) 
                           
        for i in range(len(y_data)):
            if y_data[i]==-1:
                y_data[i]=y_data[i]+1
                
        np.random.seed(1)
        
        # randomly initialize our weights with mean 0 from (-1,1)
        self.syn0 = (2*np.random.random((X_data.shape[1],self.num_hidden)) - 1)/10.0
        self.syn1 = (2*np.random.random((self.num_hidden,1)) - 1)/10.0
        #self.syn0 = (2*np.random.random((X_data.shape[1],1)) - 1)/10.0
        iteration = 0
        l0 = X_data
        while True:            # Feed forward through layers 0, 1, and 2
            
            l1 = 1/(1+np.exp(-np.dot(l0,self.syn0)))

            l2 = 1/(1+np.exp(-np.dot(l1,self.syn1)))

            #l1_delta=l0.T.dot((np.dot(l0,self.syn0)-y_data))
            #self.syn0+=-l1_delta*self.learning_rate
            
            # how much did we miss the target value?
            l2_error = y_data - l2
                       
            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            l2_delta = l2_error*(l2*(1-l2))
            
            # how much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(self.syn1.T)
            
            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            l1_delta = l1_error *(l1*(1-l1))
            
            #self.syn1 =self.syn1+ (l1.T.dot(l2_delta))*self.learning_rate
            #self.syn0 =self.syn0+ (l0.T.dot(l1_delta))*self.learning_rate
            
            self.syn1 += (l1.T.dot(l2_delta))*self.learning_rate-self.gamma*self.learning_rate*self.syn1
            self.syn0 += (l0.T.dot(l1_delta))*self.learning_rate-self.gamma*self.learning_rate*self.syn0
            iteration += 1 
            '''
            if max_iterations!=0 and iteration >= max_iterations: 
                break
            elif max_iterations==0 and np.mean(np.abs(l1_delta)) < 1e-3 and np.mean(np.abs(l2_delta)) < 1e-3:
                break
            '''
            if self.max_iters!=0 and iteration >= self.max_iters: 
                break
            elif self.max_iters==0 and np.mean(np.abs(l2_error)) < 1e-3 and np.mean(np.abs(l2_delta)) < 1e-3:
                break

    def predict(self, X):
        """ Predict -1/1 output """
        X_array=np.asarray(X) 
        X_test=X_array[:,0:X_array.shape[1]-1]
        l0_test = X_test
        l1_test = 1/(1+np.exp(-np.dot(l0_test,self.syn0)))
        l2_test = 1/(1+np.exp(-np.dot(l1_test,self.syn1)))
        #predictions=np.ones((len(l2_test),1))
        
        for j in range(len(l2_test)):
            if l2_test[j]>0.5:
                l2_test[j]=1.0
            else:
                l2_test[j]=-1.0
        print type(l2_test)
        return l2_test

    def predict_proba(self, X):
        """ Predict probabilistic output """
        X_array=np.asarray(X) 
        X_test=X_array[:,0:X_array.shape[1]-1]
        l0_test = X_test
        l1_test = 1/(1+np.exp(-np.dot(l0_test,self.syn0)))
        l2_test = 1/(1+np.exp(-np.dot(l1_test,self.syn1)))
        l2_test_no_column=l2_test.reshape((len(l2_test),))
        return l2_test_no_column
    
