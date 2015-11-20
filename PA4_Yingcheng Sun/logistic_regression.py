"""
The Logistic Regression Classifier
"""
import numpy as np
import scipy


class LogisticRegression(object):


    def __init__(self, lamb=0):
        """
        Constructs a logistic regression classifier

        @param lambda : Regularisation constant parameter
        """
        self.lamb=lamb

    def fit(self, X, y,schema):
        """ Fit a logistic regression of weights and lambda """
        X_array=np.asarray(X) 

        X_data_int=X_array[:,0:X_array.shape[1]-1]
        y_data=X_array[:,X_array.shape[1]-1:X_array.shape[1]]
        X_data= X_data_int/1.0

       
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

        #self.W  = np.ones((X_data.shape[1]+1, 1))/1.0
        self.W =(2*np.random.random((X_data.shape[1]+1,1)) - 1)/10.0
        """append 1 to the X data, then you can just use the w array to avoid specify a intercept term b"""
        X_data=np.append(X_data,np.ones((X_array.shape[0],1)), axis=1)
        
        """Tuning"""
        k=5
        pos_X=X_array[X_array[:,-1] == 1]
        neg_X=X_array[X_array[:,-1] == -1]
        
        total_pos = float(sum(y==1))
        total_neg = float(sum(y==-1))
        
        n_pos_fold = int(np.ceil(total_pos/float(k))) 
        n_neg_fold = int(np.ceil(total_neg/float(k)))
   
        folds=[]
        for i in range(0,k): #n_folds folds
            pos_fold = pos_X[n_pos_fold*i:n_pos_fold*i+n_pos_fold,:]
            neg_fold = neg_X[n_neg_fold*i:n_neg_fold*i+n_neg_fold,:]
            pos_fold=np.append(pos_fold, neg_fold, axis=0)     
            folds.append(pos_fold)
            
        selection=np.random.randint(0, k)    
        test_array = np.asarray(folds.pop(selection))
        train = []
        for fold in folds: 
            train.extend(fold)
        train_array=np.asarray(train) 
        

        train_array=train_array[:,0:train_array.shape[1]-1]
        train_y_data=train_array[:,train_array.shape[1]-1:train_array.shape[1]]
        train_array=np.append(train_array,np.ones((train_array.shape[0],1)), axis=1)
        
        test_array=test_array[:,0:test_array.shape[1]-1]
        test_y_data=test_array[:,test_array.shape[1]-1:test_array.shape[1]]
        test_array=np.append(test_array,np.ones((test_array.shape[0],1)), axis=1)
        
        for i in range(len(train_y_data)):
            if train_y_data[i]==-1:
                train_y_data[i]=y_data[i]+1
        
        for i in range(len(test_y_data)):
            if test_y_data[i]==-1:
                test_y_data[i]=test_y_data[i]+1
                  
        lamb=[0, 0.001, 0.01, 0.1, 1, 10, 100]
        accuracy=0
        for l in lamb:
            WW =(2*np.random.random((train_array.shape[1],1)) - 1)/10.0
            m=train_array.shape[0]
            while True:
                output = 1/(1+np.exp(-np.dot(train_array,WW)))
                error = train_y_data - output
                two_norm=WW        
                delta= np.dot(train_array.T,error)/m
                WW=WW+l*two_norm+delta
                if np.mean(np.abs(delta)) < 1e-2:
                    break
 
            preds = np.zeros(test_array[0])
            for i in range((test_array[0])):
                ex=test_array[i]
                R = np.append(ex[:],[1])
                R=R.reshape(1,len(R))
                cc=np.dot(R,WW)
                if (cc>0):
                    preds[i]=1.0
                else:
                    preds[i]=-1.0
            diff=preds-test_y_data
            print sum(diff==0)/test_y_data[0]
            if (sum(diff==0)/test_y_data[0])>accuracy:
                accuracy=sum(diff==0)/test_y_data[0]
                self.lamb=l
        
        """Traing Part"""
        m=X_data.shape[0]
        while True:
         
            output = 1/(1+np.exp(-np.dot(X_data,self.W)))
            error = y_data - output
            two_norm=self.W        
            delta= np.dot(X_data.T,error)/m

            self.W=self.W+0.001*two_norm+delta
            
            if np.mean(np.abs(delta)) < 1e-2:
                break

    
        
    def predict(self, X,schema):
        """ Fit a logistic regression of weights and lambda """
        X_array=np.asarray(X) 

        X_data_int=X_array[:,0:X_array.shape[1]-1]
        y_data=X_array[:,X_array.shape[1]-1:X_array.shape[1]]
        X_data= X_data_int/1.0
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
                
        m= len(X)
        preds = np.zeros(m)
        for i in range((m)):
            ex=X_data[i]
            R = np.append(ex[:],[1])
            R=R.reshape(1,len(R))
            cc=np.dot(R,self.W)
            #print cc
            if (cc>0):
                preds[i]=1.0
            else:
                preds[i]=-1.0
           
        return preds


    def predict_proba(self, X):
        m= len(X)
        preds = np.zeros(m)
        for i, ex in enumerate(X):
            ex=np.asarray(ex) 
            R = np.append(ex[0:-1],[1])
            cc=np.dot(R,self.W)
            preds[i]=cc
        return preds