"""
The Naive Bayes Classifier
"""
import numpy as np
import scipy

class NaiveBayes(object):


    def __init__(self, alpha=0,m=0):
        """
        Constructs a Naive Bayes classifier

        @param m : Smoothing parameter (0 for no smoothing)
        """
        self.m=m;
        self.Pxi_pos=[]
        self.Pxi_neg=[]
        
    def extract_counts(self, L):
        """
        Take a 1D numpy array as input and return a dict mapping values to counts
        """
        uniques = set(list(L))
        counts = dict((u, np.sum(L==u)) for u in uniques)
        return counts 

    def fit(self, X, y,schema):
               
        """
        Returns log ((P(c=1) / P(c=0)) * prod_i P(x_i | c=1) / P(x_i | c=0))
        using additive smoothing
        
        Input
            :X - numpy.array with shape (num_points, num_features)
                 num_features must be the same as data used to fit model
        """
        
        X_array=np.asarray(X) 
        #X_data_int=X_array[:,0:X_array.shape[1]-1]
        y_data=X_array[:,X_array.shape[1]-1:X_array.shape[1]]
        #X_data= X_data_int/1.0
        total_pos = float(sum(y==1))
        total_neg = float(sum(y==-1))

        total = total_pos + total_neg

        self.pos_prior = total_pos / total
        self.neg_prior = total_neg / total
        
        pos_X=X_array[X_array[:,-1] == 1]
        neg_X=X_array[X_array[:,-1] == -1]

        
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
        '''          
        m_value=[0, 0.001, 0.01, 0.1, 1, 10, 100]
        accuracy=0
        for l in m_value:
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
                self.m=l
        '''
              
        """Traing Part"""
        for i in range(pos_X.shape[1]-1):
            if schema.is_nominal(i):
                p=1.0/len(schema.nominal_values[i])
                pos_poss_Xi=dict((int(v), (pos_X[pos_X[:,i]==int(v)].shape[0]+self.m*p)/(total_pos+self.m)) for v in schema.nominal_values[i])
                self.Pxi_pos.append(pos_poss_Xi)
                
                neg_poss_Xi=dict((int(v), (neg_X[neg_X[:,i]==int(v)].shape[0]+self.m*p)/(total_neg+self.m)) for v in schema.nominal_values[i])
                self.Pxi_neg.append(neg_poss_Xi)
            else:
                pos_poss_Xi=dict()
                neg_poss_Xi=dict()
                max_Xi= X_array[:,i].max()
                min_Xi= X_array[:,i].min()
                interval = (max_Xi-min_Xi)/10.0
                p=1.0/10
                
                pox_X_Temp = pos_X
                for k in range(1,11):
                    pos_poss_Xi[k*interval+min_Xi]=(pox_X_Temp[pox_X_Temp[:,i]<=(k*interval+min_Xi)].shape[0]+self.m*p)/(total_pos+self.m)
                    pox_X_Temp=pox_X_Temp[pox_X_Temp[:,i]>(k*interval+min_Xi)]
  
                self.Pxi_pos.append(pos_poss_Xi)

                
                neg_X_Temp = neg_X
                for k in range(1,11):
                    neg_poss_Xi[k*interval+min_Xi]=(neg_X_Temp[neg_X_Temp[:,i]<=(k*interval+min_Xi)].shape[0]+self.m*p)/(total_neg+self.m)
                    neg_X_Temp=neg_X_Temp[neg_X_Temp[:,i]>(k*interval+min_Xi)]
                self.Pxi_neg.append(neg_poss_Xi)


       
    def predict(self, X, schema):
       
        m= len(X)
        n=len(X[0])-1

        alpha =self.m
        #tot_neg = self._total_neg
        #tot_pos = self._total_pos
        preds = np.zeros((m,1))
        self.predict_p=np.zeros((m,1))
        #preds=[]
        for i, xi in enumerate(X):
            Pxi_test_pos = np.zeros((n,1))
            Pxi_test_neg = np.zeros((n,1))
           
            for j, v in enumerate(xi[:-1]):
                # Compute probabilities with additive smoothing
                #print j
                if schema.is_nominal(j):
                    Pxi_test_pos[j] = self.Pxi_pos[j].get(v)
                    Pxi_test_neg[j] =  self.Pxi_neg[j].get(v)
                else:
                    items=self.Pxi_pos[j].items()
                    items.sort()
                    for pair in items:
                        if v<=pair[0]:
                            Pxi_test_pos[j]=pair[1]
                            break
                    
                    items=self.Pxi_neg[j].items()
                    items.sort()
                    
                    for pair in items:
                        if v<=pair[0]:
                            Pxi_test_pos[j]=pair[1]
                            break
                    
            # Compute log pos / neg class ratio
            #print Pxi_test_pos
            cc=np.log(self.pos_prior) + np.sum(np.log(Pxi_test_pos)) - \
                       np.log(self.neg_prior) - np.sum(np.log(Pxi_test_neg))
            print cc
            self.predict_p[i]=cc
            if (cc>0):
                preds[i]=1.0
            else:
                preds[i]=-1.0
        #print preds
        return preds
        
    def predict_proba(self, X):
        
        return self.predict_p
