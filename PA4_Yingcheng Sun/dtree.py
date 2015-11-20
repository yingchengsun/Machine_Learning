"""
The Decision Tree Classifier
"""
import numpy as np
import scipy
from math import log



import random 

class Binner(object): #default implementation
    def __call__(self,value): 
        return value   
        
class ContBinner(object): 
    def __init__(self): 
        self.threshold = None
                
    def __call__(self,value): 
        if value <= self.threshold: 
            return 0
        else: return 1        


class DecisionTree(object):


    def __init__(self, depth=None):
        """
        Constructs a Decision Tree Classifier

        @param depth=None : maximum depth of the tree,
                            or None for no maximum depth
        """
        self.dataset = []
        self.k = 0
        self.MAX_DEPTH =depth
        self.children = {}
        self.attr_index = 0 #
        
        self.is_leaf = False #
        self.classifier = None 
        self.binner = Binner()
    
    def calc_entropies(self,part_data):
        H_y_x = 0
        H_x = 0 
        n_data = sum([len(sub_data) for feature,sub_data in part_data.iteritems()])

        for bin,data in part_data.iteritems(): 
            n_bin = float(len(data))
            
            p_bin = n_bin/n_data
            p_plus = sum([1 for ex in data if ex[-1]])/n_bin
            p_minus = 1.0 - p_plus
            
            try: 
                H_y_x_bin = -p_plus*log(p_plus,2)-p_minus*log(p_minus,2)
            except ValueError: #0log0 defined as 0 
                H_y_x_bin = 0  
                
            H_x += -1*p_bin*log(p_bin,2) 
            H_y_x += p_bin*H_y_x_bin
        return H_x,H_y_x  
    
    def partition_data(self,dataset,feature_index,attr_set):
        
        part_data = {}

        if attr_set[feature_index]:
            for i,ex in enumerate(dataset): 
                bin = self.binner(ex[feature_index]) 
                part_data.setdefault(bin,[]).append(ex)
            H_x,H_y_x = self.calc_entropies(part_data)
        else:           
            self.binner = ContBinner()
            dataset = sorted(dataset,key=lambda x:x[feature_index])

            max_entropy_set = (None,None,None)
            
            for i,(ex1,ex2) in enumerate(zip(dataset[:-1],dataset[1:])): 
            #for i,(ex1,ex2) in enumerate(zip(dataset[0:10],dataset[1:11])):    
                part_data = {}
                if ex1[-1] == ex2[-1]: #not a threshold
                    continue
                   
                else:     
                    self.binner.threshold = ex1[feature_index]
                    for ex in dataset: 
                        bin = self.binner(ex[feature_index]) 
                        part_data.setdefault(bin,[]).append(ex)  
                    H_x,H_y_x = self.calc_entropies(part_data)  
                if H_y_x > max_entropy_set[1]: 
                    max_entropy_set =  H_x,H_y_x,part_data
            #print i   
            H_x,H_y_x,part_data = max_entropy_set   
        return H_x,H_y_x,part_data 
                  
        
    def max_GR(self,ex_set,schema): 
        """returns a 2-tuple of (attr_index,part_data) for the attr with the max GR"""
        n_data = float(len(ex_set))
         
        plus=0
        for ex in ex_set:
            if ex[-1]==1:
                plus+=1

        
        p_plus = plus/n_data
        p_minus = 1.0 - p_plus

        try: 
            H_y = -p_plus*log(p_plus,2)-p_minus*log(p_minus,2) 
        except ValueError: 
            H_y = 0       

        GR = 0
        max_GR = (None,None)
        
        for attr_index in range(len(schema)): 
            #print "checking attr: ", attr_index
            
            H_x, H_y_x, part_data = self.partition_data(ex_set,attr_index,schema)
            
            if not H_x: #not partable
                continue
            try: #the data might not be partable on all attrs
                gain_ratio = (H_y - H_y_x)/H_x
                
                if gain_ratio > GR: 
                    GR = gain_ratio
                    max_GR = (attr_index,part_data)
            except ZeroDivisionError: 
                continue        
                 
        return max_GR
    
    def check_ex_set(self,ex_set,attr_set):     
        
        n_data = len(ex_set)
        n_half_data = n_data/2.0
        n_pos = 0
       
        #both checks are True if homogeneous data
        attr_check = True 
        classifier_check = True
        for i,ex in enumerate(ex_set): 
            n_pos += ex[-1]
                
            if attr_check: #still looks homogeneous for attrs
                attr_check = all([ex[attr]==ex_set[0][attr] for attr in range(len(attr_set))])

            if classifier_check: #still looks homogeneous for classifiers    
                classifier_check = ex[-1] == ex_set[0][-1]

            if ((not attr_check) and (not classifier_check)) and i > n_half_data: 
                break
        #calc mcc
        if n_pos == n_half_data: #pick randomly
            mcc = bool(random.randint(0,1))
        else: 
            mcc = n_pos > n_half_data    
                
        return mcc, not(attr_check or classifier_check)        
    
    def fit(self, X, y, schema, depth=0,sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """
            
        dataset=map(list, X)
        
        for i in range(len(dataset)):
             dataset[i].append(y[i])
        f=open('train.txt','w')     
        for d in dataset:
            for dd in d[:-1]:
                f.write(str(dd))
                f.write(',')
            f.write(str(d[-1]))
            f.write('\n')
        f.close()
        
        
        
        
        '''
        attr,part_data = self.max_GR(dataset,schema)
        

        mcc,partable = self.check_ex_set(dataset,schema)
         
        if part_data and not (depth == self.MAX_DEPTH and self.MAX_DEPTH > 0):
            
            self.attr_index = attr
            
            if schema[attr]: 
                new_attr_set = schema  
            else: 
                new_attr_set = schema[:]
                del(new_attr_set[attr])
                print len(new_attr_set)
                #new_attr_set.remove(attr)    

            for feature,sub_data in part_data.iteritems():
                
                self.children[feature] = DecisionTree()
                self.children[feature].fit(sub_data,y,new_attr_set,depth+1)
                
        else: 
            self.is_leaf = True
            self.classifier = mcc #most common classifier        
        '''
    def predict(self, X,depth=0):
        """ Return the -1/1 predictions of the decision tree """
        if self.is_leaf: 
            return self.classifier
        try:     
            bin = self.binner(X[self.attr_index])
            return self.children[bin].predict(X,depth+1)
        except KeyError: #if the classifier is not in this child, randomly select a value
            return random.randint(0,1)    

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        '''
        mean = average(X)
        stddev = std(X, ddof=1)
        t_bounds = t.interval(0.95, len(X) - 1)
        ci = [mean + critval * stddev / sqrt(len(X)) for critval in t_bounds]
        print "Confidence Interval 95%%: %f, %f" % (ci[0], ci[1]) 
  
        return ci[0], ci[1]
        '''
        pass

    def size(self):
        """
        Return the number of nodes in the tree
        """
        size=1    
        if self.children: 
            for feature,child in self.children.iteritems(): 
                c_shape = child.size()
                size += c_shape
                  
        return size       
    

    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        depth = 0
        if self.children: 
            c_depths = []
            for feature,child in self.children.iteritems(): 
                c_shape = child.depth()
                c_depths.append(c_shape)
             
            depth += 1+max(c_depths)
                   
        return depth     
    
