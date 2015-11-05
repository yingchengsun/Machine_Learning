"""
Statistics Computations
"""
import numpy as np
import scipy


class StatisticsManager(object):

    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []
        self.prediction_scores = []
        self.training_times = []
        self.statistics = {
            'accuracy' : (accuracy,  self.predicted_labels),
            'precision': (precision, self.predicted_labels),
            'recall'   : (recall,    self.predicted_labels),
            'auc'      : (auc,       self.prediction_scores),
        }

    def add_fold(self, true_labels, predicted_labels,
                 prediction_scores, training_time):
        """
        Add a fold of labels and predictions for later statistics computations

        @param true_labels : the actual labels
        @param predicted_labels : the predicted binary labels
        @param prediction_scores : the real-valued confidence values
        @param training_time : how long it took to train on the fold
        """
        self.true_labels.append(true_labels)
        self.predicted_labels.append(predicted_labels)
        self.prediction_scores.append(prediction_scores)
        self.training_times.append(training_time)

    def get_statistic(self, statistic_name, pooled=True):
        """
        Get a statistic by name, either pooled across folds or not

        @param statistic_name : one of {accuracy, precision, recall, auc}
        @param pooled=True : whether or not to "pool" predictions across folds
        @return statistic if pooled, or (avg, std) of statistic across folds
        """
        if statistic_name not in self.statistics:
            raise ValueError('"%s" not implemented' % statistic_name)

        statistic, predictions = self.statistics[statistic_name]

        if pooled:
            predictions = np.hstack(map(np.asarray, predictions))
            labels = np.hstack(map(np.asarray, self.true_labels))
            return statistic(labels, predictions)
        else:
            stats = []
            for l, p in zip(self.true_labels, predictions):
                stats.append(statistic(l, p))
            return np.average(stats), np.std(stats)

def accuracy(labels, predictions):
    count=0.0
    for i in range(len(labels)):
        if labels[i]==predictions[i]:
            count+=1
    return count/len(labels)

def precision(labels, predictions):
    TP=0.0
    FP=0.0
    for i in range(len(labels)):
        if (labels[i]==1) and (predictions[i]==1):
            TP+=1
        if (labels[i]!=1) and (predictions[i]==1):
            FP+=1
    if TP+FP==0:
        return 0
    else:
        return TP/(TP+FP)

def recall(labels, predictions):
    TP=0.0
    FN=0.0
    for i in range(len(labels)):
        if (labels[i]==1) and (predictions[i]==1):
            TP+=1
        if (labels[i]!=-1) and (predictions[i]==-1):
            FN+=1
    return TP/(TP+FN)

def auc(labels, predictions):
    TP_rate = []
    FP_rate = []
    #thresh=0.5
    
    #need to sort the data by result
    results,labels = zip(*sorted(zip(predictions,labels),reverse=True))
    #calculation for AROC
    for thresh in results:
        TP=FP=FN=TN=0.0
        for i in range(len(labels)):
            if results[i]>thresh and labels[i]==1:
                TP+=1
            if results[i]>thresh and labels[i]==-1:
                FP+=1
            if results[i]<=thresh and labels[i]==1:
                FN+=1
            if results[i]<=thresh and labels[i]==-1:
                TN+=1
        if (FP+TN)!=0:
            FP_rate.append(FP/float(FP+TN))
        else:
            FP_rate.append(0.0)
        if (TP+FN)!=0:
            TP_rate.append(TP/float(TP+FN))
        else:
            TP_rate.append(0.0)
        
    #get the last one
    #tp,fn,fp,tn = cont_table(ROC_set,results,results[-1]*.9)
    aroc = 0  
    for p1,p2 in zip(zip(FP_rate[0:-1],TP_rate[0:-1]),zip(FP_rate[1:],TP_rate[1:])):
        aroc += (p2[0]-p1[0])*(p2[1]+p1[1])/2.0  
        
    return aroc
