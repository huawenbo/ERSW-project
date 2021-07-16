# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:43:47 2021

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc


def roc_calculate(y_real, y_pred):
    fpr, tpr, thresholds  =  roc_curve(y_real, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc

def roc_plot(fpr, tpr, roc_auc,model_name = '',title = 'Receiver operating characteristic'):
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.plot(fpr, tpr,label = model_name + ' (area = %0.4f)' % roc_auc )
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
def pr_plot(value, name, title ='Precision recall characteristic'):
    fp = value[0]*wrong
    tn = wrong - fp
    tp = value[1]*truth
    fn = truth - tp
    recall = tp/(tp + fn)
    precision = np.insert(tp[1:]/(tp[1:] + fp[1:]),1,1)
    plt.plot(recall, precision,label = name)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.7, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show()

        
data = np.load('roc/123/valid/10.npy')
print(roc_auc_score(data[0],data[1]))
roc_plot(data[0],data[1],roc_auc_score(data[0],data[1]))
# %%

import numpy as np

a = np.array([2,2])
print(np.insert(a,-1,1))
