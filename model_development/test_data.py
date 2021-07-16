# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:54:36 2021

@author: Administrator
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np
# %%
data = torch.load('dataset/dic_3_1_7878.pt')
print(len(data))
print(data[0][2].shape)
# %%

for i in data:
    for j in i[2]:
        for k in j:
            if k<-10000:
                print(k)
print('dowm')

# %%
data_new = []
for i in data:
    if i[3].sum() != 0:
        data_new.append(i)

torch.save(data_new, 'dic_3_1_7899_5_new.pt')

print('down')



# %%
writer = SummaryWriter()
writer.add_pr_curve(tag, labels, predictions)
# %%
result = np.load('roc/test/1.npy')
print(result[0].shape)
print(roc_auc_score(result[0], result[1]))


# %%
da = torch.load('dataset/mimic_dic_1210.pt')
# %%
num = 0
for i in da:
    num += i[-1]
print(num, len(da)-num)