# -*- coding: utf-8 -*-

import numpy as np
import os

# reading the source data
pathdir = os.listdir('./')
print(pathdir)

# f1 = open('patientsdata1.txt','r',encoding='gbk')
# f1 = open('patientsdata2.txt','r',encoding='gbk')
f1 = open('patientsdata3.txt','r',encoding='gbk')
h = f1.readlines()
dic=[]
for i in range(len(h)):
    dic.append(h[i][0:len(h[i])-1].split('\t'))

print('data read successfully')

# checking the data is regular
print('长度：',len(dic))
num = []
for i in range(len(dic)):
    num.append(len(dic[i]))
print(min(num),max(num))
print(dic[1][7])

# delete the empty values
del_index = []
for i in range(len(dic)):
    if(dic[i][12] == ''):
       del_index.append(i)
print(len(del_index))
del_index.reverse()
for i in del_index:
    dic.pop(i)
print(len(dic))

# check
dic = np.array(dic)
chart = dic[:,12]
print(type(chart))
index = np.where(chart == '')
print(index)

# divide the data with subject_id
def dic_get(hadm_location):
    dic_getting = [];
    for i in hadm_location[0]:
        dic_getting.append(dic[i+1])
    return dic_getting

hadm_id = []
hadm_where = []
for i in range(1,len(dic)):
    hadm_id.append(dic[i][0])
hadm = list(set(hadm_id))
print(type(hadm[0]))
print(len(hadm))
print(hadm[0])
hadm_id = np.array(hadm_id)
for i in hadm:
    hadm_location = np.where(hadm_id == i)
    dici1 = dic_get(hadm_location)
    # dici1=timesort(dici)
    # print(dici == dici1)
    f = open('hospital/' + i + '.txt','w')
    for spec in dici1[0:-1]:
        for j in spec[0:-1]:
            f.write(j+'\t')
        f.write(spec[-1])
        f.write('\n')
    for j in dici1[-1][0:-1]:
        f.write(j+'\t')
    f.write(dici1[-1][-1])
    f.write('\n')
    f.close()
print('work down!')