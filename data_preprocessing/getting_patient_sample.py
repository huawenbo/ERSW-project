# -*- coding: utf-8 -*-

import numpy as np
import os

# N_time: sample time windows length with 8h
N_time = 3
# timestep: prediction time windows length with 8h
timestep = 1

def mask_sub(vals):
    mask_sub = np.zeros(vals.shape)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if vals[i,j] != '':
                mask_sub[i,j] = 1
    return mask_sub

def get_sic_dic(path, listdir, N_time = 3, timestep = 1):
    sic = []
    dic = []
    for sub_id in listdir:
        sub = np.load(path + sub_id)
        for index, vals in enumerate(sub):
            if index >= N_time + timestep - 1:
                if vals[3] !='-1':
                    sic.append((sub_id, sub[index - N_time - timestep + 1:index - timestep + 1, 5],
                    sub[index - N_time - timestep + 1:index - timestep + 1, 6:], 
                    mask_sub(sub[index - N_time - timestep + 1:index - timestep + 1, 6:]), 
                    vals[3]))
                if vals[4] !='-1':
                    dic.append((sub_id, sub[index - N_time - timestep + 1:index - timestep + 1, 5],
                    sub[index - N_time - timestep + 1:index - timestep + 1, 6:], 
                    mask_sub(sub[index - N_time - timestep + 1:index - timestep + 1, 6:]), 
                    vals[4]))
    return sic, dic

def get_sic_dic_all(path, listdir,  N_time = 3, timestep = 1):
    sic = []
    dic = []
    for sub_id in listdir:
        sub = np.load(path + sub_id)
        for index, vals in enumerate(sub):
            if index >= N_time + timestep - 1:
                if vals[3] !='-1':
                    sic.append((sub_id, sub[index - N_time - timestep + 1:index + 1, :]))
                if vals[4] !='-1':
                    dic.append((sub_id, sub[index - N_time - timestep + 1:index + 1, :]))
    return sic, dic

def check_delete_sub(data):
    index_delete = []
    for index, sub in enumerate(data):
        if (sub[2] == '').all():
            index_delete.append(index)
    data_checked = np.delete(data, index_delete, axis=0)
    return data_checked

def check_delete_sub_new(data):
    index_delete = []
    for index, sub in enumerate(data):
        if (sub[2][-2:] == '').all():
            index_delete.append(index)
    data_checked = np.delete(data, index_delete, axis=0)
    return data_checked

def data_corrected(data):
    for i,sub in enumerate(data):
        data[i][2][np.where(sub[2] == '')] = 0
        data[i][2] = data[i][2].astype(np.float32)
        data[i][-1] = np.float32(data[i][-1])
        for j,k in enumerate(data[i][2]):
            for index in range(174):
                if data[i][2][j, index] < np.float32(-100000):
                    temp = str(data[i][2][j, index])
                    data[i][2][j, index] = np.float32(temp[7:])
    return data

if __name__ == '__main__':
    path = 'sepsis_296/'
    listdir = os.listdir(path)
    sic, dic = get_sic_dic(path, listdir, N_time, timestep)
    sic = check_delete_sub(sic)
    dic = check_delete_sub(dic)
    sic = data_corrected(sic)
    dic = data_corrected(dic)
    if not os.path.exists('sample/'):
        os.makedirs('sample/')
    np.save('sample/sic.npy',sic)
    np.save('sample/dic.npy',dic)
    print('working dowm for sic, dic')