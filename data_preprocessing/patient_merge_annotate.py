# -*- coding: utf-8 -*-

import numpy as np
import os
import datetime
import copy

path = 'sepsis/sepsis_3809/'
pathdir = os.listdir(path)

def index_acquire(subject):
    time_point = [0,8,16,24]
    time = []
    for i in subject:
        time.append(datetime.datetime.strptime(i[16],'%Y/%m/%d %H:%M:%S'))
    time_min = datetime.datetime.combine(time[0].date(),datetime.datetime.min.time())
    time_max = datetime.datetime.combine(time[-1].date(),datetime.datetime.min.time())
    time_cycle = (time_max - time_min).days
    time_index = []
    # sample from per day method：
    # for i in range((time_cycle + 1)):
    #     index = []
    #     for ind, j in enumerate(time):
    #         if j >= time_min + datetime.timedelta(hours = 24*i) and j < time_min + datetime.timedelta(hours = 24*(i+1)):
    #             index.append(ind)
    #     index.reverse()
    #     time_index.append(index)
    
    # sample using [0,7,17.24] of one day
    for i in range((time_cycle + 1)):
        for point in range(3):
            index = []
            for ind, j in enumerate(time):
                if j >= time_min + datetime.timedelta(hours = i*24 + time_point[point]) and j < time_min + datetime.timedelta(hours = i*24 + time_point[point+1]):
                    index.append(ind)
                    # print(ind)
            index.reverse()
            # print(index)
            time_index.append(index)

    # sample from 12 hours
    # for i in range(2 * (time_cycle + 1)):
    #     index = []
    #     for ind, j in enumerate(time):
    #         if j >= time_min + datetime.timedelta(hours = 12*i) and j < time_min + datetime.timedelta(hours = 12*(i+1)):
    #             index.append(ind)
    #     index.reverse()
    #     time_index.append(index)
    
    # delete empty values
    for i in range(len(time_index)):
        if time_index[i] != []:
            i_max = i
            break
    if i_max >= 3:
        print(i_max) 
    if i_max > 0:
        empty = list(range(0,i_max))
        empty.reverse()
        for i in empty:
            if time_index[i] == []:
                time_index.pop(i)
            else:
                break

    empty = list(range(len(time_index)))
    empty.reverse()
    for i in empty:
        if time_index[i] == []:
            time_index.pop(i)
        else:
            break
    
    return time_index
# supple and merge data
def data_merge(subject, time_index):
    sub_items = []
    for ind, check in enumerate(time_index):
        value = []
        time = ind*8
        if check == []:
            value.append(time)
            for i in range(174):
                value.append('')
        else:
            sub = subject[check, 22:]
            # print(check)
            value.append(time)
            # print(np.shape(sub))
            # print(sub)
            for i in range(174):
                sub_new = sub[:,i]
                for num, j in enumerate(sub_new):
                    if j != '':
                        value.append(j)
                        break
                    elif num == len(sub_new)-1 and j == '':
                        value.append('')
        sub_items.append(value)
    sub_items = np.array(sub_items)
    return sub_items 

def data_supple(chart_score):
    data = copy.deepcopy(chart_score)
    supple = ['200', '0', '200', '0', '100', '0']
    for i,j in enumerate(data):
        if j == []:
            data[i] = supple[i]
    return data

def mark_grade(chart_data):
    
    grade = ['0', '0', '0', '0', '0']
    data = data_supple(chart_data)
    # print(data)
    grade[0] = str(np.size(chart_data[0])) + str(np.size(chart_data[1])) + str(np.size(chart_data[3])) + str(np.size(chart_data[4])) + str(np.size(chart_data[5]))
    sic_value = np.size(chart_data[0])+np.size(chart_data[1])
    # print(sic_value)
    isth_value = np.size(chart_data[2])+np.size(chart_data[3])+np.size(chart_data[4])+np.size(chart_data[5])
    # print(isth_value)

    if sic_value == 2:
        # PLT
        if float(chart_data[0])<100:
            sic_1 = 2
        elif float(chart_data[0])>=100 and float(chart_data[0])<150:
            sic_1 = 1
        else:
            sic_1 = 0
        # INR(PT)
        if float(chart_data[1])>1.4:
            sic_2 = 2
        elif float(chart_data[1])>1.2 and float(chart_data[1])<=1.4:
            sic_2 = 1
        else:
            sic_2 = 0
        
        grade[1] = str(sic_1 + sic_2)
        if sic_1 + sic_2 >= 2:
            grade[3] = '1'
        else:
            grade[3] = '0' 
    else:
        # PLT
        if float(data[0])<100:
            sic_1 = 2
        elif float(data[0])>=100 and float(data[0])<150:
            sic_1 = 1
        else:
            sic_1 = 0
        # INR(PT)
        if float(data[1])>1.4:
            sic_2 = 2
        elif float(data[1])>1.2 and float(data[1])<=1.4:
            sic_2 = 1
        else:
            sic_2 = 0
        
        grade[1] = str(sic_1 + sic_2)
        if sic_1 + sic_2 >= 2:
            grade[3] = '1'
        else:
            grade[3] = '-1'

    if isth_value == 4:
        # PLT
        if float(chart_data[2])<50:
            isth_1 = 2
        elif float(chart_data[2]) >=50 and float(chart_data[2]) <100:
            isth_1 = 1
        else:
            isth_1 = 0
        # D-Dimer
        if float(chart_data[3]) >=3000 and float(chart_data[3])<7000:
            isth_2 = 2
        elif float(chart_data[3])>=7000:
            isth_2 = 3
        else:
            isth_2 = 0
        # Fib-F
        if float(chart_data[4])<100:
            isth_3 = 1
        else:
            isth_3 = 0
        # PT
        if float(chart_data[5])>=19:
            isth_4 = 2
        elif float(chart_data[5])>=16 and float(chart_data[5])<19:
            isth_4 = 1
        else:
            isth_4 = 0
        
        grade[2] = str(isth_1+isth_2+isth_3+isth_4)
        if isth_1+isth_2+isth_3+isth_4>=4:
            grade[4] = '1'
        else:
            grade[4] = '0'
    else:
        # PLT
        if float(data[2])<50:
            isth_1 = 2
        elif float(data[2]) >=50 and float(data[2]) <100:
            isth_1 = 1
        else:
            isth_1 = 0
        # D-Dimer
        if float(data[3]) >=3000 and float(data[3])<7000:
            isth_2 = 2
        elif float(data[3])>=7000:
            isth_2 = 3
        else:
            isth_2 = 0
        # Fib-F
        if float(data[4])<100:
            isth_3 = 1
        else:
            isth_3 = 0
        # PT
        if float(data[5])>=19:
            isth_4 = 2
        elif float(data[5])>=16 and float(data[5])<19:
            isth_4 = 1
        else:
            isth_4 = 0
        
        grade[2] = str(isth_1+isth_2+isth_3+isth_4)
        if isth_1+isth_2+isth_3+isth_4>=4:
            grade[4] = '1'
        else:
            grade[4] = '-1'
    
    return grade

def mark_new(sub_items):
    subject1 = np.insert(sub_items[:,1:6],2,sub_items[:,1],axis = 1)
    subject1 = subject1.tolist()
    # print(subject1)
    for i1,i in enumerate(subject1):
        for j1,j in enumerate(i):
            if j == '':
                subject1[i1][j1] = []
    # print(subject1)
    grade = []
    for i in subject1:
        grade.append(mark_grade(i))
    return np.array(grade)

def state_merge(grade):
    for grade_id in [3,4]:
        patient_state = grade[:,grade_id]
        standrad_merge = [['1',3],['0',6]]
        for ind,state in enumerate(standrad_merge):
            sic_index = np.where(patient_state == state[0])[0]
            for i in range(len(sic_index) - 1):
                if (patient_state[np.arange(sic_index[i]+1,sic_index[i+1])] == '-1').all():
                    if sic_index[i+1]-sic_index[i] <= state[1]:
                        grade[np.arange(sic_index[i]+1,sic_index[i+1]),grade_id] = state[0]
    return grade

# subject = np.load(path + pathdir[1])
# subject = np.delete(subject, 0, axis = 0)
# time_index = index_acquire(subject)
# sub_items = data_merge(subject, time_index)
# grade = state_merge(mark_new(sub_items))
# sub_new = np.hstack((grade,sub_items))

if __name__ == '__main__':
    n = 0
    for i in pathdir:
        subject = np.load(path + i)
        subject = np.delete(subject, 0, axis = 0)
        time_index = index_acquire(subject)
        sub_items = data_merge(subject, time_index)
        # grade = mark_new(sub_items)
        grade = state_merge(mark_new(sub_items))
        subject_new = np.hstack((grade,sub_items))
        np.save('sepsis_3809/' + i, subject_new)
        n = n + 1
        if n % 1000 == 0:
            print('working dowm patients:', n) 
    print('working dowm for all patients:', n)