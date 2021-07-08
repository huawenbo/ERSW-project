# -*- coding: utf-8 -*-

import os
import numpy as np
import copy
import csv

item = ['subjectid', 'gender', 'birthyear', 'intime',
       'outtime', 'hospitaldays', 'sofa', 'respiration', 'coagulation', 
        'liver', 'cardiovascular', 'renal', 'charttime', 'yfy_chname', 
        'yfy_defabb', 'yfy_valuenum', 'yfy_valueuom', 'indiagnosis', 
        'outdiagnosis', 'outdiagnosis_main', 'discharge_type']
chartitem = ['D-Dimer', 'HBA1c', 'HGB', 'PL%', 'Na', 'RET%', 'EO%', 
             'APOB', 'CK-MB', 'pO2', 'SB', 'OrE#', 'TBIL', 'PAO2', 
             'Myelo#', 'Ca', 'VitB12', 'TG', 'Cys-c', 'MYE%', 'MLYMPH', 
             'TP', 'NEUT#', 'CHOI', 'TCO2', 'gGlu', 'NeuMyelo%', 'CO2CP', 
             'gHGB', 'NeuMeta%', 'PTA', 'FIB', 'NRBC %?', 'Blasts', 
             'IG#', 'SBE', 'ABE', 'MCV', 'CHE', 'Cl-', 'Lip', 'GA', 
             'HBeAg', 'CaO2', 'AFP', 'PDW', 'α-HBDH', 'CRE', 'RDW-CV', 
             'Myelo%', 'WBC', 'ALT/AST', 'P-LCR', 'GA%', 'K+', 'AMY', 
             'MONO#', 'SOD', 'CK', 'IRF%', 'TBA', 'PT', 'MCH', 'PL#', 
             'APTTR', 'NSE', 'GM test', 'fPSA', 'UA', 'PTH', 'IMA', 
             'Mg', 'Lac', 'GLO', 'EO#', 'HDL', 'FCOHb', 'LDH', 'Glu', 
             'LFR', 'IG%', 'BUN', 'Cl', 'SpO2', 'TI', 'CA-153', 
             'PaO2/PAO2', 'PA-aDO2', 'NeuMyelo#', 'FDP', 'p50', 'HCV', 
             'PLT', 'PTR', '"PCO2,t"', 'RI', 'CRP', 'MetHb', 'HIV', 
             'APOA', 'AFU', 'DBIL', 'Anti-HBs', 'FiO2', 'MCHC', 'ALB', 
             'RBC', '"pH,t"', 'FePr', 'eGFR', 'MPV', 'INR', 'TPSA', 
             'CYFRA21-1', 'CA-125', 'PA', 'Na+', 'pH', 'Anti-HBc', 
             'HCT', 'GGT', 'CEA', 'APOE', 'HFR', 'ALP', 'ReHb', 
             'hs-CRP', 'Folate', 'MLYMPH%', 'NeuMeta#', 'TPHA', 'FO2Hb', 
             'CA-199', 'AG', 'Hcy', 'PCT', 'APTT', 'G test', 'K', 'BASO#', 
             'BB', 'LPS', 'OrE%', 'HBsAg', 'proBNP', 'pCO2', 'ALT', 'A/G', 
             'Anti-HBe', 'LP', 'LDL', 'Lp(a)', 'RDW-SD', 'RBP', 'LYMPH#', 
             'MONO%', 'NRBC#', 'NEUT%', 'RET#', 'TT', 'LYMPH%', 'BASO%', 
             'procalcitonin', 'P', 'HHb', 'AST', 'IDBIL', 'CA-724', 'RTR', 
             '"PO2,t"', 'Ca2+', 'T', 'MFR', 'RT']
score_label = ['PLT','INR','D-Dimer','FIB','PT']
index = []
for i,chart in enumerate(chartitem):
    if chart in score_label:
        index.append(i)
index.reverse()
print(index)
for i in index:
    chartitem.pop(i)
chart_item = score_label
chart_item.extend(chartitem)
print(len(chart_item))

item = ['subjectid', 'gender', 'birthyear', 'intime', 'outtime', 'hospitaldays', 'sofa', 'respiration', 'coagulation', 'liver', 'cardiovascular', 'renal', 'indiagnosis', 'outdiagnosis', 'outdiagnosis_main', 'discharge_type', 'charttime']
chart_item = ['PLT', 'INR', 'D-Dimer', 'FIB', 'PT', 'HBA1c', 'HGB', 'PL%', 'Na', 'RET%', 'EO%', 'APOB', 'CK-MB', 'pO2', 'SB', 'OrE#', 'TBIL', 'PAO2', 'Myelo#', 'Ca', 'VitB12', 'TG', 'Cys-c', 'MYE%', 'MLYMPH', 'TP', 'NEUT#', 'CHOI', 'TCO2', 'gGlu', 'NeuMyelo%', 'CO2CP', 'gHGB', 'NeuMeta%', 'PTA', 'NRBC %?', 'Blasts', 'IG#', 'SBE', 'ABE', 'MCV', 'CHE', 'Cl-', 'Lip', 'GA', 'HBeAg', 'CaO2', 'AFP', 'PDW', 'α-HBDH', 'CRE', 'RDW-CV', 'Myelo%', 'WBC', 'ALT/AST', 'P-LCR', 'GA%', 'K+', 'AMY', 'MONO#', 'SOD', 'CK', 'IRF%', 'TBA', 'MCH', 'PL#', 'APTTR', 'NSE', 'GM test', 'fPSA', 'UA', 'PTH', 'IMA', 'Mg', 'Lac', 'GLO', 'EO#', 'HDL', 'FCOHb', 'LDH', 'Glu', 'LFR', 'IG%', 'BUN', 'Cl', 'SpO2', 'TI', 'CA-153', 'PaO2/PAO2', 'PA-aDO2', 'NeuMyelo#', 'FDP', 'p50', 'HCV', 'PTR', '"PCO2,t"', 'RI', 'CRP', 'MetHb', 'HIV', 'APOA', 'AFU', 'DBIL', 'Anti-HBs', 'FiO2', 'MCHC', 'ALB', 'RBC', '"pH,t"', 'FePr', 'eGFR', 'MPV', 'TPSA', 'CYFRA21-1', 'CA-125', 'PA', 'Na+', 'pH', 'Anti-HBc', 'HCT', 'GGT', 'CEA', 'APOE', 'HFR', 'ALP', 'ReHb', 'hs-CRP', 'Folate', 'MLYMPH%', 'NeuMeta#', 'TPHA', 'FO2Hb', 'CA-199', 'AG', 'Hcy', 'PCT', 'APTT', 'G test', 'K', 'BASO#', 'BB', 'LPS', 'OrE%', 'HBsAg', 'proBNP', 'pCO2', 'ALT', 'A/G', 'Anti-HBe', 'LP', 'LDL', 'Lp(a)', 'RDW-SD', 'RBP', 'LYMPH#', 'MONO%', 'NRBC#', 'NEUT%', 'RET#', 'TT', 'LYMPH%', 'BASO%', 'procalcitonin', 'P', 'HHb', 'AST', 'IDBIL', 'CA-724', 'RTR', '"PO2,t"', 'Ca2+', 'T', 'MFR', 'RT']
item.extend(chart_item)
score_label = ['PLT','INR','D-Dimer','FIB','PT']

# '0' represent negative,'1' represent positive,'-1' represent lacking or error conditions
score_result = ['-1','0','1']
item = np.array(item)
item = np.insert(item,17,['items_condition', 'sic_value', 'dic_value', 'sic_result', 'dic_result'])
chart_item = np.array(chart_item)
score_label = np.array(score_label)
score_result = np.array(score_result)

def charttime(dich):
    chart_time=[]
    t = [];
    for i in dich:
        chart_time.append(i[12])
    for i in chart_time:
        if i not in t:
            t.append(i)
    chart = np.array(chart_time)
    loca = []
    for i in t:
        loc = np.where(chart == i)
        loca.append(list(loc[0]))
        # loca.append(loc[0])
    # print(len(chart_time))
    return loca,t

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

def subject_writetxt(subject):
    # f = open('sepsis/' + name,'w',encoding='utf8')
    f = open('hospital_txt/' + name,'w',encoding='gbk')
    for spec in subject:
        for j in spec[0:-1]:
            f.write(j+'\t')
        f.write(spec[-1])
        f.write('\n')
    f.close()
def subject_writecsv(subject):
    # with open('sepsis/'+ name[0:-4] + '.csv', 'w', encoding='gbk', newline='') as f:
    with open('hospital_csv/'+ name[0:-4] + '.csv', 'w', encoding='gbk', newline='') as f:
        fw = csv.writer(f)
        fw.writerows(subject)

path = 'hospital/'
# path = 'sepsis_hospital/'
pathdir = os.listdir(path)

pathdir.pop(0)

for name in pathdir:  
    f = open( path + name,'r',encoding='gbk')
    h = f.readlines()
    dic=[]
    for i in range(len(h)):
        dic.append(h[i][0:len(h[i])-1].split('\t'))
    dic = np.array(dic)
    print(np.shape(dic))
    f.close()
    # print(dic[0,12],dic[0,14],dic[0,15])
    # this is to say that col 12 is the charttime, 14 is the chartitem, 15 is the chartvalue
    loca, t = charttime(dic)
    # print(loca,t)
    # print(len(t))
    # print(len(chart_item))
    chart_value = [['' for col in range(len(chart_item))] for row in range(len(t))]
    for i,j in enumerate(loca):
        temp = dic[j,:]
        temp = temp[:,[14,15]]
        for k in temp:
            # print(k)
            index = np.where(chart_item == k[0])
            if np.size(index) != 0:
                # print(index)
                chart_value[i][index[0][0]] = k[1]
    chart_value = np.array(chart_value)
    i = np.array(range(12))
    i = np.insert(i,12,([17,19,19,20]))
    # print(len(i))
    dic_new = dic[0:len(t),i]
    dic_new = np.insert(dic_new,len(i),t,1)
    # print(np.shape(dic_new))
    chart_score = chart_value[:,0:5]
    chart_score = np.insert(chart_score,2,chart_value[:,0],1)
    chart_score = chart_score.tolist()
    # print(type(chart_score))
    # print(chart_score)
    for i1,i in enumerate(chart_score):
        for j1,j in enumerate(i):
            if j == '':
                chart_score[i1][j1] = []
    # print(chart_score[0])
    grade = []
    for i in chart_score:
        grade.append(mark_grade(i))
    grade = np.array(grade)
    # print(np.shape(grade))
    # print(grade)
    # print(np.shape(dic_new))
    # print(np.shape(chart_value))
    # print(np.shape(grade))
    # print(np.shape(item))
    subject_new = np.vstack((item,np.hstack((np.hstack((dic_new,grade)),chart_value))))
    # print(np.shape(subject_new))
    # print(subject_new[0])
    # subject_writetxt(subject_new)
    # subject_writecsv(subject_new)
    np.save('hospital_npy/' + name[0:-4] + '.npy', subject_new)
    print('working down with',name[0:-4])

print('all working down')