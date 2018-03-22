#############生成libffm的训练集以及测试集文件###################


import numpy as np
import pandas as pd


##########读入文件#######################
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

test.insert(1,'target',0)


combine = pd.concat([train,test])
combine = combine.reset_index(drop=True)
unwanted = combine.columns[combine.columns.str.startswith('ps_calc_')]
combine.drop(unwanted,inplace=True,axis=1)



features = combine.columns[2:]
categories = []
for c in features:
    trainno = len(combine.loc[:train.shape[0],c].unique())
    testno = len(combine.loc[train.shape[0]:,c].unique())
	
###########对取值数目较多的特征做cut#################	
combine.loc[:,'ps_reg_03'] = pd.cut(combine['ps_reg_03'], 50,labels=False)
combine.loc[:,'ps_car_12'] = pd.cut(combine['ps_car_12'], 50,labels=False)
combine.loc[:,'ps_car_13'] = pd.cut(combine['ps_car_13'], 50,labels=False)
combine.loc[:,'ps_car_14'] =  pd.cut(combine['ps_car_14'], 50,labels=False)
combine.loc[:,'ps_car_15'] =  pd.cut(combine['ps_car_15'], 50,labels=False)


test = combine.loc[train.shape[0]:].copy()
train = combine.loc[:train.shape[0]].copy()



train.drop('id',inplace=True,axis=1)
test.drop('id',inplace=True,axis=1)


categories = train.columns[1:]
numerics = []






currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

noofrows = train.shape[0]
noofcolumns = len(features)

#############生成训练集文件##############
with open("../output/alltrainffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        datastring = ""
        datarow = train.iloc[r].to_dict()
        datastring += str(int(datarow['target']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)
        
noofrows = test.shape[0]
noofcolumns = len(features)
#############生成测试集文件##############
with open("../output/alltestffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        datastring = ""
        datarow = test.iloc[r].to_dict()
        datastring += str(int(datarow['target']))


        for i, x in enumerate(catdict.keys()):
            if(catdict[x]==0):
                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode +=1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)