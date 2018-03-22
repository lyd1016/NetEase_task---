#######################lightGbmModel构造#################
# -*- coding: utf-8 -*-
############导包###################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from kaggleEvalFunc import gini_cal as gini_c
from kaggleEvalFunc import gini_lgb_used as gini_lgb_used
#####################################读入文件####################
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')






######################################特征预处理#######################
##########对原始特征进行预处理，生成复合特征，空置数目特征，以及对离散特征做one-hot编码，删除'calc特征'
##########离散特征并不是特征名称包含'_cat'的字段，而是所有字段中取值范围大于2小于7的字段。##############


def prePro_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    # for c in dcol:
        # if '_bin' not in c: #standard arithmetic
            # df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            # df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
    one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
				
    unwanted = df.columns[df.columns.str.startswith('ps_calc_')]
    df = df.drop(unwanted, axis=1)  
    return df
def cat_feature_encoder(df_train,df_test,feature):
	temp_cat_df_mean = df_train.groupby(feature,as_index=False)['target'].mean()
	temp_cat_df_mean.rename(columns={'target':feature+'mean'},inplace=True)
	temp_cat_df_sum = df_train.groupby(feature,as_index=False)['target'].sum()
	temp_cat_df_sum.rename(columns={'target':feature+'sum'},inplace=True)
	temp_cat_df_median = df_train.groupby(feature,as_index=False)['target'].median()
	temp_cat_df_median.rename(columns={'target':feature+'median'},inplace=True)
	
	df_train = pd.merge(df_train,temp_cat_df_mean,how='left',on=feature)
	df_train = pd.merge(df_train,temp_cat_df_sum,how='left',on=feature)
	df_train = pd.merge(df_train,temp_cat_df_median,how='left',on=feature)
	
	df_test = pd.merge(df_test,temp_cat_df_mean,how='left',on=feature)
	df_test = pd.merge(df_test,temp_cat_df_sum,how='left',on=feature)
	df_test = pd.merge(df_test,temp_cat_df_median,how='left',on=feature)
	
	return df_train,df_test


train_target = train.target
train = train.drop(['target'],axis=1)
combine= pd.concat([train,test],axis=0)
combine = prePro_df(combine)

train=combine[:train.shape[0]]
test=combine[train.shape[0]:]
train['target'] = train_target
X = train
features = X.columns
X = X.values
y = train['target'].values
         
test_id = test.id.values
sub = pd.DataFrame()
sub['id'] = test_id
sub['model_lgb_submit'] = np.zeros_like(test_id)
cv_vaild_ = train['target']*0
id_train = train.id

######################5cv划分train数据集，并对cat类型特征做目标编码#################################

kfold = 5  
skf = StratifiedKFold(n_splits=kfold, random_state=0)
score_list = []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' gbm kfold: {}  of  {} : '.format(i+1, kfold))
    tempDfTrain = train.ix[train_index]
    tempDfTest = train.ix[test_index]
    encode_feature = ['ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat',\
	'ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat']
	
    for f in encode_feature:
        tempDfTrain,tempDfTest = cat_feature_encoder(tempDfTrain,tempDfTest,f)
	
    tempDfTrain=tempDfTrain.drop(encode_feature,axis=1)
    tempDfTest=tempDfTest.drop(encode_feature,axis=1)
	
    X_train = tempDfTrain.drop(['id','target'],axis=1)
    y_train = y[train_index]
    X_valid = tempDfTest.drop(['id','target'],axis=1)
    y_valid = y[test_index]
    LGB = LGBMClassifier(n_estimators=2000,max_depth=8,learning_rate=0.01,subsample=0.7,colsample_bytree=0.7,min_child_weight=50)
    eval_set=[(X_valid, y_valid)]
    LGB.fit(X_train, y_train,early_stopping_rounds=100,eval_metric=gini_lgb_used,eval_set=eval_set,verbose=100)
    pred = LGB.predict_proba(X_valid,num_iteration=LGB.best_iteration)[:,1]
    
    
    cv_vaild_.iloc[test_index] =pred 
    score = gini_c(pred,y_valid)
    print('valid-gini:'+ str(score))
    score_list.append(score)
    p_test = LGB.predict_proba(test,num_iteration=LGB.best_iteration)[:,1]
    p_test = np.log(p_test)
    sub['model_lgb_submit'] += p_test/kfold
    
print('avg-valid-gini:'+ str(sum(score_list)/kfold))

sub['target'] = np.e**sub['model_lgb_submit']
sub.to_csv('../output/model_lgb_submit.csv', index=False, float_format='%.5f') 


cv_vaild = pd.DataFrame()
cv_vaild['id'] = id_train
cv_vaild['model_lgb'] = cv_vaild_.values
cv_vaild.to_csv('../output/model_lgb_offLine.csv',index=False)
	
#######################成绩记录
#gbm kfold: 1  of  5 : 
#valid-gini:0.288473547843
#gbm kfold: 2  of  5 : 
#valid-gini:0.286108833258
#gbm kfold: 3  of  5 : 
#valid-gini:0.289432654475
#gbm kfold: 4  of  5 :
#valid-gini:0.296236777255
#gbm kfold: 5  of  5 :
#valid-gini:0.280906689611
#avg-valid-gini:0.288231700488