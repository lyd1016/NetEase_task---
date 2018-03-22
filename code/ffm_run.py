#!/usr/bin/env python3
from sklearn.model_selection import StratifiedKFold
import subprocess, sys, os, time
import pandas as pd
import numpy as np
from kaggleEvalFunc import gini_cal as gini_c
NR_THREAD = 1

start = time.time()



col_names = ['target']
for i in range(39):
    col_names.append(str(i)+"field")
X_train = pd.read_csv('../output/alltrainffm.txt',sep=' ',names=col_names)
y_train = X_train['target'].copy()
train_id = pd.read_csv('./output/train_id.csv')

col_names.remove('target')
X_test = pd.read_csv('../output/alltestffm.txt',sep=' ',names=col_names)

K = 5 #number of folds
cv_vaild_ = y_train*0
y_preds = np.zeros((np.shape(X_test)[0],K))
score_ = []
kfold = StratifiedKFold(n_splits = K, 
                            random_state = 0, 
                            shuffle = True)    

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    X_train_f.to_csv('./data/X_train_f.txt',sep=' ',index=False,header=False)
    X_val_f.to_csv('./data/X_val_f.txt',sep=' ',index=False,header=False)
    
    cmd = './ffm-train -r 0.1 -k 4 -t 20 -s {nr_thread} -p ./data/X_val_f.txt ./data/X_train_f.txt model'.format(nr_thread=NR_THREAD) 
    subprocess.call(cmd, shell=True)    

    cmd = './ffm-predict ./data/X_val_f.txt model ./data/pred.txt'.format(nr_thread=NR_THREAD) 
    subprocess.call(cmd, shell=True)
    
    cmd = './ffm-predict ./data/alltestffm.txt model ./data/pred_test.txt'.format(nr_thread=NR_THREAD) 
    subprocess.call(cmd, shell=True)
    
    pred = pd.read_csv('./data/pred.txt',names=['pred'])
    val_gini = gini_lgb(pred.values,y_val_f.values)   
    cv_vaild_.iloc[outf_ind] = pred['pred'].values 
    print ('Validation gini: %.5f\n' % (val_gini))
    pred_test = pd.read_csv('./data/pred_test.txt',names=['pred'])        
    y_preds[:,i] = pred_test['pred'].values  
    score_.append(val_gini)

print('Mean out of fold gini: %.5f' % np.mean(score_))
y_pred_final = np.mean(y_preds, axis=1)
print(score_)


df_sub = pd.read_csv('./data/sample_submission.csv')
df_sub['target'] = y_pred_final
df_sub.to_csv('./data/model_ffm_submit.csv',index=False)     

#Mean out of fold gini: 0.28373
#[0.28970919419483354, 0.2766385744928575, 0.28659692986219421, 0.28673948944925648, 0.27896835432508577]

#Mean out of fold gini: 0.28427
#[0.28624180877109406, 0.28187641368107813, 0.27850016335923822, 0.29434828793956824, 0.28037006608527937]



cv_vaild = pd.DataFrame()
cv_vaild['model_ffm'] = cv_vaild_.values
cv_vaild.to_csv('./data/model_ffm.csv',index=False)
print('time used = {0:.0f}'.format(time.time()-start))

