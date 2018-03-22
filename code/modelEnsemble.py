#############ModelEnsemble#################
######对各个模型的结果做stacking融合############

########导包#########
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge
from sklearn.model_selection import StratifiedKFold
from kaggleEvalFunc import gini_cal as gini_c

print('loading files...')

##########读入各个模型对训练集的预测结果######

model_lgb = pd.read_csv('../output/model_lgb_offLine.csv')
model_nn = pd.read_csv('../output/model_nn_offLine.csv')
model_nn_embedding = pd.read_csv('../output/NN_EntityEmbed_offline.csv')
model_ffm = pd.read_csv('../output/model_ffm.csv')

train = pd.read_csv('../data/train.csv')
train = train[['id','target']]

train = pd.merge(train,model_lgb,on=['id'],how='outer')
train = pd.merge(train,model_nn,on=['id'],how='outer')
train['model_embedding'] = model_nn_embedding
train['model_ffm'] = model_ffm

     
##########读入各个模型对测试集的预测结果###############

model_lgb_submit = pd.read_csv('../output/model_lgb_submit.csv')
model_lgb_submit.rename(columns={'target':'model_lgb'},inplace=True)
model_nn_submit = pd.read_csv('../output/model_nn_submit.csv')
model_nn_submit.rename(columns={'target':'model11'},inplace=True)
model_nn_embedding_submit = pd.read_csv('../output/NN_Entity_submit.csv')
model_nn_embedding_submit.rename(columns={'target':'model_embedding'},inplace=True)
model_nn_ffm_submit = pd.read_csv('../output/model_ffm_submit.csv')
model_nn_ffm_submit.rename(columns={'target':'model_ffm'},inplace=True)

test = pd.read_csv('../data/test.csv')
test = test[['id']]
test = pd.merge(test,model_lgb_submit,on=['id'],how='outer')
test = pd.merge(test,model_nn_submit,on=['id'],how='outer')
test = pd.merge(test,model_nn_embedding_submit,on=['id'],how='outer')
test = pd.merge(test,model_nn_ffm_submit,on=['id'],how='outer')




sub = pd.DataFrame()
sub['id'] = test['id']
sub['target'] = 0      
   
   
X = train.drop(['id','target'],axis=1).values
y = train['target'].values

T = test.drop(['id'],axis=1).values
np.random.seed(0)

# Run CV
kfold = 5  
skf = StratifiedKFold(n_splits=kfold, random_state=0)
score = []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):  
    # Create data for this fold
    print(i)
    y_train, y_valid = y[train_index], y[test_index]
    X_train, X_valid = X[train_index], X[test_index]
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = X_valid.mean(axis=1)
    score_ = gini_c(y_pred,y_valid)
    print(score_)  
    score.append(score_)   
    y_pred_test =clf.predict(T) 
    sub['target'] += y_pred_test/kfold
       
print('avg-valid-gini:'+ str(sum(score)/kfold))      

sub.to_csv('../output/model_stacking.csv', float_format='%.6f', index=False)


#0
#0.291978156444
#1
#0.289730295232
#2
#0.290653263676
#3
#0.294228540574
#4
#0.282684387932
#avg-valid-gini:0.289854928772