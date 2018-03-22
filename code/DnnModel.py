#########################DNN模型构造####################

#########################相关包导入######################
import numpy as np
np.random.seed(20)
import pandas as pd

from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization,Embedding,Activation,pooling,Merge, Reshape
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from kaggleEvalFunc import gini_cal as gini_c


##################文件读入############################
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

################特征预处理############################
#######drop调会带来性能下降的特征，对类别特征进行one_hot编码##
########生成交叉特征，交叉特征主要有：
########df['ps_car_13'] * df['ps_reg_03']
########df['ps_car_14'] * df['ps_reg_03']
########df['ps_car_14'] * df['ps_car_13']
########df['ps_car_11'] * df['ps_car_13']

to_drop = ['ps_car_11_cat', 'ps_car_10_cat','ps_ind_14','ps_ind_06_bin', 
           'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', "ps_ind_18_bin", 
           'ps_ind_13_bin']

def prePro_df(df):
    df['negative_one_vals'] = np.sum((df==-1).values, axis=1)
	
    cols_use = [c for c in df.columns if (not c.startswith('ps_calc_'))
             & (not c in to_drop)]
    df = df[cols_use]
    one_hot = {c: list(df[c].unique()) for c in df.columns}
    columns = []
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 105:
            columns.append(c)
            for val in one_hot[c]:
                newcol = c + '_oh_' + str(val)
                df[newcol] = (df[c].values == val).astype(np.int)
    df = df.replace(-1, np.NaN)
    df = df.fillna(-1)
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['ps_car_14_x_ps_reg_03'] = df['ps_car_14'] * df['ps_reg_03']
    df['ps_car_14_x_ps_car_13'] = df['ps_car_14'] * df['ps_car_13']
    df['ps_car_11_x_ps_reg_03'] = df['ps_car_11'] * df['ps_reg_03']
    df['ps_car_11_x_ps_car_13'] = df['ps_car_11'] * df['ps_car_13']
    df.drop(['ps_car_14','ps_car_11'],axis=1,inplace=True)
    return df
	
########对需要embedding的特征进行预处理
def split_features(X):
    XUsed = X.copy()
    X_list = []

    ps_car_01_cat_ = XUsed['ps_car_01_cat_'].values
    X_list.append(ps_car_01_cat_)
    
    
    for i in EntityEmbedding_:
        XUsed.drop(i,axis=1,inplace=True)
    X_list.append(XUsed.values)
    return X_list


#############定义Dnn的网络结构###########

class NN_with_EntityEmbedding():

    def __init__(self, X, y):
        super().__init__()
        self.nb_epoch = 15
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.__build_keras_model()
        self.fit(X, y)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self):
        models = []
		############对ps_car_01_cat特征做embedding###################

        model_ps_car_01_cat_ = Sequential()
        model_ps_car_01_cat_.add(Embedding(14, 8, input_length=1))
        model_ps_car_01_cat_.add(Reshape(target_shape=(8,)))
        models.append(model_ps_car_01_cat_)
        

        model_data = Sequential()
        model_data.add(Reshape(target_shape=(198,),input_shape=(198,)))
        models.append(model_data)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dense(100,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.6))      
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def fit(self, X, y):
        df_X = self.preprocessing(X)
        self.model.fit(df_X, y,nb_epoch=self.nb_epoch, batch_size=2048,verbose=0)
        
        
    def fit_(self, X, y, X_val, y_val):
        self.model.fit(self.preprocessing(X), y,
                       validation_data=(self.preprocessing(X_val), y_val),
                       nb_epoch=self.nb_epoch, batch_size=2048,verbose=0
                       # callbacks=[self.checkpointer],
                       )
        
    def predict(self, T):
        df_T = self.preprocessing(T)
        result = self.model.predict(df_T)
        return result

    def summary(self):
        print(self.model.summary())


id_train = train.id
X_train, y_train = train.iloc[:,2:], train.target
X_test, test_id = test.iloc[:,1:], test.id


cv_vaild_ = y_train*0
X_combine = pd.concat([X_train,X_test],axis=0)

X_combine = prePro_df(X_combine)

EntityEmbedding = ['ps_car_01_cat']
EntityEmbedding_= ['ps_car_01_cat_']

for i in EntityEmbedding:
    le = LabelEncoder()
    X_combine[i+"_"] = le.fit_transform(X_combine[i])




X_train=X_combine[:train.shape[0]]
X_test=X_combine[train.shape[0]:]


#######################模型训练过程5折交叉验证，每折做3次bagging，每次bagging改变随机种子##########

K = 5 #number of folds
runs_per_fold = 3 #bagging on each fold

cv_ginis = []
y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 0, 
                            shuffle = True)    

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
          
########对正样例进行上采样##############################################################
    pos = (pd.Series(y_train_f == 1))
    # Add positive examples
    Xpostive = X_train_f.loc[pos].copy()
    ypostive = y_train_f.loc[pos].copy()
    X_train_f = pd.concat([X_train_f, Xpostive,Xpostive], axis=0)
    y_train_f = pd.concat([y_train_f, ypostive,ypostive], axis=0)
    
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    
    #track oof bagged prediction for cv scores
    val_preds = 0
    score = []
    for j in range(runs_per_fold):
        #--model
        model1=Sequential()
        model1.add(Dense(100,activation='relu',input_dim=np.shape(X_train_f)[1]))
        model1.add(BatchNormalization())
        model1.add(Dropout(0.6))
        model1.add(Dense(1,activation='sigmoid'))
        model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        set_random_seed(1000*i+j)
        model1.fit(X_train_f.values, y_train_f.values, epochs=15, batch_size=2048, verbose=0)
        
        val_preds += model1.predict(X_val_f.values)[:,0] / (2*runs_per_fold)
        y_preds[:,i] += model1.predict(X_test.values)[:,0] / (2*runs_per_fold)

        model = NN_with_EntityEmbedding(X_train_f,y_train_f)        
        val_preds += model.predict(X_val_f)[:,0] / (2*runs_per_fold)                
        y_preds[:,i] += model.predict(X_test)[:,0] / (2*runs_per_fold)   
        
        val_gini = gini_c(model.predict(X_val_f)[:,0]/2+model1.predict(X_val_f.values)[:,0]/2,y_val_f.values)   
        print ('\nFold %d Run %d Results *****' % (i, j))
        print ('Validation gini: %.5f\n' % (val_gini))
        score.append(val_gini)
        cv_ginis.append(val_gini)
    print ('\nFold %i prediction cv gini: %.5f\n' %(i,np.mean(score)))
    cv_vaild_.iloc[outf_ind] =val_preds 
print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
y_pred_final = np.mean(y_preds, axis=1)

df_sub = pd.DataFrame({'id' : test_id, 
                       'target' : y_pred_final},
                       columns = ['id','target'])
df_sub.to_csv('../output/model_nn_submit.csv', index=False)

cv_vaild = pd.DataFrame()
cv_vaild['id'] = id_train
cv_vaild['model11'] = cv_vaild_.values
cv_vaild.to_csv('../output/model_nn_offLine.csv',index=False)





#Fold 0 Run 0 Results *****
#Validation gini: 0.27942
#
#
#Fold 0 Run 1 Results *****
#Validation gini: 0.27718
#
#
#Fold 0 Run 2 Results *****
#Validation gini: 0.28090
#
#
#Fold 0 prediction cv gini: 0.27917
#
#
#Fold 1 Run 0 Results *****
#Validation gini: 0.29177
#
#
#Fold 1 Run 1 Results *****
#Validation gini: 0.29296
#
#
#Fold 1 Run 2 Results *****
#Validation gini: 0.29270
#
#
#Fold 1 prediction cv gini: 0.29248
#
#
#Fold 2 Run 0 Results *****
#Validation gini: 0.27047
#
#
#Fold 2 Run 1 Results *****
#Validation gini: 0.27246
#
#
#Fold 2 Run 2 Results *****
#Validation gini: 0.27182
#
#
#Fold 2 prediction cv gini: 0.27158
#
#
#Fold 3 Run 0 Results *****
#Validation gini: 0.29498
#
#
#Fold 3 Run 1 Results *****
#Validation gini: 0.29465
#
#
#Fold 3 Run 2 Results *****
#Validation gini: 0.29478
#
#
#Fold 3 prediction cv gini: 0.29481
#
#
#Fold 4 Run 0 Results *****
#Validation gini: 0.26348
#
#
#Fold 4 Run 1 Results *****
#Validation gini: 0.26664
#
#
#Fold 4 Run 2 Results *****
#Validation gini: 0.26488
#
#
#Fold 4 prediction cv gini: 0.26500
#
#Mean out of fold gini: 0.28061