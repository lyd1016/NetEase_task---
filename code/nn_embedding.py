########################NN_Embedding模型构造###################


#################导包并设置模型的随机种子#######################
import numpy as np
import pandas as pd
np.random.seed(10)
from tensorflow import set_random_seed
set_random_seed(15)
from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Reshape, Dropout,BatchNormalization
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from kaggleEvalFunc import gini_cal as gini_c


###################读入数据，并做基本的预处理################################
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

X_train, y_train = train.iloc[:,2:], train.target
X_test = test.iloc[:,1:]

cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))]

X_train = X_train[cols_use]
X_test = X_test[cols_use]

col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if c.endswith('_cat')}

embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c])>2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c])) #打印所有cat特征的取值个数，以获取embedding的维度

print('\n')




def split_features(X):

    input_list = []
 
    for c in embed_cols:
        raw_vals = np.unique(X[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list.append(X[c].map(val_map).values)
     
    #the rest of the columns
    other_cols = [c for c in X.columns if (not c in embed_cols)]
    input_list.append(X[other_cols].values)

    return input_list  





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
		############对所有cat特征做embedding##################

    
        model_ps_ind_02_cat = Sequential()
        model_ps_ind_02_cat.add(Embedding(5, 3, input_length=1))
        model_ps_ind_02_cat.add(Reshape(target_shape=(3,)))
        models.append(model_ps_ind_02_cat)
		
        model_ps_ind_04_cat = Sequential()
        model_ps_ind_04_cat.add(Embedding(3, 2, input_length=1))
        model_ps_ind_04_cat.add(Reshape(target_shape=(2,)))
        models.append(model_ps_ind_04_cat)
		
        model_ps_ind_05_cat = Sequential()
        model_ps_ind_05_cat.add(Embedding(8, 5, input_length=1))
        model_ps_ind_05_cat.add(Reshape(target_shape=(5,)))
        models.append(model_ps_ind_05_cat)
		
        model_ps_car_01_cat = Sequential()
        model_ps_car_01_cat.add(Embedding(13, 7, input_length=1))
        model_ps_car_01_cat.add(Reshape(target_shape=(7,)))
        models.append(model_ps_car_01_cat)
		
        model_ps_car_02_cat = Sequential()
        model_ps_car_02_cat.add(Embedding(3, 2, input_length=1))
        model_ps_car_02_cat.add(Reshape(target_shape=(2,)))
        models.append(model_ps_car_02_cat)
		
        model_ps_car_03_cat = Sequential()
        model_ps_car_03_cat.add(Embedding(3, 2, input_length=1))
        model_ps_car_03_cat.add(Reshape(target_shape=(2,)))
        models.append(model_ps_car_03_cat)
		
        model_ps_car_04_cat = Sequential()
        model_ps_car_04_cat.add(Embedding(10, 5, input_length=1))
        model_ps_car_04_cat.add(Reshape(target_shape=(5,)))
        models.append(model_ps_car_04_cat)
		
        model_ps_car_05_cat = Sequential()
        model_ps_car_05_cat.add(Embedding(3, 2, input_length=1))
        model_ps_car_05_cat.add(Reshape(target_shape=(2,)))
        models.append(model_ps_car_05_cat)
		
        model_ps_car_06_cat = Sequential()
        model_ps_car_06_cat.add(Embedding(18, 8, input_length=1))
        model_ps_car_06_cat.add(Reshape(target_shape=(8,)))
        models.append(model_ps_car_06_cat)
		
        model_ps_car_07_cat = Sequential()
        model_ps_car_07_cat.add(Embedding(3, 2, input_length=1))
        model_ps_car_07_cat.add(Reshape(target_shape=(2,)))
        models.append(model_ps_car_07_cat)
		
        model_ps_car_09_cat = Sequential()
        model_ps_car_09_cat.add(Embedding(6, 3, input_length=1))
        model_ps_car_09_cat.add(Reshape(target_shape=(3,)))
        models.append(model_ps_car_09_cat)
		
        model_ps_car_10_cat = Sequential()
        model_ps_car_10_cat.add(Embedding(3, 2, input_length=1))
        model_ps_car_10_cat.add(Reshape(target_shape=(2,)))
        models.append(model_ps_car_10_cat)
		
        model_ps_car_11_cat = Sequential()
        model_ps_car_11_cat.add(Embedding(104, 10, input_length=1))
        model_ps_car_11_cat.add(Reshape(target_shape=(10,)))
        models.append(model_ps_car_11_cat)
		
        model_rest = Sequential()
        model_rest.add(Dense(16, input_dim=24))
        models.append(model_rest)
        
 

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        
        self.model.add(Dense(120))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(.5))  
#        self.model.add(BatchNormalization())
        self.model.add(Dense(80,activation='relu'))
        self.model.add(Dropout(.35))
		

#        self.model.add(BatchNormalization())
        self.model.add(Dense(20,activation='relu'))
        self.model.add(Dropout(.15))
		
#        self.model.add(BatchNormalization())
        self.model.add(Dense(10,activation='relu'))
        self.model.add(Dropout(.15))
		
           
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


#network training
K = 5
runs_per_fold = 3
n_epochs = 15

cv_ginis = []
full_val_preds = np.zeros(np.shape(X_train)[0])
y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 0, 
                            shuffle = True)    

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    
    X_test_f = X_test.copy()
    
    #######################做上采样
    pos = (pd.Series(y_train_f == 1))
    
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)
    
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    
    
    val_preds = 0
    score = []
    
    for j in range(runs_per_fold):
        model = NN_with_EntityEmbedding(X_train_f,y_train_f)
        val_preds += model.predict(X_val_f)[:,0] / runs_per_fold
        y_preds[:,i] += model.predict(X_test_f)[:,0] / runs_per_fold
        
    full_val_preds[outf_ind] += val_preds
        
    cv_gini = gini_c(val_preds,y_val_f.values)
    cv_ginis.append(cv_gini)
    print ('\nFold %i prediction cv gini: %.5f\n' %(i,cv_gini))
    
print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
print('Full validation gini: %.5f' % gini_c(full_val_preds,y_train.values))

y_pred_final = np.mean(y_preds, axis=1)

df_sub = pd.DataFrame({'id' :test.id, 
                       'target' : y_pred_final},
                       columns = ['id','target'])
df_sub.to_csv('../output/NN_Entity_submit.csv', index=False)

pd.DataFrame(full_val_preds).to_csv('../output/NN_EntityEmbed_offline.csv',index=False)





#Fold 0 prediction cv gini: 0.26651
#
#
#Fold 1 prediction cv gini: 0.28345
#
#
#Fold 2 prediction cv gini: 0.26106
#
#
#Fold 3 prediction cv gini: 0.28453
#
#
#Fold 4 prediction cv gini: 0.25607
#

#Mean out of fold gini: 0.27033