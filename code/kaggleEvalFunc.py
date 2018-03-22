##################误差函数计算######################

import numpy as np
	
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_cal(preds, y):
    score = gini(y, preds) / gini(y, y)
    return score

def gini_lgb_used(y, preds):
    score = gini(y, preds) / gini(y, y)
    return 'gini',score,True