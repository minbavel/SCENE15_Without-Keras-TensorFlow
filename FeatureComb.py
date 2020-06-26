import Eval_Kernel
import pandas as pd
import numpy as np
import scipy.io

def featurecomb(Ktr, Kte, name, sn, NumberofTrainingData):
    for loop in range(1, sn):
        s2 = "feature_%d.mat" % (loop) 
        s2 = name + s2
        feature_load = scipy.io.loadmat(s2)
        YYM_H = feature_load.get('struct')
        YYM_H = pd.DataFrame(YYM_H)
        H_train1 = YYM_H.iloc[:,0:NumberofTrainingData]
        H_test1 = YYM_H.iloc[:,NumberofTrainingData:]
        
        Ktr_temp = Eval_Kernel.eval_kernel(H_train1.apply(np.conj).T, H_train1.apply(np.conj).T, 'linear', 1)
        Ktr = Ktr + Ktr_temp

        H_test1 = H_test1.apply(np.conj).T
        H_test1 = H_test1.reset_index(drop=True)
        
        Kte_temp = Eval_Kernel.eval_kernel(H_test1, H_train1.apply(np.conj).T, 'linear', 1)
        Kte = Kte + Kte_temp
        YYM_H = None
        
    return Ktr,Kte
