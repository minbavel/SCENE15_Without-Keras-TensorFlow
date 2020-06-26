import pandas as pd
import numpy as np
import scipy.io
import MapMinMax
import cmath

def LayerFirst(train_data, test_data, Elm_Type, NumberofHiddenNeurons, ActivationFunction, C, kkkk, sn, name):
    
    # Macro Defination
    REGRESSION = 0
    CLASSIFIER = 1
    fdafe = 0
    
    # Load training dataset
    T = pd.DataFrame(train_data.iloc[:,0]).apply(np.conj).T
    P = train_data.iloc[:,1:len(train_data.columns)].apply(np.conj).T
    OrgP = P
    train_data = None
    
    # Load testing dataset
    TV_T = pd.DataFrame(test_data.iloc[:,0]).apply(np.conj).T
    TV_P = test_data.iloc[:,1:len(test_data.columns)].apply(np.conj).T
    test_data = None
    
    NumberofTrainingData = len(P.columns)
    NumberofTestingData = len(TV_P.columns)
    NumberofInputNeurons = len(P)
    
    #Labels Generation for Classification Task
    if Elm_Type!=REGRESSION:
        #Preprocessing the data of classification
        sorted_target = pd.concat([T, TV_T], axis=1, ignore_index=True).reindex(TV_T.index)
        (sorted_target.values).sort()
        label = np.zeros(shape=(1,15)).astype(int)
        label = pd.DataFrame(label)
        label[0][0] = sorted_target[0][0]
        
        j = 0
        for i in range(0, NumberofTrainingData+NumberofTestingData):
            if sorted_target[i][0] != label[j][0]:
                j = j+1
                label[j][0] = sorted_target[i][0]
        number_class = j + 1
        NumberofOutputNeurons = number_class

        #Preprossesing targets of training
        temp_T = pd.DataFrame(np.zeros(shape=(NumberofOutputNeurons,NumberofTrainingData)).astype(int))
        for i in range(0, NumberofTrainingData):
            for j in range(0, number_class):
                if label[j][0] == T[i][0]:
                    break
            temp_T[i][j] = 1
            
        T = (temp_T*2-1)
        
        #Preprossesing targets of testing
        temp_TV_T = pd.DataFrame(np.zeros(shape=(NumberofOutputNeurons,NumberofTestingData)).astype(int))
        for i in range(0, NumberofTestingData):
            for j in range(0, number_class):
                if label[j][0] == TV_T[i][0]:
                    break
            temp_TV_T[i][j] = 1
            
        TV_T = (temp_TV_T*2-1)   #End of Elm_Type
        
    #Training Part
    NumberofCategory = len(T)
    
    saveT = T
    
    #Calculate weights & biases
    for subnetwork in range(1, sn):
        for j in range(1, kkkk):
            if j == 1:
                count = 1
            else:
                count = 1
            
            for nxh in range(0, count):  
                if j == 1:
                #Random generate input weights InputWeight (a_f) and biases
                #(b_f) of the initial subnetwork node  based on equation (2)
                    BiasofHiddenNeurons1 = pd.DataFrame(np.random.uniform(0,1,NumberofHiddenNeurons)) 
                    BiasofHiddenNeurons1 = pd.DataFrame(scipy.linalg.orth(BiasofHiddenNeurons1))
                    InputWeight1= pd.DataFrame(np.random.uniform(0,1,size=(NumberofHiddenNeurons, NumberofInputNeurons))*2-1) 
                    if NumberofHiddenNeurons > NumberofInputNeurons:
                        InputWeight1 = pd.DataFrame(scipy.linalg.orth(InputWeight1))
                    else:
                        InputWeight1 = pd.DataFrame(scipy.linalg.orth(InputWeight1.apply(np.conj).T)).apply(np.conj).T
                        
                        
                    tempH=InputWeight1 @ P 
                    ind = pd.DataFrame(np.ones(shape=(1,NumberofTrainingData)).astype(int))
                    BiasMatrix = pd.concat([BiasofHiddenNeurons1[:100]]*1500, axis=1, ignore_index=True) #Extend the bias matrix BiasofHiddenNeurons to match the demention of H
                    tempH=tempH+BiasMatrix
                    
                    # initial subnetwork node generation End #
                else:
                    # update a_f and b_f based on equation (4)-(5)
                    if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                        #PP1=(-log((1./PP)-1))  
                        print("Unknown Activation Function")
                        None        
                    if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                        arcsin_complex = lambda t: cmath.asin(t)
                        vfunc = np.vectorize(arcsin_complex)
                        PP1 = vfunc(PP)
                        PP1 = pd.DataFrame(PP)

                    PP=None
                    PP1=PP1.apply(np.real)    # Get error feedback g^{-1}(u_j(P_{c-1})) in euqation (5) 
                    P=P_save
                    H= None
                    a_1 = (pd.DataFrame(np.eye(len(P))) / C + (P @ (P.apply(np.conj).T)))
                    b_1 = (P @ (PP1.apply(np.conj).T))
                    InputWeight1= (pd.DataFrame(np.linalg.solve(a_1, b_1))).apply(np.conj).T   
                    # input weights calculation in equation (5)

                    fdafe=0;
                    tempH=InputWeight1 @ P
                    
                    YYM_H=InputWeight1 @ pd.concat([P, TV_P], axis=1, ignore_index=True).reindex(TV_P.index) 
                    BB1=PP1.shape  
                    BB2 = pd.DataFrame.sum(pd.DataFrame.sum(tempH-PP1))

                    PP1=None
                   
                    BBP=BB2/BB1[1]   #%%%%%%%%%%%% biases calculation in equation (5) 
                    
                    tempH = (tempH.apply(np.conj).T - BBP.T).apply(np.conj).T  
                    YYM_tempH = (YYM_H.apply(np.conj).T - BBP.T).apply(np.conj).T
                    YYM_H = None
                    
                #Calculation equation (4)-(5) completed %%%%%%
                #Calculate subspace features in equation (6) %%%%%%%
                # Sine

                H = tempH.apply(np.sin)
                tempH = None    
                BiasMatrix = None
                 
                #Save subspace features $H_i$ in harddisk 
                if j>1:
                    YYM_H = YYM_tempH.apply(np.sin)
                    YYM_H, temp_fea = MapMinMax.mapminmax(YYM_H, -1, 1)
                    temp_fea = None
                    H = YYM_H.iloc[:,0:NumberofTrainingData]   # %%%%%% Training features go to the second layer 
                    s2 = "feature_%d.mat" % (subnetwork) # %%%% All the features from Training and Testing are save in the harddisk
                    
                    s2=name + s2
                    scipy.io.savemat(s2,{'struct':YYM_H.values})
                
                # subspace features save End
                P_save = P
                P = H
                H = None
                FT = pd.DataFrame(np.zeros(shape=(3,17766)).astype(int))
                E1 = T
                
                for i in range(0, 2):
                    Y2 = E1
                    tempH = None
                    # get u_n(y) in equation (3) 
                    if fdafe == 0:
                        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                            Y22, PS_subnetwork = MapMinMax.mapminmax(Y2, 0.01, 0.99)
                        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                            Y22, PS_subnetwork = MapMinMax.mapminmax(Y2, -1, 1)
                            Y22_temp, ps = MapMinMax.mapminmax_rev(Y2)
                    else:
                        Y22 = MapMinMax.mapminmax_apply(Y2, PS_subnetwork.xrange, PS_subnetwork.xmin, PS_subnetwork.yrange, PS_subnetwork.ymin) ##################### -1 is ymin
                    
                    Y2 = Y22
                    
                    if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                        print("Unknown Activation Function")
                        #a = (1./Y2)-1
                        #Y4 = -log(a) 
                    if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                        arcsin_complex = lambda t: cmath.asin(t)
                        vfunc = np.vectorize(arcsin_complex)
                        Y4 = vfunc(Y2)
                        Y4 = pd.DataFrame(Y4)
                    
                    Y4 = Y4.apply(np.real)
                    
                    if fdafe == 0:
                        a = (pd.DataFrame(np.eye(len(P))) / C + (P @ (P.apply(np.conj).T)))
                        b = (P @ (Y4.apply(np.conj).T))
                        YYM = pd.DataFrame(np.linalg.solve(a, b))  
                        YJX = ((YYM.apply(np.conj).T) @ P).apply(np.conj).T
                    else:
                        a = Y4.apply(np.conj).T
                        eye = pd.DataFrame(np.eye(len(YYM)))
                        YYM_conj = YYM.apply(np.conj).T
                        a_ling = eye/C+YYM @ YYM_conj
                        ling_solve = np.linalg.solve(a_ling , YYM)
                        ling_solve_df = pd.DataFrame(ling_solve).apply(np.conj).T
                        PP = a @ ling_solve_df
                        PP = PP.apply(np.conj).T  
                        
                        YJX = (PP.apply(np.conj).T) @ YYM
                        
                    BB1 = Y4.shape
                    BB2 = pd.DataFrame(pd.DataFrame.sum(YJX - (Y4.apply(np.conj).T))).T
                    BB = BB2/BB1[1]
                    BB = BB[0]
                    
                    BB = pd.DataFrame(np.full((1500,15), BB[0]))
                    
                    GXZ111 = (YJX.apply(np.conj).T - BB.T).apply(np.conj).T       
                    if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
                        print("Unknown Activation Function")
                        None 
                        #GXZ2=1./(1+exp(-GXZ111')); 
                    if ActivationFunction == 'sin' or ActivationFunction == 'sine':
                        GXZ2=(GXZ111.apply(np.conj).T).apply(np.sin)
                    
                    FYY = pd.DataFrame(ps.reverse(GXZ2.values)).apply(np.conj).T   

                    if i==0:
                        FT1 = FYY.apply(np.conj).T   
                        E1 = E1 - FT1
                    if i==1:
                        FT2 = FYY.apply(np.conj).T
                        E1 = E1 - FT2
                    
                    if i==0:
                        fdafe=1
            
            PP = PP+P
        
        T=E1
        P=OrgP
        fdafe=0
    return NumberofTrainingData