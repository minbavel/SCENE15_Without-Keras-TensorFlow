import pandas as pd
import numpy as np
import scipy.io
import MapMinMax
import cmath

def lastLayer(TrainingData_File, TestingData_File, Elm_Type, ActivationFunction, kkk, C):

    # Macro Defination
    REGRESSION = 0
    CLASSIFIER = 1
    
    # Load training dataset
    train_data = TrainingData_File
    T = pd.DataFrame(train_data.iloc[:,0]).apply(np.conj).T
    P = train_data.iloc[:,1:len(train_data.columns)].apply(np.conj).T
    train_data = None
    
    # Load testing dataset
    test_data = TestingData_File
    TV_T = pd.DataFrame(test_data.iloc[:,0]).apply(np.conj).T
    TV_P = test_data.iloc[:,1:len(test_data.columns)].apply(np.conj).T
    test_data = None
    
    NumberofTrainingData = len(P.columns)
    NumberofTestingData = len(TV_P.columns)
    
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
        #print(temp_T)
        for i in range(0, NumberofTestingData):
            for j in range(0, number_class):
                if label[j][0] == TV_T[i][0]:
                    break
            temp_TV_T[i][j] = 1
            
        TV_T = (temp_TV_T*2-1)   #End of Elm_Type

    # Training Part %%%%%%%%%%%%%%%%%

    D_YYM_i = pd.DataFrame()
    shape1 = T.shape
    Y = pd.DataFrame(np.zeros(shape=T.shape).astype(int))
    E1=T

    for i in range (1, kkk):
        Y2=E1

        # Get $e_{c-1}$ in equation (9) %%%%%%%%
        
        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            Y22, PS_i = MapMinMax.mapminmax(Y2, 0.01, 0.99)
            Y22_temp, ps = MapMinMax.mapminmax_rev(Y2, 0.01, 0.99)
        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            Y22, PS_i = MapMinMax.mapminmax(Y2, 0, 1)
            Y22_temp, ps = MapMinMax.mapminmax_rev(Y2)

        Y2=Y22.apply(np.conj).T

        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            a = (1./Y2)-1
            minlog_complex = lambda t: -cmath.log(t)
            vfunc = np.vectorize(minlog_complex)
            Y4 = vfunc(a)
            Y4 = pd.DataFrame(Y4).apply(np.conj).T

        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            Y4 = pd.DataFrame(Y2.apply(np.arcsin)).apply(np.conj).T
        # End %%%%%%%%

        #Get input weights of a subnetwork node $a^c_p$ in euqation (9) %%%%
        P = P.reset_index(drop=True)
        a = (pd.DataFrame(np.eye(len(P))) / C + P @ (P.apply(np.conj).T))
        b = (P @ (Y4.apply(np.conj).T))
        YYM = pd.DataFrame(np.linalg.solve(a, b))

        D_YYM_i = YYM 

        YJX = P.apply(np.conj).T @ YYM

        BB1=Y4.shape  
        BB2 = pd.DataFrame.sum(pd.DataFrame.sum(YJX-Y4.apply(np.conj).T))
        BB =BB2/BB1[1]

        GXZ111 = P.apply(np.conj).T @ YYM - BB
        
        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            #print("Unknown Activation Function")
            GXZ2 = 1./(1+(-GXZ111.apply(np.conj).T).apply(np.exp))  
        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            GXZ2 = (GXZ111.apply(np.conj).T).apply(np.sin)

        FYY = pd.DataFrame(ps.reverse(GXZ2.values))

        # End %%%%%%
        # updated training error %%%%

        FT1=FYY
        E1=E1-FT1   
        # End 

        Y=Y+FYY  #Total output 

        # Training accuracy calculation 
        if Elm_Type == CLASSIFIER:
            MissClassificationRate_Training=0
            for i in range(0, len(T.columns)):
                x, label_index_expected = T.iloc[:,i].max(0), T.iloc[:,i].idxmax(0)
                x, label_index_actual = Y.iloc[:,i].max(0), Y.iloc[:,i].idxmax(0)
            if label_index_actual!=label_index_expected:
                MissClassificationRate_Training=MissClassificationRate_Training+1

            TrainingAccuracy=1-MissClassificationRate_Training/len(T.columns)

        # End 

    # Training Part End 

    
    #  Testing part begian 
    temp = pd.DataFrame(np.zeros(shape=TV_T.shape).astype(int))
    TY2= temp*TV_T;

    # Get output for testing samples
    for i in range(1,kkk):
        TV_P = TV_P.reset_index(drop=True)
        GXZ1= (D_YYM_i.apply(np.conj).T) @ TV_P - BB 
        if ActivationFunction == 'sig' or ActivationFunction == 'sigmoid':
            #print("Unknown Activation Function")
            GXZ2 = 1./(1+(-GXZ1.apply(np.conj).T).apply(np.exp)) 
        if ActivationFunction == 'sin' or ActivationFunction == 'sine':
            GXZ2 = (GXZ1.apply(np.conj).T).apply(np.sin)
        
        FYY = pd.DataFrame(ps.reverse((GXZ2.apply(np.conj).T).values))
        
        TY2=TY2+FYY

    # End 
    # Testing accuracy calculation 
    if Elm_Type == CLASSIFIER:
        MissClassificationRate_Testing=0
        for i in range(0, len(TV_T.columns)):
            x, label_index_expected = TV_T.iloc[:,i].max(0), TV_T.iloc[:,i].idxmax(0)
            x, label_index_actual = TY2.iloc[:,i].max(0), TY2.iloc[:,i].idxmax(0)
            if label_index_actual!=label_index_expected:
                MissClassificationRate_Testing=MissClassificationRate_Testing+1

        TestingAccuracy=1-MissClassificationRate_Testing/len(TV_T.columns)

    # End 

    # Training Part End 
    return TrainingAccuracy, TestingAccuracy
