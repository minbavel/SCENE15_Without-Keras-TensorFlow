
def sample_G(df, train_per_image):
    
    import pandas as pd
    import DeepLearning

    nclass = df['Labels'].max()
    type(int(nclass))

    fdatabase_label = df.iloc[:,0]

    tr_idx=[]
    ts_idx=[]
    import random

    tr_fea = pd.DataFrame()
    ts_fea = pd.DataFrame()
    for jj in range(nclass+1):
        print(jj)
        idx_label = DeepLearning.find((fdatabase_label), lambda x: x == jj)
        num = len(idx_label)
        tr_num = train_per_image
        idx_label_random = idx_label
        random.shuffle(idx_label_random)   
        tr_idx = idx_label_random[:100]
        ts_idx = idx_label_random[100:]
        tr_fea_classwise = df.iloc[tr_idx,:]
        ts_fea_classwise = df.iloc[ts_idx,:]
        tr_fea = tr_fea.append(tr_fea_classwise, ignore_index = True)
        ts_fea = ts_fea.append(ts_fea_classwise, ignore_index = True)

    return tr_fea, ts_fea