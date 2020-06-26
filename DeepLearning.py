import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cmath
import time
import MapMinMax
import Sample_G
import LayerFirst
import LastLayer
import FeatureComb
import Eval_Kernel
import sys
import time
import math
import scipy.io

start_time = time.time()

Ktr=0;Kte=0;  # Ktr denotes training features extracted from hundreds of subnetwork nodes. 
##############  Kte denotes testing features extracted from hundreds of subnetwork nodes.

mat = scipy.io.loadmat('D:/Deep Learning/Scene/Scene15/spatialpyramidfeatures4scene15.mat')

type(mat)

for key in mat.keys(): 
    print (key)
 
labels_raw = mat.get('labelMat')
images_raw = mat.get('featureMat')

labels_trans = np.transpose(labels_raw)
images_trans = np.transpose(images_raw)
len(labels_trans)
len(images_trans)
print(images_trans[0])
print(images_trans[1])
print(images_trans[2])

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
    
labels = []
images = [] 

for x in range(0, len(labels_trans)):
    labels.append(find(labels_trans[x], lambda x: x == 1))
    images.append(images_trans[x])

images
images
type(labels)
len(images)

import pandas as pd
df = pd.DataFrame(images)
print(df)

labels_int = [] 

for i in range(len(labels)):
    for j in range(len(labels[i])):
        print(labels[i][j], end=' ')
        labels_int.append(labels[i][j])
    print()
    
df.insert(0,"Labels", labels_int)
print(df)
type(labels_int[0])

train_per_image = 100

Training, Testing = Sample_G.sample_G(df, train_per_image);
np.set_printoptions(threshold=sys.maxsize)

for x in range(10):
    name = "scene15_channel_%d" %x
    num_subnetwork_node = 4
    dimension = 100
    C1 = 2**8
    
    NumberofTrainingData = LayerFirst.LayerFirst(Training, Testing, 1, dimension, 'sine', C1, 3, num_subnetwork_node, name)

    (Ktr, Kte) = FeatureComb.featurecomb(Ktr, Kte, name, 4, NumberofTrainingData)

Training =  pd.concat([Training.iloc[:,0], Ktr], axis=1, ignore_index=True).reindex(Ktr.index)
Testing = pd.concat([Testing.iloc[:,0], Kte], axis=1, ignore_index=True).reindex(Kte.index)

C2 = 2 ** 12
(train_accuracy11, test_accuracy) = LastLayer.lastLayer(Training, Testing, 1, 'sig', 2, C2)

print("--- %s seconds ---" % (time.time() - start_time))
print("Testing Accuracy: ", test_accuracy)