import numpy as np

def eval_kernel(feaset_1, feaset_2, kernel, kparam):
    if(feaset_1.shape[1] != feaset_2.shape[1]):
        print('Error in shape(eval_kernel)')
    (L1, dim) = feaset_1.shape
    (L2, dim) = feaset_2.shape
    if kernel == 'plus':
        print("Unknown Kernel")
        #K = feaset_1.apply(np.conj).T + feaset_2.apply(np.conj).T
    if kernel == 'linear':
        K = feaset_1 @ feaset_2.apply(np.conj).T
    else:
        print('Unknown Kernel')

    return K
    ##################### Other cases needed to be updated