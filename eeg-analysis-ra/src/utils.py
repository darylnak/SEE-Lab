import numpy as np
from tqdm import tqdm

def downSample(data_train, label_train):

    data_train = np.array(data_train)
    label_train = np.array(label_train)
    
    debug = False

    # since labels are bools, number of Trues is the number of seizures
    numSeizures = np.sum(label_train)
    
    # mask to select all non-seizure data for subject
    all_false = label_train == False

    # select all non-seizure data for subject
    data_train_false = data_train[all_false]

    # randomly sample numSeizures rows from non-seizure data
    sample = np.random.randint(np.size(data_train_false, 0), size = numSeizures)
    data_train_false = data_train[sample, :]
    
    if debug:
        print('Number of seizures {}'.format(numSeizures))
        print('Shape of false data after sample {}'.format(data_train_false.shape))
    
    # create same amount of zeroes (False) as number of seizures
    label_false = np.zeros(numSeizures, dtype='bool')

    if debug:
        print(label_false)
    
    # capture all data marked as seizure
    data_train_true = data_train[np.logical_not(all_false), :]
    label_true = np.ones(numSeizures, dtype='bool')
                                  
    if debug:
        print(label_true)
    
    data_train_down = np.concatenate((data_train_false, data_train_true))
    label_train_down = np.concatenate((label_false, label_true))

    if debug:
        print('Shape of false data labels after sample {}'.format(label_train_down.shape))
        print(label_train_down)
    
    return data_train_down, label_train_down

def hd_kernel(data):
    """
    HD encoding kernel
    
    Input: data - the data to be encoded
    Returns: The encoded data
    """
#     d = 10000

#     phi = np.random.normal(size=(d, np.shape(data)[1]))
#     phi /= np.linalg.norm(phi, axis=1)[:, None] # make d x 1 for division
#     print(phi)
#     b = np.random.uniform(data.min(), data.max(), size=(d, 1)) # constant
#     H = np.sign(np.matmul(phi, data.T) + b) 
#     H = H.T.dot(H)

    
    
    return data@data.T

def MMD(data, kernel_enc):
    """
    Compute the MMD
    """
    mmd = []
    
    for N in tqdm(range(1, data.shape[0])):
            M = data.shape[0] - N
            Kxx = kernel_enc[:N,:N].sum()
            Kxy = kernel_enc[:N,N:].sum()
            Kyy = kernel_enc[N:,N:].sum()
            mmd.append(np.sqrt(
                ((1/float(N*N))*Kxx) + 
                ((1/float(M*M))*Kyy) -
                ((2/float(N*M))*Kxy)
            ))
            
    mmd = np.array(mmd)
    ws = []
    mmd_corr = np.zeros(mmd.size)
            
    for ix in range(1,mmd_corr.size):
        w = ((data.shape[0]-1) / float(ix*(N-ix))) # because N is still in scope from the for-loop above (Python things...)
        ws.append(w)
        mmd_corr[ix] = mmd[ix] - w*mmd.max()
    
    return mmd, mmd_corr