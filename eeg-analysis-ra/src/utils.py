import numpy as np

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
    d = 5000

    phi = np.random.normal(size=(d, np.shape(data)[1]))
    phi /= np.linalg.norm(phi, axis=1)[:, None] # make d x 1 for division
    b = np.random.uniform(data.min(), data.max(), size=(d, 1)) # constant
    H = np.sign(phi.dot(data.T).T + b.T) 
    H = np.sign(np.matmul(phi, data.T) + b) 
    H = H.T.dot(H)
    
    return H

def mmd(data, kernel):
    """
    Compute the MMD
    """
    