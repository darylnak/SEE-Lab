# EEG Analysis
This project aims to utilizie machine learning to identify seizures in an EEG. The model is trained on the "Physionet" 
database collected by the Children's Hospital of Boston.

## Status

##### *3/24/20 - 02:08am*
Re-wrote `mmd()` with `pairwise_kernels` from SciKitLearn. The kernel is crashing on `seiz_Kxx = pairwise_kernels(seizures, seizures, metric='rbf')`. A fix is being investigated.

## In-Progress

  * Implement graph of MMD with respect to seizure and non-seizure over time
  * Optimize MMD calculation
  * Implement additional kernel functions
  * ~~Compute true negtive and true positive for MMD between subject readings and seizure and non-seizure data~~
