# EEG Analysis
This project aims to utilizie machine learning to identify seizures in an EEG. The model is trained on the "Physionet" 
database collected by the Children's Hospital of Boston.

## Status

##### *3/25/20 - 04:29am*
Further optimized MMD function. Computing all pairwise dot products for `non_seizures` takes ~2.2 hours, down from > 4700 hours. 

## In-Progress

  * Implement graph of MMD with respect to seizure and non-seizure over time
  * Optimize MMD calculation
  * Implement additional kernel functions
  * ~~Compute true negtive and true positive for MMD between subject readings and seizure and non-seizure data~~
