# EEG Analysis
This project aims to utilizie machine learning to identify seizures in an EEG. The model is trained on the "Physionet" 
database collected by the Children's Hospital of Boston.

## Status

##### *4/21/20 - 05:14pm*
New code is currently being implemented into the current codebase for the MMD calculation. Additionally, more features will be added to the features vector. Specifically, variance, skew, and kurtosis will be added, as well as any other features that come up during further research.

## In-Progress
  * Integrate new code into current codebase
  * Implement graph of MMD with respect to seizure and non-seizure over time
  * Add additional features for MMD calculation
  * ~~Compute true negtive and true positive for MMD between subject readings and seizure and non-seizure data~~
