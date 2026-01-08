## SKOLR: Structured Koopman Operator Linear RNN for Time-Series Forecasting ([paper link](https://arxiv.org/pdf/2506.14113))

This paper has been accepted at ICML 2025.

In this work, we consider the task of time-series forecasting
and establish both an explicit connection and a direct architectural match between Koopman operator approximation
and linear RNNs. 
Building on this connection, we introduce Structured
Koopman Operator Linear RNN (SKOLR). SKOLR implements a structured Koopman operator through a highly parallel linear RNN stack. Through
a learnable spectral decomposition of the input signal,
the RNN chains jointly attend to different dynamical patterns from different representation subspaces, creating a
theoretically-grounded yet computationally efficient design
that naturally aligns with Koopman principles.

## Getting Started


1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

## Training 

All the scripts are in the directory ```./scripts```. For example,
```
sh ./scripts/etth2.sh