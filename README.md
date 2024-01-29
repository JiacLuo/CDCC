# Cross-Domain Contrastive Learning for Time Series Clustering
The source code is for reproducing experiments of the paper entitled "Cross-Domain Contrastive Learning for Time Series Clustering"
# Datasets
The UCR dataset used in the paper are available at : http://www.timeseriesclassification.com/ .
In order to read the data intuitively and save space, we converted the data into csv format and compressed it. 

# Usage

## Install packages

You can use your favorite package manager, or create a new environment of python 3.6 or greater and use the packages listed in requirements.txt
`pip install -r requirements.txt`

## Setting parameters

Set parameters in file config/CDCC.yaml. 
Hyperparameters were adjusted using grid search and optimized using the Adam optimizer. The hyperparameter learning rate is searched from {0.01, 0.001, 0.0003}, the number of layers num_layers of the BiLSTM are searched from {1, 2, 3}, the parameter batch_size is set according to the size of the dataset, which is searched from {8,16,32,64,128,256}, and the dropout rate p is searched from {0.1, 0.3, 0.5}.

## Run

`python main.py -f config/CDCC.yaml`