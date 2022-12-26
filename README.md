# Fork of Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations

This codebase contains the python scripts for MAN-S, the model for the EMNLP 2020 paper [link](https://www.aclweb.org/anthology/2020.emnlp-main.676/). The original codebase is present at: [link](https://github.com/midas-research/man-sf-emnlp)

## Environment & Installation Steps
Dependencies can be installed via: 
```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset from [StockNet dataset](https://github.com/yumoxu/stocknet-dataset) and [Temporal Relational Stok ranking]https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)

## Run
Execute the following python command to preprocess the data
```python
python data_preprocess.py
```


Execute the following python command to train MAN-SF: 
```python
python train.py
```

The output will be saved in finetuned folder

To run the evaluation: 
```python
python eval.py
```

The training log is as in the image 
![Training log](Tensorboard_log.png?raw=true "Tensorboard log")
