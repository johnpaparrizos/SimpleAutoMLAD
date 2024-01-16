# AutoTSAD

### 1. Environment Set Up

Pytorch 1.11

TensorFlow 2.6.0

scikit-learn 0.22.1

Required installation
```
cd repo/matrixprofile
pip install .
```

### 2. Get Started

#### Introduction of each folders
weights: Weights for pretrained model selector

models, utils: Codes for model selector

TSB_UAD: TSB Benchmark suite

#### Main File
```
python main_preds.py -a Avg_ens Pointwise_Preds Pointwise_Preds_Ens Pairwise_Preds Pairwise_Preds_Ens Listwise_Preds Listwise_Preds_Ens
```

Output:

(1) Avg_ens --> Anomaly Score

(2) Pointwise_Preds, Pairwise_Preds, Listwise_Preds --> Top Selected Anomaly Detector with its hyperparameter (e.g., IForest_1_100)

(3) Pointwise_Preds_Ens, Pairwise_Preds_Ens, Listwise_Preds_Ens --> Top 5 Selected Anomaly Detector with its hyperparameter

#### Things to modify

(1) Det_Pool: Candidate Model Set for Model Selector 
(One of the anomaly detector: NORMA is protected by patent application FR2003946)

(2) Det_Pool_Avg: Candidate Model Set for Average Ensembling

(3) filepath: main_preds.py Line 50