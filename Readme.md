# University Second Major Recommnder System using R-GCN

* A python implementation of major recommender system using R-GCN

## RGCN implementation
```
python3 utils.py
python3 model.py
python3 main.py --n-epochs 100000 --evaluate-every 1000 --graph-batch-size 35000
python3 save_result.py
```

## Requirements
* CUDA 10.1
* torch==1.6.0
* torch-geometric==1.7.0

## Reference
https://github.com/MichSchli/RelationPrediction   
https://aclanthology.org/D14-1162.pdf

## Data
num_entity: 43389
num_relation: 17
num_train_triples: 1410295
num_valid_triples: 7409
num_test_triples: 26937  

## Result
**Valid MRR(filtered):0.747374   
Test MRR (filtered): 0.810421   
Hits (filtered) @ 1: 0.687392   
Hits (filtered) @ 3: 0.928158   
**Hits (filtered) @ 10: 0.984811****   

Loss : 0.1378   


