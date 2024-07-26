## Env
- Python 3.8
- Numpy 1.24.3
- Scipy 1.9.3
- Sklearn 1.2.0
- Networkx 2.8.4

To build the environment, please use conda to run the following command.
```sh
$ conda create -n LTGE python=3.8
$ conda activate LTGE
$ conda install numpy==1.24.3
$ conda install scipy==1.9.3
$ conda install networkx==2.8.4
$ conda install -c anaconda scikit-learn
```
## Datasets
Movielen and Serendipity please download from [here](https://grouplens.org/datasets/movielens/) 

Taobao please download from [here](https://tianchi.aliyun.com/dataset/140281)

Other datasets please download from [here](https://snap.stanford.edu/data/)

Please download datasets and create a folder for each dataset in data folder.

## Preprocessing for future link predictioin
```sh
$ python data_prep.py --data superuser --bipartite 0 --train 0.7 --topn 0
```

## Preprocessing for top-N recommendation
```sh
$ python data_prep.py --data bitcoin --bipartite 0 --train 0.7 --topn 1
```

## Preprocessing for incremental future link prediction
```sh
$ python data_prep.py --data wikitalk --bipartite 0 --topn 0 --incremental 1
```

## Evaluation
#### Future link prediction
```sh
$ python LTGE.py --data superuser --k 400 --dimension 32 --task linkprediction
```

#### Recommendation
```sh
$ python LTGE.py --data bitcoin --k 800 --dimension 32 --task recommendation
```

#### Incremental link prediction
```sh
$ python LTGE_in.py --data wikitalk --k 400 --dimension 32 --split 0.2 --split_each 0.02
```

#### Example of future link prediction output on Superuser
```sh
Embedding time is 11.851
Link prediction roc: 0.912
Link prediction AP: 0.931
```
#### Example of top-10 recommendation output on Bitcoin
```sh
recommendation metrics: F1 : 0.7845, MAP : 0.6795, MRR : 0.9411, NDCG : 0.7582
```
