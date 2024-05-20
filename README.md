# HNECL
This is the official PyTorch implementation for the paper:

Hongjie Wei, Junli Wang, Yu Ji, Chungang Yan, Mingjian Guang. Hierarchical Neighbor-Enhanced Graph Contrastive Learning for Recommendation. 2024.

## Overview
We propose a novel graph contrastive learning method called HNECL for recommendation, which aims to fully exploit the hierarchical neighbor relationships from both structural and semantic perspectives for better contrastive view construction.
![image](https://github.com/TJWHJ/HNECL/assets/62538637/231612b8-9d9f-44ef-88f7-2beffa6bb789)

## Requirements
```
recbole==1.0.0
python==3.7.7
pytorch==1.7.1
faiss-gpu==1.8.0
```

## Datasets preparation
For the three datasets (`amazon-books`, `gowalla-merged` and `yelp`) used in our paper, they will be downloaded automatically via RecBole and saved to the `dataset/` folder once you run the main program. Take yelp for example,
```
python main.py --dataset yelp
```
You can also download other datasets and move them to the dataset folder. Note that you may need to add the corresponding configuration file (`dataset_name.yaml`) to the `properties/` folder.

## Hyperparameter configuration
The hyperparameter configuration files are saved in the `properties/` folder, and you can customize hyperparameters for each dataset or use default settings.

For example, if your dataset is named as  `abc`, you can create `properties/abc.yaml` and modify the detailed args as you wish.
```
cd  properties/
touch abc.yaml
```

In implementation, the program will automatically read the configuration file of the corresponding dataset.

## Running 
```
python main.py --dataset amazon-books
```
You can replace amazon-books to gowalla-merged or yelp to reproduce the results reported in our paper.

Also, you can run our model on customized datasets.

