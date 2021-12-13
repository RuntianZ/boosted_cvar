# Boosted CVaR Classification

Runtian Zhai, Chen Dan, Arun Sai Suggala, Zico Kolter, Pradeep Ravikumar  
NeurIPS 2021

## Table of Contents
- [Quick Start](#quick-start)
  - [Train](#train)
  - [Evaluation](#evaluation)
- [Introduction](#introduction)
- [Algorithms](#algorithms)
- [Parameters](#parameters)
- [Citation and Contact](#citation-and-contact)

## Quick Start

Before running the code, please install all the required packages in `requirements.txt` by running:
```shell
pip install -r requirements.txt
```

In the code, we solve linear programs with the MOSEK solver, which requires a license. You can acquire a free academic license from https://www.mosek.com/products/academic-licenses/. Please make sure that the license file is placed in the correct folder so that the solver could work.

### Train

To train a set of base models with boosting, run the following shell command:
```shell
python train.py --dataset [DATASET] --data_root /path/to/dataset 
                --alg [ALGORITHM] --epochs [EPOCHS] --iters_per_epoch [ITERS]
                --scheduler [SCHEDULER] --warmup [WARMUP_EPOCHS] --seed [SEED]
```

Use the `--download` option to download the dataset if you are running for the first time. Use the `--save_file` option to save your training results into a `.mat` file. Set the training hyperparameters with `--alpha`, `--beta` and `--eta`.

For example, to train a set of base models on Cifar-10 with AdaLPBoost, use the following shell command:
```shell
python train.py --dataset cifar10 --data_root data --alg adalpboost 
                --eta 1.0 --epochs 100 --iters_per_epoch 5000
                --scheduler 2000,4000 --warmup 20 --seed 2021
                --save_file cifar10.mat
```

### Evaluation

To evaluate the models trained with the above command, run:
```shell
python test.py --file cifar10.mat
```

## Introduction

In this work, we study the CVaR classification problem, which requires a classifier to have low &alpha;-CVaR loss, i.e. low average loss over the worst &alpha; fraction of the samples in the dataset. While previous work showed that no deterministic model learning algorithm can achieve a lower &alpha;-CVaR loss than ERM, we address this issue by learning randomized models. Specifically we propose the Boosted CVaR Classification framework that learns ensemble models via Boosting. Our motivation comes from the direct relationship between the CVaR loss and the LPBoost objective. We implement two algorithms based on the framework: one uses LPBoost, and the other named AdaLPBoost uses AdaBoost to pick the sample weights and LPBoost to pick the model weights.

## Algorithms

We implement three algorithms in `algs.py`:

| Name      | Description |
| ----------- | ----------- |
| uniform      |  All sample weight vectors are uniform distributions. |
| lpboost   | Regularized LPBoost (set `--beta` for regularization).  |
| adalpboost  |  &alpha;-AdaLPBoost. |

`train.py` only trains the base models. After the base models are trained, use `test.py` to select the model weights by solving the dual LPBoost problem.

## Parameters
All default training parameters can be found in `config.py`. For Regularized LPBoost we use &beta; = 100 for all &alpha;. For AdaLPBoost we use &eta; = 1.0.

## Citation and Contact
To cite this work, please use the following BibTex entry:
```
@inproceedings{
zhai2021boosted,
title={Boosted {CV}aR Classification},
author={Runtian Zhai and Chen Dan and Arun Suggala and J Zico Kolter and Pradeep Kumar Ravikumar},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=INsYqFjBWnF}
}
```
To contact us, please email to the following address: `Runtian Zhai <rzhai@cmu.edu>`
