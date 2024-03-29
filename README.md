# MC-DARTS

MC-DARTS (Model Size Constrained Differentiable
Architecture Search) adding constraints to DARTS to search for a network architecture considering the accuracy and the model size.

MC-DARTS has been accepted at NeurIPS 2022 Has it Trained Yet? A Workshop for Algorithmic Efficiency in Practical Neural Network Training!
[MC-DARTS : Model Size Constrained Differentiable
Architecture Search](https://openreview.net/pdf?id=jKJ6OcvqdQ)

This code was created based on the [DARTS](https://github.com/quark0/darts).
New branches for loss and update formulas have been added ([Detailed modifications](https://github.com/itigo-11111/MC-DARTS/blob/main/Modifications.md) ).

## Requirement ( my setting version)

Python (3.7)

Numpy (1.21.5)

Pytorch (1.5.0)

tqdm (4.64.1)

torchsummary (1.5.1)

python-graphviz (0.20)

## Usage

#### Architecture search (CIFAR10)
```
python train_search.py --limit_param 2500000
```

The paper prepares it in advance, but this code downloads the dataset so that it can be run immediately.

#### Architecture evaluation ( arch => use architecture)
```
python train.py --arch "mc_darts2900000"
```

## Network Architecture (Searched)
See [genotypes.py](https://github.com/itigo-11111/MC-DARTS/blob/main/genotypes.py)

## Licence

[apache license 2.0](https://github.com/itigo-11111/MC-DARTS/blob/main/LICENSE)


## Citation
If you use our code, please cite our paper.
```
@inproceedings{
hemmi2022mcdarts,
title={{MC}-{DARTS} : Model Size Constrained Differentiable Architecture Search},
author={Kazuki Hemmi and Yuki Tanigaki and Masaki Onishi},
booktitle={Has it Trained Yet? NeurIPS 2022 Workshop},
year={2022},
url={https://openreview.net/forum?id=jKJ6OcvqdQ}
}
```
