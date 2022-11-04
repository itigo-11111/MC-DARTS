# MC-DARTS

MC-DARTS (Model Size Constrained Differentiable
Architecture Search) adding constraints to DARTS to search for a network architecture considering the accuracy and the model size.

This code was created based on the [DARTS](https://github.com/quark0/darts).
New branches for loss and update formulas have been added.

## Requirement ( my setting version)

Python (3.7)

Numpy (1.21.5)

Pytorch (1.5.0)

tqdm (4.64.1)

torchsummary (1.5.1)

python-graphviz (0.20)

## Usage

##### architecture search 
```
python train_search.py --limit_param 2500000
```

##### architecture evaluation ( arch => use architecture)
```
python train.py --arch "mc_darts2900000"
```

## Network Architecture (Searched)
See [genotypes.py](https://github.com/itigo-11111/MC-DARTS/blob/main/genotypes.py)

## Reference


## Licence

[apache license 2.0](https://github.com/itigo-11111/MC-DARTS/blob/main/LICENSE)


## Citation
