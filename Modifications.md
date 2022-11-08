# Modifications
Modifications changed from DARTS to MC-DARTS


## architect.py

#### Add library
```
import logging
import random
from model import NetworkCIFAR as Network2
```

#### Add main
- Add an optimizer to update the part of the alpha
- Add a param_step, a param_backward_step for Change Alpha's Priority and new loss calculation for the MC-DARTS


## genotypes.py 
- Add PRIMITIVES_PARAM for operation candidates with convolution calculations
- Add architectures searched for in this experiment

## model.py
- Change fully-connected layer inputs

## model_search.py

#### Add library
```
from genotypes import PRIMITIVES,PRIMITIVES_PARAM
import numpy as np
```
#### Add main
- Add a process that uses a part of operations to MixedOp
- Change nn.AdaptiveAvgPool2d to match pytorch version
- Add variables to be defined in initialize_alphas

## model_search.py

#### Add library
```
import torch.nn.functional as F
```
#### Add main
- Add a process that uses a part of operations to MixedOp
- Change FactorizedReduce to process differently depending on whether the input is even or odd.

## train.search.py

#### Add library
```
import datetime
import csv
from collections import namedtuple
import train as training
import visualize as vis
from model import NetworkCIFAR as Network2
from torchvision import transforms
import random
from tqdm import tqdm
from torchsummary import summary
```
#### Add argument
```
parser.add_argument('--target_layers', type=int, default=20, help='target total number of layers')
parser.add_argument('--gammas_learning_rate', type=float, default=6e-2, help='learning rate for arch encoding')
parser.add_argument('--multigpu', default=True, action='store_true', help='If true, training is not performed.')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--train_mode', action='store_true', default=True, help='use train after search')
parser.add_argument('--val_mode', action='store_true', default=True, help='use validation and check accuracy')
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument("--iteration", default=1, type=int, help="iteration")
parser.add_argument("--id", default=1, type=int, help="sampler id")
parser.add_argument("--limit_param", default=2000000, type=int, help="upper limit of params")
parser.add_argument("--lambda_a", default=0.1, type=float, help="lambda of architecture")
```
#### Add main
- Add csv output function for the number of parameters, training loss, etc.
- Add a code to save architecture visualization results per Epochs
- Add a proposed method of calculating the number of parameters and changing the process depending on the results
- Add a code to measure total time, validation time, and the latency
- Add a process that can move directly to the re-training code(train.py) using the architecture with the highest accuracy in the search
- Display of calculation progress using tqdm library

## train.py

#### Add library
```
import csv
import random
import time as tm
import datetime

from torchvision import transforms
from tqdm import tqdm
from torchsummary import summary
```
#### Add argument
```
parser.add_argument('--multigpu', default=True, action='store_true', help='If true, training is not performed.')
parser.add_argument("--iteration", default=1, type=int, help="iteration")
parser.add_argument("--id", default=1, type=int, help="sampler id")
parser.add_argument("--limit_param", default=2000000, type=int, help="upper limit of params")
parser.add_argument("--lambda_a", default=0.0001, type=float, help="lambda of architecture") 
parser.add_argument('--gammas_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
```
#### Add main
- Add csv output function for the number of parameters, the training loss, etc.
- Add a code to measure total time, validation time, and the latency
- Display of calculation progress using tqdm library

## utils.py

#### Add library
```
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
```
#### Add main
- Change torch.eq to match pytorch version
- Added Padding processing to Transform

## visualize.py

#### Add library
```
import os
```
#### Add main
- Changed to be able to move an architecture from train_search.py
