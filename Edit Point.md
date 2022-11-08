# architect.py

### Add library
```
import logging
import random
from model import NetworkCIFAR as Network2
```

### Add main
- Add optimizer to update part of alpha
- Add param_step , param_backward_step for Change Alpha's Priority and new loss calculation for the MC-DARTS.


# genotypes.py 
- Add PRIMITIVES_PARAM for operation candidates with convolution calculations
- Add architectures searched for in this experiment.

# model.py
- Change fully-connected layer inputs
