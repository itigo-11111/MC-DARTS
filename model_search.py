import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES,PRIMITIVES_PARAM
from genotypes import Genotype
import numpy as np

class MixedOp(nn.Module):

  def __init__(self, C, stride,num_flag):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.num_flag = num_flag
    for primitive in PRIMITIVES:
      op = OPS[primitive](C , stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)


  def forward(self, x, weights,num_flag):
    if num_flag == 1:
      # param optimization
      i  = 0
      temp1 = 0.0
      for w, op in zip(weights, self._ops[4:]):
        temp1 += w*op(x)
        i += 1
    else:
      temp1 = sum(w * op(x) for w, op in zip(weights, self._ops))
    return temp1

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, num_flag):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.num_flag = num_flag

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride,num_flag)
        self._ops.append(op)

  def forward(self, s0, s1, weights,num_flag):
     # weights => 14*8 or 14*4  weights => 14(2-3-4-5)
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j],num_flag) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1) # concut outputs without s0,s1


class Network(nn.Module):

  def __init__(self,num_flag, device, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.device = device
    self.num_flag = num_flag

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, num_flag)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Linear(C_prev, num_classes)
 
    self._initialize_alphas(device)

  def new(self,device):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to(device)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input,num_flag):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        if num_flag == 1:
          weights = F.softmax(self.gammas_reduce, dim=-1)
        else:
          weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        if num_flag == 1:
          weights = F.softmax(self.gammas_normal, dim=-1)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights,num_flag)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target,num_flag):
    logits = self(input,num_flag)
    return self._criterion(logits, target) 

  def param_loss(self, input, target,num_flag):
    logits = self(input,num_flag)
    return self._criterion(logits, target) 

  def _initialize_alphas(self,device):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    # all operations
    num_ops = len(PRIMITIVES)
    # learnable operations
    num_ops_param = len(PRIMITIVES_PARAM)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).to(device), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).to(device), requires_grad=True)
    self.gammas_normal = Variable(1e-3*torch.randn(k, num_ops_param).to(device), requires_grad=True)
    self.gammas_reduce = Variable(1e-3*torch.randn(k, num_ops_param).to(device), requires_grad=True)
    #param_gate is binary gate : none ,max_poll,avg_pool,skip_connect(weight-free) => 0, sep_conv_x,dil_conv=> 1 
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]
    self._arch_gammas_parameters = [
      self.gammas_normal,
      self.gammas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def arch_gammas_parameters(self):
    return self._arch_gammas_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().detach().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().detach().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
