import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import csv
import random
import time as tm
import datetime

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from torchvision import transforms
from tqdm import tqdm
from torchsummary import summary

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./cifar10/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument('--arch', type=str, default='mc_darts2900000', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--multigpu', default=True, action='store_true', help='If true, training is not performed.')
parser.add_argument("--iteration", default=1, type=int, help="iteration")
parser.add_argument("--id", default=1, type=int, help="sampler id")
parser.add_argument("--limit_param", default=2000000, type=int, help="upper limit of params")
parser.add_argument("--lambda_a", default=0.0001, type=float, help="lambda of architecture") 
parser.add_argument('--gammas_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')

args = parser.parse_args()

# def main(genotype = eval("genotypes.%s" % args.arch),limit_param=9990000):
def main():
  # args.limit_param = limit_param
  # args.layers = layers
  args.img_size = (32, 32)

  args.save = './mc-darts_cifar10_train/name_{}_id_{}/'.format(args.arch,args.id)  
  create_dir(args.save)
  # utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
    
  args.seed = args.id *1000 + args.iteration

  random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  torch.manual_seed(args.seed)
  # cudnn.benchmark = True
  cudnn.enabled=True
  # logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  start_time = tm.time()

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  # args.arch : genotype of original paper 
  # args.arch = "mc_darts" + str(args.limit_param)

  genotype = eval("genotypes.%s" % args.arch) 
  logging.info('genotype = %s', genotype)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype,0)
  model = model.to(device)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_train_acc = 0.0
  best_acc = 0.0
  csv_file_path = args.save + 'output.csv'
  with open(csv_file_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['train_loss','train_acc','val_loss','val_acc', 'time', 'lr' , 'epoch','val_time','latency'])

  pytorch_total_params = sum(p.numel() for p in model.parameters())
  pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print("number of parameters : {} (trainable only : {})".format(pytorch_total_params,pytorch_total_params_train))
  
  with open(args.save+'param.txt', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['param_size', 'param_size_train'])
    writer.writerow([pytorch_total_params,pytorch_total_params_train])
  
  for epoch in range(1, args.epochs + 1):


    if epoch != 1:
      scheduler.step()
    # logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    # summary(model,(3,32,32))
    # print(aaa)

    train_acc, train_obj = train(train_queue, model, criterion, optimizer,device)
    if train_acc > best_train_acc:
      best_train_acc = train_acc
    # logging.info('train_acc %f, best_train_acc %f', train_acc, best_train_acc)

    valid_acc, valid_obj,latency,val_time = infer(valid_queue, model, criterion,device)
    if valid_acc > best_acc:
        best_acc = valid_acc
    # logging.info('valid_acc %f, best_acc %f valid_time %f', valid_acc, best_acc,valid_time)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logging.info('Epoch[{0:03}/{1:03}]  Train Loss:{2:.6f} Train Acc:{3:.6f} Best Train Acc:{4:.6f} Val Loss:{5:.6f} Val Acc:{6:.6f} Best Val Acc : {7:.6f} Val time:{8:.6f} Latency:{9:.6} Now {10} '.format(\
                  epoch, args.epochs, train_obj, train_acc, best_train_acc, valid_obj, valid_acc, best_acc,  val_time, latency, now_time))

    with open(csv_file_path, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([train_obj,train_acc,valid_obj, valid_acc, now_time, optimizer.param_groups[0]['lr'], epoch,val_time,latency])
    # utils.save(model, os.path.join(args.save, 'weights.pt'))

  end_time = tm.time()
  interval = end_time - start_time
  interval = str("time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
  with open(args.save+'time.txt', 'a') as f:
      writer = csv.writer(f)
      writer.writerow(['time'])
      writer.writerow([interval])


def train(train_queue, model, criterion, optimizer,device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  bar = tqdm(desc = "Training", total = len(train_queue), leave = False)
  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).to(device)
    target = Variable(target).to(device)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(objs.avg, top1.avg))
    bar.update()
    # if step % args.report_freq == 0:
    #   logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  bar.close()
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion,device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  latency_flag = 0
  start = tm.time()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if latency_flag == 0:
        start_latency = tm.time()

      input = Variable(input).to(device)
      target = Variable(target).to(device)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      # if step % args.report_freq == 0:
      #   logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if latency_flag == 0:
        latency = tm.time() - start_latency
        latency_flag = 1 


    elapsed_time = tm.time() - start
    return top1.avg, objs.avg,latency*1000,elapsed_time


if __name__ == '__main__':
  main() 
