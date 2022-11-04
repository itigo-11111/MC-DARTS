import os
import sys
import time as tm
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import datetime
import csv
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from collections import namedtuple
import train as training
import visualize as vis

from torch.autograd import Variable
from model_search import Network
from model import NetworkCIFAR as Network2

from architect import Architect
from torchvision import transforms
import random
from tqdm import tqdm
from torchsummary import summary

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./cifar10/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--target_layers', type=int, default=20, help='target total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--gammas_learning_rate', type=float, default=6e-2, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--multigpu', default=True, action='store_true', help='If true, training is not performed.')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower') # 11/4 need check
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--train_mode', action='store_true', default=True, help='use train after search')
parser.add_argument('--val_mode', action='store_true', default=True, help='use validation and check accuracy')
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument("--iteration", default=1, type=int, help="iteration")
parser.add_argument("--id", default=1, type=int, help="sampler id")
parser.add_argument("--limit_param", default=2000000, type=int, help="upper limit of params")
parser.add_argument("--lambda_a", default=0.1, type=float, help="lambda of architecture")
args = parser.parse_args()

def main():
       
  args.img_size = (32, 32)
  Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

  param_set = "dart_param"
  args.save = './test_result2/darts/{}/{}/'.format(param_set,args.limit_param)

  create_dir(args.save)
  # utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10
  args.seed = args.id *1000 + args.iteration
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  # cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  # torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  start_time = tm.time()
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  num_flag = 0

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)
  model = Network(num_flag,device,args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  # model = torch.nn.DataParallel(model)
  model = model.to(device)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  print(split) # 11/4 add

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)


  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  csv_file_path = args.save + 'output.csv'
  with open(csv_file_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['train_acc','train_loss',  'time', 'lr' ,'param','val_acc','val_loss', 'epoch'])
  
  csv_file_path_param = args.save + 'param.csv'
  with open(csv_file_path_param, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['param','latency', 'val_time', 'val_acc' , 'epoch','model','param_size'])
    # writer.writerow(['param','model','param_size'])

  csv_file_path_param_ite = args.save + 'param_ite.csv'
  with open(csv_file_path_param_ite, 'w') as f:
    writer = csv.writer(f)
    # writer.writerow(['param','latency', 'val_time', 'val_acc' , 'epoch','model','param_train'])
    writer.writerow(['param','model','train_loss','train_acc','epoch'])

  best_acc, best_train_acc = 0.0, 0.0
  pytorch_total_params_train, max_step , param_prev = 0, 0, 0
  for epoch in range(1, args.epochs + 1):
    if epoch != 1:
      scheduler.step()
    lr = scheduler.get_last_lr()[0]

    genotype = model.genotype()
    model2 = Network2(36, CIFAR_CLASSES, args.target_layers, args.auxiliary, genotype)
    pytorch_total_params = sum(p.numel() for p in model2.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    
    # training
    train_acc, train_obj, max_step,train_param,genotype = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch,device,pytorch_total_params_train,args.limit_param,num_flag,max_step,args.lambda_a,param_prev,genotype,csv_file_path_param_ite,CIFAR_CLASSES)
    param_prev = train_param
    # logging.info('train_acc %f', train_acc)

    vis.plot(genotype.normal, "normal", epoch,pytorch_total_params_train,param_set,args.limit_param)
    vis.plot(genotype.reduce, "reduce", epoch,pytorch_total_params_train,param_set,args.limit_param)
    
    if train_acc > best_train_acc and args.limit_param > param_prev :
      best_train_acc = train_acc
      if not args.val_mode:
        best_genotype = genotype

    # validation
    if not args.val_mode:
      if args.epochs-epoch<=1:
        valid_acc, valid_obj,latency,val_time = infer(valid_queue, model, criterion,device,num_flag)
      now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
      logging.info('Epoch[{0:03}/{1:03}]  Train Loss:{2:.6f} Train Acc:{3:.6f} Best Train Acc:{4:.6f}  Num of Param:{5} Now {6}'.format(\
                    epoch, args.epochs, train_obj, train_acc, best_train_acc,param_prev, now_time))
      with open(csv_file_path_param, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([pytorch_total_params,genotype])
        
    else:
      valid_acc, valid_obj,latency,val_time = infer(valid_queue, model, criterion,device,num_flag)
      if valid_acc > best_acc:
        best_acc = valid_acc
        best_genotype = genotype

      # pytorch_total_params = sum(p.numel() for p in model.parameters())
      # pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)


      # logging.info('number of parameters : %s (trainable only : %s)',pytorch_total_params,pytorch_total_params_train)
      now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
      logging.info('Epoch[{0:03}/{1:03}]  Train Loss:{2:.6f} Train Acc:{3:.6f} Best Train Acc:{4:.6f} Val Loss:{5:.6f} Val Acc:{6:.6f} Best Val Acc : {7:.6f} Val time:{8:.6f} Latency:{9:.6} Num of Param:{10} Now {11} '.format(\
                    epoch, args.epochs, train_obj, train_acc, best_train_acc, valid_obj, valid_acc, best_acc, val_time, latency, param_prev, now_time))
      with open(csv_file_path_param, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([param_prev,latency,val_time,valid_acc,epoch,genotype, utils.count_parameters_in_MB(model)])

      time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
      with open(csv_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([train_acc, train_obj,  time, optimizer.param_groups[0]['lr'],param_prev,valid_acc,valid_obj, epoch])


    # utils.save(model, os.path.join(args.save, 'weights.pt'))
  end_time = tm.time()
  interval = end_time - start_time
  interval = str("time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
  with open(args.save+'time.txt', 'a') as f:
      writer = csv.writer(f)
      writer.writerow(['time'])
      writer.writerow([interval])
  if args.train_mode:
    # val_mode ON -> use best val_acc  OFF -> use best train_acc
    training.main(best_genotype,args.limit_param)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch,device,pytorch_total_params_train,limit_param,num_flag,max_step,lambda_a,param_prev,genotype,csv_file_path_param_ite,CIFAR_CLASSES):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  param = utils.AvgrageMeter()

  bar = tqdm(desc = "Training", total = len(train_queue), leave = False)
  arcstep_flag = 0
  i = valid_queue
  train_param = pytorch_total_params_train

  for step, (input, target) in enumerate(train_queue):
    model.train()

    n = input.size(0)
    input = Variable(input, requires_grad=False).to(device)
    target = Variable(target, requires_grad=False).to(device)

    # get a random minibatch from the search queue with replacement
    # input_search, target_search = next(i)
    try:
     input_search, target_search = next(valid_queue_iter)
    except:
     valid_queue_iter = iter(valid_queue)
     input_search, target_search = next(valid_queue_iter)
    
    input_search = Variable(input_search, requires_grad=False).to(device)
    target_search = Variable(target_search, requires_grad=False).to(device)

    # epochs >= 15 -> 2 -> 15
    if epoch>=2:
      # if arcstep_flag == 0:
      model_copy = model
      genotype = model.genotype()
      # model2 = Network2(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
      model2 = Network2(36, CIFAR_CLASSES, args.target_layers, args.auxiliary, genotype)
      pytorch_total_params_train = sum(p.numel() for p in model2.parameters() if p.requires_grad)

      if pytorch_total_params_train > limit_param :
        # if step == max_step:
        # if step == 0:
        num_flag = 1
        train_param, genotype = architect.param_step(input, target, input_search, target_search, lr, optimizer,pytorch_total_params_train,step,limit_param, num_flag,lambda_a,param_prev)
        # print(model.arch_parameters())
        # arcstep_flag = 1
        # top1_val.update(acc_x.data.item(), n)
        # print(train_param,param_prev)
        param_prev = train_param

      else:
        model = model_copy
        architect.step(input, target, input_search, target_search, lr, optimizer,pytorch_total_params_train,step,limit_param,num_flag, unrolled=args.unrolled)
        # top1_val.update(acc_x.data.item(), n)
        param_prev = pytorch_total_params_train


    num_flag = 0
    optimizer.zero_grad()
    logits = model(input,num_flag)
    loss = criterion(logits, target)


    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if epoch == 1:
      param_prev = pytorch_total_params_train

    # bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f} Val Acc: {1:.6f}".format(objs.avg, top1.avg, top1_val.avg))
    bar.set_description("Loss: {0:.6f}, Acc: {1:.6f} param:{2}".format(objs.avg, top1.avg, param_prev))
    bar.update()

    with open(csv_file_path_param_ite, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([param_prev,genotype,objs.avg,top1.avg,epoch])

    # if step % args.report_freq == 0:
      # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  bar.close()
  if epoch == 1:
    max_step = step
  return top1.avg, objs.avg, max_step, param_prev, genotype


def infer(valid_queue, model, criterion,device,num_flag):
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

      #input = input.cuda()
      #target = target.cuda(non_blocking=True)
      input = Variable(input).to(device)
      target = Variable(target).to(device)
      logits = model(input,num_flag)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if latency_flag == 0:
        latency = tm.time() - start_latency
        latency_flag = 1      

      # if step % args.report_freq == 0:
      #   logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    elapsed_time = tm.time() - start
    return top1.avg, objs.avg,latency*1000,elapsed_time


if __name__ == '__main__':
  main() 
