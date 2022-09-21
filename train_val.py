# -*- coding: utf-8 -*-

import time
import sys
import numpy as np
import torch
import torch.nn as nn
import csv
import time

from tqdm import tqdm


def accuracy(output, target):
    pred = output.data.max(1, keepdim = True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()
    return acc
    
# Training
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    train_acc, train_loss, n_train = 0, 0, 0
    model.train()

    bar = tqdm(desc = "Training", total = len(train_loader), leave = False)
    
    for i_batch, sample_batched in enumerate(train_loader):
            data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc += accuracy(output, target)
            train_loss += loss.item() * target.size(0)
            n_train += target.size(0)

            bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(train_loss / n_train, float(train_acc) / n_train))
            bar.update()
    bar.close()
    return train_loss / n_train, float(train_acc) / n_train

def val(args, model, device, test_loader, criterion):
    model.eval()
    val_acc, val_loss, n_val, latency_flag = 0, 0, 0, 0
    start = time.time()
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            if latency_flag == 0:
                start_latency = time.time()
            data, target = sample_batched["image"].to(device), sample_batched["label"].to(device)
            output = model(data)
            loss = criterion(output, target)
            val_acc += accuracy(output, target)
            val_loss += loss.item() * target.size(0)
            n_val += target.size(0)
            if latency_flag == 0:
                latency = time.time() - start_latency
                latency_flag = 1 
    
    elapsed_time = time.time() - start

    return val_loss / n_val, float(val_acc) / n_val ,latency*1000, elapsed_time
