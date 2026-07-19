from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import os
import torchnet as tnt
import spacetorch.variants.assets.simclr.utils as utils
import PIL
import pickle
from tqdm import tqdm
import time

from spacetorch.variants.assets.simclr.algorithms import Algorithm
from pdb import set_trace as breakpoint


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ClassificationModel(Algorithm):
    def __init__(self, opt, model_wrapper=None, args=None, positions=None, layernames=None):
        Algorithm.__init__(self, opt, model_wrapper=model_wrapper, args=args, positions=positions, layernames=layernames)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        dataX = self.tensors['dataX']
        batch_load_time = time.time() - start
        #********************************************************

        #********************************************************
        start = time.time()
        if do_train: # zero the gradients
            self.optimizers['model'].zero_grad()
        #********************************************************

        #***************** SET TORCH VARIABLES ******************
        dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train))
        #********************************************************

        #************ FORWARD THROUGH NET ***********************
        proj_var, spatial_outputs = self.networks['model'](dataX_var)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        record = {}
        loss_task = self.criterions['loss'](proj_var)
        loss_total = self.networks['model'].get_joint_loss((loss_task, spatial_outputs))
        record['prec1'] = 0.
        record['loss'] = loss_total.data.item()
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['model'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record
