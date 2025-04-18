import argparse
import os
import random
import time

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, grad_scaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from .mixup import mixup_criterion, mixup_data
from .optim_utils import get_optimizer
from .utils import (
    AverageMeter,
    get_model_from_configs,
    Logger,
    ModelLoader,
    ModelUtils,
)


class Trainer:
    def __init__(self, args, dset_loaders, ckpt_saver, logger, neptune_logger):
        self.args = args
        self.model = get_model_from_configs(self.args)
        self.ckpt_saver = ckpt_saver
        self.logger = logger
        self.logger.info(f'Model Name {self.model.__class__}')

        total_params = ModelUtils.count_parameters(self.model)
        self.logger.info(f"Total parameters: {total_params:,}")
            
        self.dset_loaders = dset_loaders
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(self.args.optimizer, optim_policies = self.model.parameters(), lr = self.args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = self.args.epochs, eta_min = 0.)
        
        if args.neptune_logging:  
            self.neptune_logger = neptune_logger
            if self.args.cal_model_size :
                self.neptune_logger['parameters/params'] = self.args.params
                self.neptune_logger['parameters/gflops'] = self.args.gflops
            
        if self.args.model_path:
            assert self.args.model_path.endswith('.tar') and os.path.isfile(self.args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(self.args.model_path)
            # resume from checkpoint
            if self.args.init_epoch > 0:
                self.model, self.optimizer, self.scheduler, epoch_idx, ckpt_dict = ModelLoader.load_model(
                    self.args.model_path, 
                    self.model, 
                    self.optimizer, 
                    self.scheduler
                )
                self.ckpt_saver.set_best_from_ckpt(ckpt_dict)
                self.logger.info('Model and states have been successfully loaded from {}'.format(self.args.model_path))
                self.args.init_epoch = epoch_idx
            # init from trained model
            else:
                self.model = ModelLoader.load_model(self.args.model_path, self.model)
                self.logger.info('Model has been successfully loaded from {}'.format(self.args.model_path))
                
        self.epoch = self.args.init_epoch
        
    def train(self):
        while self.epoch < self.args.epochs:
            train_loss, train_acc = self.run_epoch(self.dset_loaders['train'])
            test_loss, test_acc = self.evaluate(self.dset_loaders['test'])
            self.scheduler.step()
            
            # -- save checkpoint
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict' : self.scheduler.state_dict()
            }
            
            self.ckpt_saver.save(save_dict, test_acc)
            
            if self.args.neptune_logging:
                self.neptune_logger['logs/train loss'].log(train_loss)
                self.neptune_logger['logs/train acc'].log(train_acc)
                self.neptune_logger['logs/test loss'].log(test_loss)
                self.neptune_logger['logs/test acc'].log(test_acc)
                
            self.epoch += 1

    def run_epoch(self, dset_loader):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        inter_time = AverageMeter()

        lr = ModelUtils.show_lr(self.optimizer)

        self.logger.info('-' * 10)
        self.logger.info('Epoch {}/{}'.format(self.epoch, self.args.epochs - 1))
        self.logger.info('Current learning rate: {}'.format(lr))

        self.model.train()
        running_loss = 0.
        running_corrects = 0.
        running_all = 0.

        scaler = grad_scaler.GradScaler()
        end = time.time()
        interval = int(len(dset_loader)/10)
        for batch_idx, (input, lengths, labels) in enumerate(dset_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input, labels_a, labels_b, lam = mixup_data(input, labels, self.args.alpha)
            labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

            self.optimizer.zero_grad()
            loss_func = mixup_criterion(labels_a, labels_b, lam)
            
            if self.args.amp:
                with autocast():
                    logits = self.model(input.unsqueeze(1).cuda(), lengths=lengths)
                    loss = loss_func(self.criterion, logits)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
            else:
                logits = self.model(input.unsqueeze(1).cuda(), lengths=lengths)
                loss = loss_func(self.criterion, logits)
                loss.backward()
                self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            inter_time.update(time.time() - end)
            end = time.time()
            # -- compute running performance
            _, predicted = torch.max(F.softmax(logits, dim=1), dim=1)
            running_loss += loss.item()*input.size(0)
            running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
            running_all += input.size(0)
            
            # -- log intermediate results
            if batch_idx % interval == 0 or (batch_idx == len(dset_loader)-1):
                if batch_idx / interval != 10:
                    Logger.update_logger_batch(
                        self.args, 
                        self.logger, 
                        dset_loader, 
                        batch_idx, 
                        running_loss, 
                        running_corrects, 
                        running_all, 
                        batch_time, 
                        data_time, 
                        inter_time
                    )
            
        return running_loss / running_all, running_corrects / running_all
    
    def evaluate(self, dset_loader):
        self.model.eval()

        running_loss = 0.
        running_corrects = 0.

        with torch.no_grad():
            for batch_idx, (input, lengths, labels) in enumerate(tqdm(dset_loader)):
                logits = self.model(input.unsqueeze(1).cuda(), lengths=lengths)
                _, preds = torch.max(F.softmax(logits, dim=1), dim=1)
                running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()
                loss = self.criterion(logits, labels.cuda())
                running_loss += loss.item() * input.size(0)
                
        val_loss = running_loss / len(dset_loader.dataset) 
        val_acc = running_corrects / len(dset_loader.dataset)
        
        self.logger.info('Test set evaluation\nLoss: {:.4f},  Acc:{:.4f}'.format(val_loss, val_acc))
        
        return val_loss, val_acc 