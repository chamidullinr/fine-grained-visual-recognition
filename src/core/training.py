import os
import time
from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler

from ..utils import io
from ..utils.progress_log import ProgressBar, CSVProgress, format_seconds
from .metrics import classification_scores, regression_scores


def ReduceLROnPlateau(optimizer, *args, **kwargs):
    return lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)


def OneCycleLR(optimizer, max_lr, steps_per_epoch, epochs, *args, **kwargs):
    return lr_scheduler.OneCycleLR(
        optimizer, max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)


OPTIMIZERS = {
    'adam': Adam,
    'sgd': lambda *a, **k: SGD(*a, momentum=0.9, **k)}
SCHEDULERS = {
    'none': lambda *args, **kwargs: None,
    'reduce_lr_on_plateau': ReduceLROnPlateau,
    'clr': OneCycleLR}


def save_model(model, model_filename, path='.'):
    if not model_filename.endswith('.pth'):
        model_filename += '.pth'
    filepath = os.path.join(path, model_filename)
    io.create_path(filepath)
    torch.save(model.state_dict(), filepath)


def load_model(model, model_filename, path='.'):
    if not model_filename.endswith('.pth'):
        model_filename += '.pth'
    filepath = os.path.join(path, model_filename)
    state_dict = torch.load(filepath, map_location=torch.device('cpu'))
    return model.load_state_dict(state_dict)


def train_epoch(model, dataloader, criterion, optimizer, *, accumulation_steps=1,
                scheduler=None, device=None, pbar=None, return_preds=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pbar is None:
        pbar = ProgressBar().progress_bar

    model.to(device)
    model.train()
    optimizer.zero_grad()
    avg_loss = 0.
    preds_all, targs_all = [], []
    for i, (imgs, targs) in pbar(enumerate(dataloader), total=len(dataloader)):
        imgs = imgs.to(device)
        targs = targs.to(device)

        preds = model(imgs)
        loss = criterion(preds, targs)
        avg_loss += loss.item() / len(dataloader)

        # scale the loss to the mean of the accumulated batch size
        loss = loss / accumulation_steps
        loss.backward()

        # make optimizer step
        if (i - 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        if return_preds:
            preds_all.append(preds.detach().cpu().numpy())
            targs_all.append(targs.detach().cpu().numpy())

    if return_preds:
        preds_all = np.concatenate(preds_all, axis=0)
        targs_all = np.concatenate(targs_all, axis=0)
    else:
        preds_all, targs_all = None, None
    return preds_all, targs_all, avg_loss


def predict(model, dataloader, criterion=None, *, device=None, pbar=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pbar is None:
        pbar = ProgressBar().progress_bar

    model.to(device)
    model.eval()
    avg_loss = 0.
    preds_all, targs_all = [], []
    for i, (imgs, targs) in pbar(enumerate(dataloader), total=len(dataloader)):
        imgs = imgs.to(device)
        targs = targs.to(device)

        with torch.no_grad():
            preds = model(imgs)

        if criterion is not None:
            loss = criterion(preds, targs)
            avg_loss += loss.item() / len(dataloader)

        preds_all.append(preds.cpu().numpy())
        targs_all.append(targs.cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)
    targs_all = np.concatenate(targs_all, axis=0)
    if criterion is None:
        avg_loss = None
    return preds_all, targs_all, avg_loss


class Trainer:
    def __init__(self, model, trainloader, criterion, opt_fn=None, sched_fn=None, *,
                 validloader=None, accumulation_steps=1, path='.',
                 model_filename=None, history_filename=None, metrics=[],
                 regression_task=False, device=None):
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.criterion = criterion
        self.opt_fn = opt_fn
        self.sched_fn = sched_fn
        self.accumulation_steps = accumulation_steps
        self.path = path
        self.model_filename = model_filename
        self.history_filename = history_filename
        self.metrics = metrics
        self.regression_task = regression_task
        self.compute_scores_fn = regression_scores if regression_task else classification_scores
        self.primary_metric = 'mse' if regression_task else 'f1_score'
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def create_optimizer(self, lr, *, wd=0):
        optimizer = self.opt_fn(self.model.parameters(), lr=lr, weight_decay=wd)
        return optimizer

    def create_optimizer_and_scheduler(self, lr, no_epochs=None, *, wd=0):
        optimizer = self.create_optimizer(lr, wd=wd)
        scheduler = self.sched_fn(
            optimizer, max_lr=lr,
            steps_per_epoch=len(self.trainloader), epochs=no_epochs)
        return optimizer, scheduler

    def train_epoch(self, optimizer, scheduler=None, pbar=None, *, return_preds=False):
        return train_epoch(
            self.model, self.trainloader, self.criterion, optimizer, scheduler=scheduler, 
            accumulation_steps=self.accumulation_steps, device=self.device, 
            pbar=pbar, return_preds=return_preds)
    
    def predict(self, dataloader, pbar=None):
        return predict(
            self.model, dataloader, self.criterion,
            device=self.device, pbar=pbar)

    def save_model(self, model_filename=None, *, models_dir='models'):
        if model_filename is None and self.model_filename is None:
            raise ValueError('Param "model_filename" is None')
        elif model_filename is None:
            model_filename = self.model_filename
        path = os.path.join(self.path, models_dir)
        save_model(self.model, model_filename, path)

    def load_model(self, model_filename=None, *, models_dir='models'):
        if model_filename is None and self.model_filename is None:
            raise ValueError('Param "model_filename" is None')
        elif model_filename is None:
            model_filename = self.model_filename
        path = os.path.join(self.path, models_dir)
        load_model(self.model, model_filename, path)

    def train(self, no_epochs, lr=0.01, *, wd=0, optimizer=None, scheduler=None,
              train_scores=False):
        if optimizer is None or scheduler is None:
            optimizer, scheduler = self.create_optimizer_and_scheduler(
                lr, no_epochs, wd=wd)

        # create progress loggers
        progress_bar = ProgressBar()
        csv_progress = CSVProgress(self.history_filename, self.path) \
            if self.history_filename is not None else None

        # apply training loop
        best_loss, best_score = np.inf, 0
        best_state_dict = None
        for epoch in progress_bar.master_bar(range(no_epochs)):
            # apply training and validation on one epoch
            start_time = time.time()
            train_preds, train_targs, train_loss = self.train_epoch(
                optimizer, scheduler, pbar=progress_bar.progress_bar, return_preds=train_scores)
            valid_preds, valid_targs, valid_loss = None, None, None
            if self.validloader is not None:
                valid_preds, valid_targs, valid_loss = self.predict(
                    self.validloader, pbar=progress_bar.progress_bar)
            elapsed_time = time.time() - start_time

            # evaluate metrics
            scores = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss}
            if train_preds is not None and train_targs is not None:
                scores.update({f'train_{k}': v for k, v in
                               self.compute_scores_fn(train_preds, train_targs).items()})
                
            if valid_preds is not None and valid_targs is not None:
                scores.update(self.compute_scores_fn(valid_preds, valid_targs))
                if len(self.metrics) > 0:
                    scores.update({met.__name__: met(valid_preds, valid_targs)
                                   for met in self.metrics})
            scores['time'] = format_seconds(elapsed_time)

            # log progress
            progress_bar.log_epoch_scores(scores)
            if csv_progress is not None:
                csv_progress.log_epoch_scores(scores)

            # update ReduceLROnPlateau scheduler (if available)
            if (valid_loss is not None and scheduler is not None
                and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau)):
                scheduler.step(valid_loss)

            if self.primary_metric in scores:
                # save model checkpoint
                if self.regression_task:
                    is_better = scores[self.primary_metric] < best_score
                else:
                    is_better = scores[self.primary_metric] > best_score
                if is_better:
                    best_score = scores[self.primary_metric]
                    best_state_dict = deepcopy(self.model.state_dict())
                    if self.model_filename is not None:
                        print(f'Epoch {epoch+1} - Save Checkpoint with Best F1 Score: {best_score:.6f}')
                        self.save_model(self.model_filename, models_dir='models')

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

    def lr_find(self, fastai_approach=True):
        """
        Apply learning rate range test. For more details see: https://arxiv.org/abs/1506.01186
        
        If fastai_approach=True then method applies tweaked version used by fastai.
        The approach is faster to run but it produces less precise curves.
        """
        from torch_lr_finder import LRFinder
        
        optimizer = self.create_optimizer(lr=0.1)
        lr_finder = LRFinder(self.model, optimizer, self.criterion, device=self.device)
        if fastai_approach:
            lr_finder.range_test(
                self.trainloader, end_lr=100, num_iter=100,
                accumulation_steps=self.accumulation_steps)
            lr_finder.plot()
        else:
            lr_finder.range_test(
                self.trainloader, val_loader=self.validloader, end_lr=1, num_iter=100,
                step_mode='linear', accumulation_steps=self.accumulation_steps)
            lr_finder.plot(log_lr=False)
        lr_finder.reset()
        return lr_finder
