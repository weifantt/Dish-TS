import os
import copy
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dump=False, path='./checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.dump = dump
        self.val_loss_min = np.Inf
        self.delta = 0
        self.path = path
        self.trace_func = print

    def __call__(self, val_loss, model, epoch=0):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = self.save_checkpoint(val_loss, model) if self.dump else copy.deepcopy(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = self.save_checkpoint(val_loss, model) if self.dump else copy.deepcopy(model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(model, self.path+'/checkpoint.pt')
        return None