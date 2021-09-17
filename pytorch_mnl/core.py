# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_core.ipynb (unless otherwise specified).

__all__ = ['prepare_data', 'train_valid_split', 'LinearMNL', 'DataLoaders', 'Learner']

# Cell
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fastprogress import progress_bar, master_bar
from fastcore.all import *

# Cell
def prepare_data(data, x_cols=None, target_col=None):
    "This is far from optimal, as we shu=ould be reading values lazily"
    target_col = ifnone(target_col, list(data.columns)[-1])
    x_cols = [col for col in ifnone(x_cols, list(data.columns)) if col!=target_col]
    X_numpy = data.loc[:, x_cols].values
    target_map = {
        val: index for index, val in enumerate(data.loc[:,target_col].unique())
    }
    y_numpy = data.loc[:,target_col].map(target_map).values

    X = torch.tensor(X_numpy, dtype=torch.float32)
    y = torch.tensor(y_numpy)

    return X, y

# Cell
def train_valid_split(X, y, pct=0.2, shuffle=True):
    assert len(X) == len(y), "X and y don't have the same number of elements"
    indices = range_of(X)
    if shuffle:
        random.shuffle(indices)
    n = len(X)
    n_train = int(n * (1-0.2))
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_valid, y_valid = X[indices[n_train:]], y[indices[n_train:]]
    return X_train, y_train, X_valid, y_valid

# Cell
class LinearMNL(nn.Module):

    def __init__(self, in_dim=4, out_dim=5, bias=False):
        super().__init__()
        store_attr()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)

# Cell
class DataLoaders:
    """
    A class to store dataloaders (train/valid/test....)"""
    def __init__(self, train_dl, valid_dl=None):
        store_attr()

    def one_batch(self, dl=None):
        dl = ifnone(dl, self.train_dl)
        return next(iter(dl))

    @delegates(DataLoader, but='batch_size')
    @classmethod
    def from_datasets(cls, train_ds, valid_ds=None, batch_size=1, **kwargs):
        train_dl = DataLoader(train_ds, batch_size=batch_size, **kwargs)
        if valid_ds is not None:
            valid_dl = DataLoader(valid_ds, batch_size=2*batch_size, **kwargs)
        else:
            valid_dl = None
        return cls(train_dl, valid_dl)

    @delegates(DataLoader, but='batch_size')
    @classmethod
    def from_Xy(cls, X, y, pct=None, batch_size=1, **kwargs):
        if pct is not None:
            X_train, y_train, X_valid, y_valid = train_valid_split(X, y, pct)
        else:
            X_train, y_train, X_valid, y_valid = X, y, None, None
        train_ds = TensorDataset(X_train, y_train)
        if X_valid is not None:
            valid_ds = TensorDataset(X_valid, y_valid)
        else:
            valid_ds = None
        return cls.from_datasets(train_ds, valid_ds, batch_size)


# Cell
class Learner:
    "A wrapper around dls, model and optimizer"
    def __init__(self, dls, model, loss_func=torch.nn.CrossEntropyLoss()):
        store_attr()


    def train_one_epoch(self):
        accum_loss = 0.
        for batch, (x, y) in enumerate(self.dls.train_dl):
            pred = self.model(x)  # 1
            loss = self.loss_func(pred, y)

            #backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            accum_loss += loss.item()
        return accum_loss

    def validate(self, dl=None):
        dl = ifnone(dl, self.dls.valid_dl)
        if (dl is None):
            return 'No validation data'
        val_loss, accu = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(dl):
                pred = self.model(x)
                val_loss += self.loss_func(pred, y).item()
                accu += (pred.argmax(1) == y).type(torch.float).sum().item()
        return val_loss, accu / len(dl.dataset)

    def fit(self, n_epochs=10, lr=0.01, wd=0.01):

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd
        )

        for epoch in progress_bar(range_of(n_epochs), leave=False):
            loss = self.train_one_epoch()
            val_loss, accuracy = self.validate()
            print(f'epoch = {epoch:3.0f}, train_loss = {loss:.3f}, val_loss = {val_loss:.3f}, accuracy = {accuracy:.2f}')