import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
import copy
from utils import setup_seed
from utils.dataset import TSForecastDataset
from utils.earlystop import EarlyStopping
from utils.metric import get_metrics
from backbones import Autoformer, Informer, Transformer
from DishTS import DishTS
from REVIN import RevIN
from Model import Model


def update_args_from_model_params(args, n_series):
    model_params = {
        "embed_type":3,'factor':3,"output_attention":False,'d_model':512,'embed':'timeF','freq':'h',# Informer
        'dropout':0.05, 'n_heads':8,'d_ff':2048, 'moving_avg':25,'activation':'gelu','e_layers':2, # Autoformer
        'd_layers':1, 'distil':True, "enc_in":n_series,  "dec_in":n_series, 'c_out':n_series, 'n_series':n_series,
        }
    model_params.update(vars(args))
    args = argparse.Namespace(**model_params)
    return args


parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--output_file', type=str, default='forecast.csv')
# forecast 
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
# data
parser.add_argument('--data', type=str, default='ETTm2')
parser.add_argument('--features', type=str, default='M')
# forecast model
parser.add_argument('--model', type=str, default='Transformer')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--gpu', type=int, default=0)
# shift model
parser.add_argument('--norm', type=str, default='none') # none, revin, dishts
parser.add_argument('--affine', type=str, default=1) # revin
parser.add_argument('--dish_init', type=str, default='standard') # standard, 'avg' or 'uniform'
args = parser.parse_args()


# prepare parameters
setup_seed(args.seed)
DATA = args.data; GPU = args.gpu; MODEL = args.model; T=False
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
args.batch_size = 64 if args.pred_len > 168 else args.batch_size


# prepare dataset
val_ratio, test_ratio = 0.1, 0.2
train_dataset = TSForecastDataset(data_path=f'./dataset/{DATA}.csv', flag='train', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio))
val_dataset = TSForecastDataset(data_path=f'./dataset/{DATA}.csv', flag='val', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio))
test_dataset = TSForecastDataset(data_path=f'./dataset/{DATA}.csv', flag='test', size=(args.seq_len, args.label_len, args.pred_len), split=(val_ratio, test_ratio))

# set forecast dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

n_series = train_dataset.N
args = update_args_from_model_params(args, n_series)
# set forecast models
model_dict = {'Autoformer': Autoformer, 'Transformer': Transformer, 'Informer': Informer}
# set norm models
norm_dict = {'revin': RevIN, 'dishts': DishTS}
forecast_model = model_dict[MODEL].Model(args)
norm_model = None if args.norm == 'none' else norm_dict[args.norm](args)


unify_model = Model(args, forecast_model, norm_model).to(device)
optimizer = optim.Adam(unify_model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=args.patience, verbose=True, dump=False)
loss_fn = nn.MSELoss()


def get_init_batch(batch):
    batch_x,  batch_y = batch
    batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device).float()
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
    return batch_x,  batch_y, dec_inp


max_epochs = 100
for epoch in range(max_epochs):
    train_losses, val_losses = [], []
    # train
    unify_model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        # encoder input: b*seq_len*dim, b*(label_len+pred_len)*dim
        batch_x, batch_y, dec_inp = get_init_batch(batch)
        forecast = unify_model(batch_x, dec_inp)
        loss = loss_fn(forecast, batch_y[:, -args.pred_len:, :])
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    # validate
    with torch.no_grad():
        unify_model.eval()
        for batch in val_loader:
            batch_x, batch_y, dec_inp = get_init_batch(batch)
            forecast = unify_model(batch_x, dec_inp)
            loss = loss_fn(forecast, batch_y[:, -args.pred_len:, :])
            val_losses.append(loss.item())
    # early stop
    print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}'.format(epoch, np.mean(train_losses), np.mean(val_losses)))
    early_stopping(np.mean(val_losses), unify_model, epoch)
    if early_stopping.early_stop:
        print("Early stopping with best_score:{}".format(-early_stopping.best_score))
        break
    if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
        break

# test
model = early_stopping.best_model
model.eval()
preds, trues = None, None
with torch.no_grad():
    for batch in test_loader:
        batch_x, batch_y, dec_inp = get_init_batch(batch)
        forecast = unify_model(batch_x, dec_inp)
        # concat
        pred = forecast.detach().cpu().numpy()
        true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
        preds = pred if preds is None else np.concatenate((preds, pred), 0)
        trues = true if trues is None else np.concatenate((trues, true), 0)


mae, mse, rmse, mape, mspe = get_metrics(preds, trues)

df = pd.DataFrame([[DATA, MODEL, args.norm, args.seed, args.seq_len, args.pred_len, mse, mae, rmse]])
print(df)


