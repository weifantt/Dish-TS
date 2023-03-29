import numpy as np

# metrics
def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    true = true + 1e-8
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    true = true + 1e-8
    return np.mean(np.square((pred - true) / true))


def get_metrics(pred, true):
    mae = round(MAE(pred, true),5)
    mse = round(MSE(pred, true),5)
    rmse = round(RMSE(pred, true),5)
    mape = round(MAPE(pred, true),5)
    mspe = round(MSPE(pred, true),5)
    return mae, mse, rmse, mape, mspe
