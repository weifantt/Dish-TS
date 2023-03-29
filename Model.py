import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, forecast_model, norm_model):
        super().__init__()
        self.args = args
        self.fm = forecast_model
        self.nm = norm_model

    def forward(self, batch_x, dec_inp):
        if self.nm is not None:
            batch_x, dec_inp = self.nm(batch_x, 'forward', dec_inp)
        
        if 'former' in self.args.model:
            forecast = self.fm(batch_x, None, dec_inp, None)
        else:
            forecast = self.fm(batch_x)

        if self.nm is not None:
            forecast = self.nm(forecast, 'inverse')
        
        return forecast
