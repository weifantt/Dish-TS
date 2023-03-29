import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.affine: # args.affine: use affine layers or not
            self.gamma = nn.Parameter(torch.ones(args.n_series)) # args.n_series: number of series
            self.beta = nn.Parameter(torch.zeros(args.n_series))
        else:
            self.gamma, self.beta = 1, 0
    
    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'forward':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == 'inverse':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        self.avg = torch.mean(batch_x, axis=1, keepdim=True).detach() # b*1*d
        self.var = torch.var(batch_x, axis=1, keepdim=True).detach()  # b*1*d

    def forward_process(self, batch_input):
        temp = (batch_input - self.avg)/torch.sqrt(self.var + 1e-8)
        return temp.mul(self.gamma) + self.beta

    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.var + 1e-8) + self.avg