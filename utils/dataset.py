import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TSForecastDataset(Dataset):
    def __init__(self, data_path='ETTh1.csv', flag='train', size=(96,48,96), split=(0.1, 0.2)):
        self.seq_len, self.label_len, self.pred_len= size[0], size[1], size[2]
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.ratio_vali, self.ratio_test = split[0], split[1]
        self.__read_data__(data_path)

    def __read_data__(self, data_path):
        df_raw = pd.read_csv(data_path)
        # get cols, default all variables without date
        cols = list(df_raw.columns); cols.remove('date')
        # get rows
        num_vali, num_test = int(len(df_raw)*self.ratio_vali), int(len(df_raw)*self.ratio_test)
        num_train = len(df_raw) - num_vali - num_test
        border1s = [0, num_train, num_train+num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        left_border = border1s[self.set_type]; right_border = border2s[self.set_type]
        # get data
        self.data_x = df_raw[cols][left_border:right_border].values # input
        self.data_y = df_raw[cols][left_border:right_border].values # output
        self.N = self.data_x.shape[1] # number of series
        self.train_data = df_raw[cols][border1s[0]:border2s[0]].values # train_data

    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1