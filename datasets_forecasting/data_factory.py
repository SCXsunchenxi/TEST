from datasets_forecasting.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute
from torch.utils.data import DataLoader
import pandas as pd
from einops import rearrange

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
}


def data_provider(config, flag, drop_last_test=True, train_all=False):
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1
    percent = config['percent']
    max_len = config['max_len']

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = config['batch_size']
        freq = config['freq']
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = config['freq']
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = config['batch_size']
        freq = config['freq']
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = config['batch_size']
        freq = config['freq']

    data_set = Data(
        root_path=config['root_path'],
        data_path=config['data_path'],
        flag=flag,
        size=[config['seq_len'], config['label_len'], config['pred_len']],
        features=config['features'],
        target=config['target'],
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        max_len=max_len,
        train_all=train_all
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=config['num_workers'],
        drop_last=drop_last)
    return data_set, data_loader

def load_forecasting_dataset(data_file_path):

    dataset = pd.read_csv(data_file_path, index_col=0).to_numpy()
    dataset=rearrange(dataset, 'l n ->n l')
    dataset=rearrange(dataset, 'n (m l) ->n m l',m=1)
    return dataset

