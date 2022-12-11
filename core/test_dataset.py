from config import get_config, print_usage
config, unparsed = get_config()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
import torch.utils.data
import sys
from data import collate_fn, CorrespondencesDataset
# from oan import OANet as Model
# from deepVFC_v4 import deep_VFC as Model
from train import train
from test import test


if __name__ == "__main__":
    train_dataset = CorrespondencesDataset(config.data_tr, config.data_tr_reg, config)
    train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.train_batch_size, shuffle=False,
                num_workers=0, pin_memory=False, collate_fn=collate_fn) # 16
    train_loader_iter = iter(train_dataset)
    train_data = next(train_loader_iter)
    print(train_data)
    print(train_data['xs'].shape)
    print(train_data['xs_reg'].shape)

    print(train_data['ys'])
    print(train_data['ys_reg'])

    print(train_data['R'])
    print(train_data['R_reg'])