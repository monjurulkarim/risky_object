from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path, phase, toTensor=False,  device=torch.device('cuda')):
        self.data_path = data_path
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.n_frames = 100  # -->
        self.fps = 20
        self.dim_feature = 2048
        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)
        # print(self.files_list)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s" % (filepath)
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['feature']  # 100 x 31 x 2048
            toa = [data['toa']+0]  # 1
            detection = data['detection']  # labels : data['detection'][:,:,5] --> 100 x 30
            # track_id : data['detection'][:,:,0] --> 100 x 30

        except:
            raise IOError('Load data error! File: %s' % (data_file))

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)  # 50 x 20 x 4096
            detection = torch.Tensor(detection).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        return features, detection, toa


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./feat_extract/feature',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The batch size in training process. Default: 10')

    p = parser.parse_args()
    # data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    data_path = p.data_path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = MyDataset(data_path, 'train', toTensor=True, device=device)
    test_data = MyDataset(data_path, 'val', toTensor=True, device=device)

    traindata_loader = DataLoader(
        dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(
        dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    print('===Checking the dataloader====')
    for e in range(2):
        print('Epoch: %d' % (e))
        for i, (batch_xs, batch_det, batch_toas) in tqdm(enumerate(traindata_loader), total=len(traindata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('detection dim:', batch_det.size())
                print('toas dim:', batch_toas.size())
                print('toas batch : ', batch_toas)

    for e in range(2):
        print('Epoch: %d' % (e))
        for i, (batch_xs, batch_det, batch_toas) in tqdm(enumerate(testdata_loader), total=len(testdata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('detection dim:', batch_det.size())
                print('toas dim:', batch_toas.size())
