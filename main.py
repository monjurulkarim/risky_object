from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from models.model import RiskyObject
from dataloader import MyDataset
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./feat_extract/feature',
                    help='The relative path of dataset.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='The batch size in training process. Default: 10')
parser.add_argument('--h_dim', type=int, default=256,
                    help='hidden dimension of the gru. Default: 256')
parser.add_argument('--x_dim', type=int, default=2048,
                    help='dimension of the resnet output. Default: 2048')

p = parser.parse_args()
# data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
data_path = p.data_path
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_data = MyDataset(data_path, 'train', toTensor=True, device=device)
test_data = MyDataset(data_path, 'val', toTensor=True, device=device)

traindata_loader = DataLoader(
    dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size,
                             shuffle=False, drop_last=True)

n_frames = 100
fps = 20

model = RiskyObject(p.x_dim, p.h_dim, n_frames, fps).to(device)

print('===Checking the model construction====')
for e in range(2):
    print('Epoch: %d' % (e))
    for i, (batch_xs, batch_det, batch_toas) in tqdm(enumerate(traindata_loader), total=len(traindata_loader)):
        if i == 0:
            losses, all_outputs, all_labels = model(batch_xs, batch_det, batch_toas)
            # print('feature dim:', x.size())
            # print('detection dim:', y.size())
            # print('toas dim:', toa.size())
