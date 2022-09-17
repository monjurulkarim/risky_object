from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from models.model import RiskyObject
from dataloader import MyDataset
import argparse
import os
import numpy as np
import sys


def init_risky_object_model(model_file, x_dim, h_dim, n_frames, fps):
    # building model
    model = RiskyObject(x_dim, h_dim, n_frames, fps)

    model = model.to(device=device)
    model.eval()
    # load check point
    model, _, _ = _load_checkpoint(model, filename=model_file)
    return model


def load_input_data(feature_file, device=torch.device('cuda')):
    data = np.load(feature_file)
    features = data['feature']  # 100 x 31 x 2048
    toa = [data['toa']+0]  # 1
    detection = data['detection']  # labels : data['detection'][:,:,5] --> 100 x 30
    flow = data['flow_feat']

    features = torch.Tensor(features).to(device)  # 50 x 20 x 4096
    detection = torch.Tensor(detection).to(device)
    toa = torch.Tensor(toa).to(device)
    flow = torch.Tensor(flow).to(device)
    return features, detection, toa, flow


def _load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        # print('Checkpoint loaded')
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, help="the path to the model file.",
                        default="checkpoints/snapshot_both_attention_bbox_flow/best_ap.pth")
    parser.add_argument('--h_dim', type=int, default=256,
                        help='hidden dimension of the gru. Default: 256')
    parser.add_argument('--x_dim', type=int, default=2048,
                        help='dimension of the resnet output. Default: 2048')
    parser.add_argument('--feature_dir', type=str,
                        help="the path to the feature file.", default="feat_extract/feature/rgb_flow_1000/val")
    parser.add_argument('--output_dir', type=str,
                        help="the path to the feature file.", default="checkpoints/output/snapshot_both_attention_bbox_flow")
    p = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # feature_files = glob.glob(os.path.join(p.feature_dir, '*.npz'))
    feature_files = os.listdir(p.feature_dir)
# angle
# others
# Front-to-rear
# sidewipe
# Front-to-front
    for file in feature_files:
        feature_file = os.path.join(p.feature_dir, file)
        features, detection, toa, flow = load_input_data(feature_file, device=device)
        features = features.unsqueeze(0)
        detection = detection.unsqueeze(0)
        flow = flow.unsqueeze(0)

        model = init_risky_object_model(p.ckpt_file, p.x_dim, p.h_dim, 100, 20)

        # creating an output directory to save the results
        if not os.path.exists(p.output_dir):
            os.makedirs(p.output_dir)

        with torch.no_grad():
            losses, all_outputs, all_labels = model(features, detection, toa, flow)
            file_name = os.path.join(p.output_dir, file)
            np.savez_compressed(file_name, output=all_outputs, label=all_labels)
            print(f'----Completed----{file}')
