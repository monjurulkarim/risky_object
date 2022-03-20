import pandas as pd
import numpy as np
from natsort import natsorted
import glob
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.model import FeatureExtractor
from PIL import Image

device = ("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose(
#         [   transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5 ,0.5)),
#         ]
#     )
transform = transforms.Compose(
        [   transforms.Resize(224),
            transforms.ToTensor(),

        ]
    )
to_pil_image = transforms.ToPILImage()

# def featureExt(image):
#     feat_extractor = 'resnet50'
#     h_dim = 256
#     # image = Image.open(img_path)
#     image = transform(image)
#     image = torch.unsqueeze(image,0).float().to(device=device)
#     extractor = FeatureExtractor(device, h_dim+h_dim, feat_extractor).to(device=device)
#     extractor.eval()
#     with torch.no_grad():
#         feat = extractor(image)
#     return feat # 1x 512

def featureExt(image):
    # image = Image.open(img_path)
    image = transform(image) # 3 x 224 x 336
    # print('shape image : ', image.shape)
    image = torch.unsqueeze(image,0).float().to(device=device)
    extractor = FeatureExtractor().to(device=device)
    extractor.eval()
    with torch.no_grad():
        feat = extractor(image) # 1 x 2048 x 1 x 1
        feat = torch.squeeze(feat,2) # 1 x 2048 x 1
        feat = torch.squeeze(feat,2) # 1 x 2048
    return feat # 1x 2048


def bbox_to_imroi(video_frame, bbox):
    image = Image.open(video_frame)
    image = transform(image)
    imroi = image[:,bbox[1]:bbox[3],bbox[0]:bbox[2]]
    return imroi



def main():
    csv_files = natsorted(glob.glob(os.path.join('csv_files', '*.csv')))
    # print(csv_files)
    scaling = 3.21

    for i in csv_files:
        detections =  np.zeros((100,30,6),dtype=np.float32)
        feature = np.zeros((100,31,2048),dtype=np.float32)
        # feature = torch.zeros(100,35,512)
        vid_id_toa = i.split('.')[0].split('/')[-1]
        vid_id = vid_id_toa.split('-')[0]
        toa = int(vid_id_toa.split('-')[-1])
        video = 'data/'+vid_id+'/'
        video_frames = natsorted(glob.glob(os.path.join(video, '*.jpg')))
        print('========',vid_id,'===================')
        for j in range(len(video_frames)):
            img_path = video_frames[j]
            image = Image.open(img_path)

            feat = featureExt(image)
            # feat = feat.detach().numpy()
            feat = feat.cpu().numpy() if feat.is_cuda else feat.detach().numpy()
            feature[j,0,:]= feat

        df =pd.read_csv(i)
        df = df.reset_index()
        for index,row in df.iterrows():
            # print(row)
            detections[row['frame'],row['object_num']] = row['track_id'],row['y1'],row['x1'],row['y2'],row['x2'],row['label']
            f_num = row['frame']
            video_frame = video_frames[f_num]
            if row['object_num'] > 29:
                continue
            else:
                bbox = [int((row['y1'])//scaling), int((row['x1'])//scaling), int((row['y2'])//scaling), int((row['x2'])//scaling)]
                # print(bbox)
                object_img = bbox_to_imroi(video_frame,bbox)
                object_img = to_pil_image(object_img)
                object_feat = featureExt(object_img) # 1x512
                # object_feat = object_feat.detach().numpy()
                object_feat = object_feat.cpu().numpy() if object_feat.is_cuda else object_feat.detach().numpy()

                feature[f_num,row['object_num']+1,:] = object_feat

        save_file = 'feature/'+ vid_id+'.npz'
        np.savez_compressed(save_file, feature = feature, detection = detections, vid_id= vid_id, toa=toa)



if __name__ == '__main__':
    main()
