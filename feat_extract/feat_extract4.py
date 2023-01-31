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
import argparse
import logging
device = ("cuda:1" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [transforms.Resize(224),
        transforms.ToTensor(),
     ]
)
to_pil_image = transforms.ToPILImage()


def log_information(vid_id, video_frame, track_id, e):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('Error_log.log')
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    logger.info(
        f"Error: {e}------Video_id-{vid_id}--video_frame: {video_frame} at tracking_id {track_id}")
    return


def get_args():
    parser = argparse.ArgumentParser()
    # csv files of each videos, contain tracking, bbox, etc information (generated with det_csv.py)
    parser.add_argument("--csv_dir", default="hevi_csv_train/")
    # contains the folderwise video frames
    parser.add_argument("--data_dir", default="data/hevi_flow_images_train/")
    # extracted resnet50 features will be stored here in *.npz format
    parser.add_argument("--feature_dir", default="feature/hevi_feat_train/")
    args = parser.parse_args()
    return args


def featureExt(image):
    # image = Image.open(img_path)
    image = transform(image)  # 3 x 224 x 224 (for the frame, for object it varies)
    # print('shape image : ', image.shape)
    image = torch.unsqueeze(image, 0).float().to(device=device)
    # print(image.shape)
    extractor = FeatureExtractor().to(device=device)
    extractor.eval()
    with torch.no_grad():
        feat = extractor(image)  # 1 x 2048 x 1 x 1
        feat = torch.squeeze(feat, 2)  # 1 x 2048 x 1
        feat = torch.squeeze(feat, 2)  # 1 x 2048
    return feat  # 1x 2048


def bbox_to_imroi(video_frame, bbox):
    image = Image.open(video_frame)
    image = transform(image)
    imroi = image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]  # (x1:x2, y1:y2)
    return imroi


def main():

    args = get_args()
    csv_dir = args.csv_dir
    data_dir = args.data_dir
    feature_dir = args.feature_dir
    csv_files = natsorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    csv_files = csv_files[363:]

    # print(csv_files)
    # scaling_w = 4.82  # 1080/224
    # scaling_h = 3.21  # 720/224
    scaling_w = 8.57  # 1920/224
    scaling_h = 5.36  # 1200/224

    for i in csv_files:
        # 100 frames, 30 maximum objects, 6: 1-> track_id, (2,3,4,5)-> (y1,x1;y2,x2), 6-> object serial number in each frame
        detections = np.zeros((100, 30, 6), dtype=np.float32)
        # 100 frames, 1 global frame-level feature + 30 object level feature , resnet50 feat dimension 2048
        feature = np.zeros((100, 31, 2048), dtype=np.float32)

        vid_id_toa = i.split('.')[0].split('/')[-1]
        vid_id = vid_id_toa.split('-')[0]
        toa = int(vid_id_toa.split('-')[-1])  # time of accident
        video = data_dir+vid_id+'/'
        video_frames = natsorted(glob.glob(os.path.join(video, '*.jpg')))
        if len(video_frames) == 0:
            print('Frames not found in ', vid_id)
            continue
        print('========', vid_id, '===================')
        for j in range(len(video_frames)):
            img_path = video_frames[j]
            image = Image.open(img_path)

            feat = featureExt(image)
            # feat = feat.detach().numpy()
            feat = feat.cpu().numpy() if feat.is_cuda else feat.detach().numpy()
            feature[j, 0, :] = feat
        # print('frame_level feature extraction finished.')

        df = pd.read_csv(i)
        df = df.reset_index()
        for index, row in df.iterrows():
            # print(row)
            try:
                detections[row['frame'], row['object_num']
                           ] = row['track_id'], row['y1'], row['x1'], row['y2'], row['x2'], row['label']
            except IndexError:
                print('=====Index error=====', vid_id)
                continue
            f_num = row['frame']
            video_frame = video_frames[f_num]
            if row['object_num'] > 29:
                continue
            else:
                bbox = [int((row['y1'])//scaling_w), int((row['x1'])//scaling_h), int((row['y2']) //
                                                                                      scaling_w), int((row['x2'])//scaling_h)]  # resizing as per the scaling factor
                # print(bbox)
                object_img = bbox_to_imroi(video_frame, bbox)
                # print('roi img shape', object_img.shape)
                try:
                    object_img = to_pil_image(object_img)
                    newsize = (224, 224)  # resizing the rois to the same size of the input image
                    object_img = object_img.resize(newsize)
                    object_feat = featureExt(object_img)  # 1x2048
                    # object_feat = object_feat.detach().numpy()
                    object_feat = object_feat.cpu().numpy() if object_feat.is_cuda else object_feat.detach().numpy()

                    feature[f_num, row['object_num']+1, :] = object_feat
                except Exception as e:
                    print('---error---')
                    log_information(vid_id, video_frame, row['track_id'], e)
                    continue

        save_file = feature_dir + vid_id+'.npz'
        np.savez_compressed(save_file, feature=feature,
                            detection=detections, vid_id=vid_id, toa=toa)


if __name__ == '__main__':
    main()
