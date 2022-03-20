'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.model import FeatureExtractor
from PIL import Image

device = ("cuda" if torch.cuda.is_available() else "cpu")
feat_extractor = 'resnet50'
h_dim = 256
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5 ,0.5)),
        ]
    )
image_path = 'new_file.jpg'

image = Image.open(image_path)
image = transform(image)
image = torch.unsqueeze(image,0)

print('=======================')
print(image.shape)
print('=======================')

feat_extract = FeatureExtractor(device, h_dim+h_dim, feat_extractor)
#
#
features = feat_extract(image)

print('features shape : ', features)
'''
import pandas as pd
import numpy as np


anno_file = 'm1.csv' # annotation file that contains the video id and the risky objects traking number

vid_ids = []
# scaling_w = 4.82 # 1080/224 Unnecessary at this stage
# scaling_h = 3.21 #720/224
with open(anno_file, 'r') as f:
    for line in f.readlines():
        a = line.split(",")
        print(a)
        vid_id = a[0]
        risk_object = a[2:-1]
        last_risk = a[-1].split('\n')[0]
        risk_object.append(last_risk)
        print(risk_object)
        risk_objects = [int(num) for num in risk_object]
        toa = str(a[1])
        vid_ids.append(vid_id)
        vid_text ='track_text/'+ vid_id+'.txt' #video tracking information stored in this text file
        dummy_list = []
        with open(vid_text, 'r') as vid_text_file:
            for line in vid_text_file.readlines():
                a = line.split(" ")
                b = a[:6]
                c = [int(val) for val in b]
                c[4] = c[2]+ c[4]
                c[5] = c[3]+ c[5]
                c[0]= c[0]-1 # frame number starts at 0
                dummy_list.append(c)
            vid_text_file.close()
        df = pd.DataFrame(np.array(dummy_list), columns=['frame','track_id','y1','x1','y2','x2'])
        unique, counts = np.unique(df['frame'].values, return_counts=True)
        df['object_num'] = None
        for i in range(len(unique)):
            df.loc[df['frame'] == unique[i],'object_num'] =np.array(range(counts[i]))
        df['label']= 0
        for i in range(len(risk_objects)):
            df.loc[df['track_id'] == risk_objects[i],'label']= int(1)
        df.to_csv('csv_files/'+vid_id+'-'+toa+ '.csv', index=False) # a csv file will be stored corresponding to each video_id



# features : 100 x 35 x 512
# detections : 100 x 34 x 6 :
        #detections: track_id x bbox  x labels
            # track_id : 1
            #bbox: 4 coordinates ((normalized to 224,224 dimension) (upper_left, lower right) (w->,h))
            #labels 1 (choice 0 : negative or 1: positive)
# vid : 1 (string)
