'''
This calculates the results on the validation dataset of DoTA.
For calculating results we need the following:
    - Generate outputs/results in the server and save the *.npz file (to generate results probably use demo.py)
            (previous dota results are in "archive/results")
    - put the results in the directory 'results'
    - run the code to get frame level auc
'''

import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score
import glob


# Directory of the npz results on DoTA dataset
np_dir = '../checkpoints/output/dota_output_without_global'

np_files = glob.glob(os.path.join(np_dir, '*.npz'))
all_pred = []
all_labels = []
new_labels = []
for file in np_files:
    np_file = np.load(file, allow_pickle=True)
    output = np_file['output']
    labels = np_file['label']
    for t in range(len(output)):
        frame = output[t]
        frame_score = []
        frame_label = []
        if len(frame) == 0:
            continue
        else:
            for j in range(len(frame)):
                score = np.exp(frame[j][:, 1])/np.sum(np.exp(frame[j]), axis=1)
                frame_score.append(score)
                frame_label.append(labels[t][j]+0)
            all_pred.append(max(frame_score))
            all_labels.append(sum(frame_label))


for i in all_labels:
    if i > 0.0:
        new_labels.append(1.0)
    else:
        new_labels.append(0.0)

fpr, tpr, thresholds = metrics.roc_curve(np.array(new_labels), np.array(all_pred), pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)
