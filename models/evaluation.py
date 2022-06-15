# import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
import os
import numpy as np


def evaluation(all_pred, all_labels, epoch):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(all_labels), np.array(all_pred), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_auc_curve(fpr, tpr, roc_auc, epoch):
    curve_dir = 'charts/auc/'
    if not os.path.exists(curve_dir):
        os.makedirs(curve_dir)
    auc_curve_file = os.path.join(curve_dir, 'auc_%02d.jpg' % (epoch))

    plt.title(f'Receiver Operating Characteristic at epoch: {epoch}')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(auc_curve_file)
    plt.close()


def plot_pr_curve(all_labels, all_pred, epoch):
    pr_dir = 'charts/pr/'
    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)
    pr_curve_file = os.path.join(pr_dir, 'pr_%02d.jpg' % (epoch))
    precision, recall, thresholds = precision_recall_curve(np.array(all_labels), np.array(all_pred))
    ap = average_precision_score(np.array(all_labels), np.array(all_pred))

    plt.title(f'Precision-Recall Curve at epoch: {epoch}')
    plt.plot(recall, precision, 'b', label='AP = %0.2f' % ap)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(pr_curve_file)
    plt.close()
    return ap


def frame_auc(output, labels):
    # print(output)
    output = np.array(output)
    labels = np.array(labels)
    # print(output)
    all_pred = []
    all_labels = []

    for t in range(len(output)):
        frame = output[t]
        frame_score = []
        frame_label = []
        print(frame)

        if len(frame) == 0:
            continue
        else:
            for j in range(len(frame)):
                score = np.exp(frame[j][:, 1])/np.sum(np.exp(frame[j]), axis=1)
                frame_score.append(score)
                frame_label.append(labels[t][j]+0)
            all_pred.append(max(frame_score))
            all_labels.append(sum(frame_label))

    new_labels = []
    for i in all_labels:
        if i > 0.0:
            new_labels.append(1.0)
        else:
            new_labels.append(0.0)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(new_labels), np.array(all_pred), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc


# def evaluation(all_pred,all_labels):
#     # cm = confusion_matrix(all_labels, all_pred, labels=['no-risk','risk'])
#     TPs = 0
#     TNs = 0
#     FPs = 0
#     FNs = 0
#     for pred,gt in zip(all_pred,all_labels):
#         if gt ==0:
#             if pred ==0:
#                 TNs+=1
#             elif pred==1:
#                 FPs+=1
#         elif gt ==1:
#             if pred==0:
#                 FNs+=1
#             elif pred ==1:
#                 TPs+=1
#
#     cm = ([TNs,FPs],[FNs,TPs])
#     if TPs == 0:
#         recall =0
#         precision = 0
#         accuracy = (TPs+TNs)/(TPs+TNs+FPs+FNs)
#     else:
#         recall = TPs/(TPs+FNs)
#         precision = TPs/(TPs+FPs)
#         accuracy = (TPs+TNs)/(TPs+TNs+FPs+FNs)
#     return cm, precision, recall, accuracy
