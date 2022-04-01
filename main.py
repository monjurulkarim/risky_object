from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from models.model import RiskyObject
from models.evaluation import evaluation, plot_curve
from dataloader import MyDataset
import argparse
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np


def write_scalars(logger, epoch, losses, lr):
    # fetch results
    cross_entropy = losses['cross_entropy'].mean()
    # write to tensorboardX
    logger.add_scalars('train/loss', {'Loss': cross_entropy}, epoch)
    logger.add_scalars("train/lr", {'lr': lr}, epoch)


def write_test_scalars(logger, epoch, losses, roc_auc):
    cross_entropy = losses.mean()
    logger.add_scalars('test/loss', {'Loss': cross_entropy}, epoch)
    logger.add_scalars('test/roc_auc', {'roc_auc': roc_auc}, epoch)
    # logger.add_scalars('test/fpr', {'fpr': fpr}, epoch)
    # logger.add_scalars('test/fpr', {'tpr': tpr}, epoch)


def write_pr_curve_tensorboard(logger, test_probs, test_label):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    # tensorboard_truth = test_label == class_index
    tensorboard_truth = np.array(test_label)
    tensorboard_probs = np.array([test_probs[i][0] for i in range(len(test_probs))])

    # tensorboard_probs = test_probs[:, 1]
    # print(test_probs)
    # print(test_label[2])
    classes = ['no_risk', 'risk']

    logger.add_pr_curve(classes[1],
                        tensorboard_truth,
                        tensorboard_probs)
                        # global_step=global_step)



def write_weight_histograms(logger, model, epoch):
    logger.add_histogram('histogram/gru.weight_ih_l0', model.gru_net.gru.weight_ih_l0, epoch)
    logger.add_histogram('histogram/gru.weight_hh_l0', model.gru_net.gru.weight_hh_l0, epoch)
    logger.add_histogram('histogram/gru.bias_ih_l0', model.gru_net.gru.bias_ih_l0, epoch)
    logger.add_histogram('histogram/gru.bias_hh_l0', model.gru_net.gru.bias_hh_l0, epoch)
    # logger.add_histogram('histogram/gru.weight_ih_l1', model.gru_net.gru.weight_ih_l1, epoch)
    # logger.add_histogram('histogram/gru.weight_hh_l1', model.gru_net.gru.weight_hh_l1, epoch)
    # logger.add_histogram('histogram/gru.bias_ih_l1', model.gru_net.gru.bias_ih_l1, epoch)
    # logger.add_histogram('histogram/gru.bias_hh_l1', model.gru_net.gru.bias_hh_l1, epoch)
    #fc_layers
    logger.add_histogram('histogram/gru.dense1.weight', model.gru_net.dense1.weight, epoch)
    logger.add_histogram('histogram/gru.dense1.bias', model.gru_net.dense1.bias, epoch)
    logger.add_histogram('histogram/gru.dense2.weight', model.gru_net.dense2.weight, epoch)
    logger.add_histogram('histogram/gru.dense2.bias', model.gru_net.dense2.bias, epoch)




def test_all(testdata_loader, model):
    
    all_pred = []
    all_labels = []
    losses_all = []
    with torch.no_grad():
        for i, (batch_xs, batch_det, batch_toas) in enumerate(testdata_loader):
            losses, all_outputs, labels = model(batch_xs, batch_det, batch_toas)

            losses_all.append(losses)
            for t in range(100):
                frame = all_outputs[t]
                if len(frame)== 0:
                    continue
                else:
                    for j in range(len(frame)):
                        score = np.exp(frame[j][:,1])/np.sum(np.exp(frame[j]),axis=1)
                        all_pred.append(score)
                        all_labels.append(labels[t][j]+0) #added zero to convert array to scalar

    # all_pred = np.array([all_pred[i][0] for i in range(len(all_pred))])

    return losses_all, all_pred, all_labels


def average_losses(losses_all):
    cross_entropy = 0
    for losses in losses_all:
        cross_entropy += losses['cross_entropy']
    losses_mean = cross_entropy/len(losses_all)
    return losses_mean


def sanity_check():
    # data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    data_path = p.data_path
    model_dir = os.path.join(p.output_dir, 'snapshot')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logs_dir = os.path.join(p.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # logger = SummaryWriter(logs_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = MyDataset(data_path, 'train', toTensor=True, device=device)
    test_data = MyDataset(data_path, 'val', toTensor=True, device=device)

    traindata_loader = DataLoader(
        dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size,
                                 shuffle=False, drop_last=True)
    batch_xs, batch_det, batch_toas = next(iter(traindata_loader))
    n_frames = 100
    fps = 20

    model = RiskyObject(p.x_dim, p.h_dim, n_frames, fps)
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('pytorch_total_params : ', pytorch_total_params)
    # sys.exit()

    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    model = model.to(device=device)
    # for name, param in model.named_parameters():
    #     print(name)

    for epoch in range(p.epoch):
        print(f"Epoch  [{epoch}/{p.epoch}]")

        losses, all_outputs, all_labels = model(batch_xs, batch_det, batch_toas)
        # p,r = evaluation(all_outputs, all_labels)
        print('outputs',(all_outputs))
        print('all_labels',(all_labels))
        np.savez('eval.npz', outputs= all_outputs, labels=all_labels)
        sys.exit()
        # backward

        # clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.zero_grad()
        losses['cross_entropy'].mean().backward()
        optimizer.step()
        # print(losses['cross_entropy'].item())
        # loop.set_description(f"Epoch  [{k}/{p.epoch}]")
        # loop.set_postfix(loss= losses['cross_entropy'].item() )




def train_eval():

    # data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    data_path = p.data_path
    model_dir = os.path.join(p.output_dir, 'snapshot')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logs_dir = os.path.join(p.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logs_dir = os.path.join(p.output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)

    # logger = SummaryWriter(logs_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = MyDataset(data_path, 'train', toTensor=True, device=device)
    test_data = MyDataset(data_path, 'val', toTensor=True, device=device)

    traindata_loader = DataLoader(
        dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size,
                                 shuffle=False, drop_last=True)

    n_frames = 100
    fps = 20

    model = RiskyObject(p.x_dim, p.h_dim, n_frames, fps)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    model = model.to(device=device)

    # for param in model.parameters():
    #     print(param)

    model.train()
    start_epoch = -1
    # resume training
    # -----------------
    # TO-DO:
    # -----------------
    # write histograms
    write_weight_histograms(logger, model, 0)

    iter_cur = 0

    for k in range(p.epoch):
        loop = tqdm(enumerate(traindata_loader), total=len(traindata_loader))
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        for i, (batch_xs, batch_det, batch_toas) in loop:
            optimizer.zero_grad()
            losses, all_outputs, all_labels = model(batch_xs, batch_det, batch_toas)

            # backward
            losses['cross_entropy'].mean().backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            loop.set_description(f"Epoch  [{k}/{p.epoch}]")
            loop.set_postfix(loss=losses['cross_entropy'].item())

            # -----------------
            # write scalars
            # To-DO:
            lr = optimizer.param_groups[0]['lr']
            write_scalars(logger, k, losses, lr)
            # ---------------
            iter_cur = 0


        if k % p.test_iter == 0 and k!=0:

            model.eval()
            losses_all,all_pred, all_labels = test_all(testdata_loader, model)

            loss_val = average_losses(losses_all)
            fpr, tpr, roc_auc= evaluation(all_pred,all_labels,k)
            plot_curve(fpr,tpr,roc_auc,k)
            print('----------------------------------')
            print("Starting evaluation...")
            print('AUC : ', roc_auc)
            # print('testing loss :', loss_val)
            # keep track of validation losses
            write_test_scalars(logger, k, loss_val, roc_auc)
            model.train()

            write_pr_curve_tensorboard(logger, all_pred, all_labels)


            # -----------------
            # test and evaluate the model
            # To-DO:
            # -----------------

        # save model
        # model_file = os.path.join(model_dir, 'risky_%02d.pth' % (k))
        # torch.save({'epoch': k,
        #             'model': model.state_dict(),
        #             'optimizer': optimizer.state_dict()}, model_file)
        # print('Model has been saved as: %s' % (model_file))
        scheduler.step(losses['cross_entropy'])
        # write histograms
        write_weight_histograms(logger, model, k+1)
    logger.close()

    # print('===Checking the model construction====')
    # for e in range(2):
    #     print('Epoch: %d' % (e))
    #     for i, (batch_xs, batch_det, batch_toas) in tqdm(enumerate(traindata_loader), total=len(traindata_loader)):
    #         if i == 0:
    #             losses, all_outputs, all_labels = model(batch_xs, batch_det, batch_toas)
    #
    #             # print('feature dim:', x.size())
    # print('detection dim:', y.size())
    # print('toas dim:', toa.size())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./feat_extract/feature',
                        help='The relative path of dataset.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The batch size in training process. Default: 1')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--epoch', type=int, default=24,
                        help='The number of training epoches. Default: 30')
    parser.add_argument('--h_dim', type=int, default=256,
                        help='hidden dimension of the gru. Default: 256')
    parser.add_argument('--x_dim', type=int, default=2048,
                        help='dimension of the resnet output. Default: 2048')
    parser.add_argument('--phase', type=str, default='train', choices=['check', 'train', 'test'],
                        help='dimension of the resnet output. Default: 2048')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='The relative path of dataset.')
    parser.add_argument('--test_iter', type=int, default=4,
                        help='The number of epochs to perform a evaluation process. Default: 64')
    p = parser.parse_args()
    if p.phase == 'test':
        test_eval()
    elif p.phase == 'check':
        sanity_check()
    else:
        train_eval()
