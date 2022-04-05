from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import inspect
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = [0, 0]
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        self.dense1 = torch.nn.Linear(hidden_dim, 64)
        # self.dense2 = torch.nn.Linear(128, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU()
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.softmax()

    def forward(self, x, h):

        out, h = self.gru(x, h)
        # print('output shape', out[:, -1].shape)
        # print('===========')
        out = F.dropout(out[:, -1], self.dropout[0])  # optional
        # out = self.leaky_relu(self.dense1(out))
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        # out = self.logsoftmax(out)
        # out = F.dropout(out, self.dropout[1])
        # out = self.dense3(out)
        # # out = self.relu(out)
        # out = self.sigmoid(out)  # TO-Do: Check without using it
        return out, h


class RiskyObject(nn.Module):
    def __init__(self, x_dim, h_dim, n_frames=100, fps=20.0):
        super(RiskyObject, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.fps = fps
        self.n_frames = n_frames
        self.n_layers = 2
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2, self.n_layers)
        self.weight = torch.Tensor([0.25, 1]).cuda()
        # self.bce_loss = torch.nn.BCELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss(weight = self.weight,reduction='mean')

    def forward(self, x, y, toa, hidden_in=None, testing=False):
        """
        :param x (batchsize, nFrames, 1+maxBox, Xdim)
        :param y (batchsize, nFrames, maxBox, 6)
        :toa (batchsize, 1)
        :batchsize = 1, currently we support batchsize=1
        """
        losses = {'cross_entropy': 0}
        h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim)
                     )  # TO-DO: hidden_in like dsta
        h = h.to(x.device)
        h_all_in = {}
        h_all_out = {}
        all_outputs = []
        all_labels = []
        for t in range(x.size(1)):
            # projecting to a lower dimensional space
            # 2048 --> 256
            x_val = self.phi_x(x[:, t])  # 1 x 31 x 256
            img_embed = x_val[:, 0, :].unsqueeze(1)  # 1 x 1 x 256
            img_embed = img_embed.repeat(1, 30, 1)  # 1 x 30 x 256
            obj_embed = x_val[:, 1:, :]   # 1 x 30 x 256 # TO-DO: DSA --> try spatial attention

            # If don't want to project to a lower dimensional space
            # need to unblock the following code
            # ----------------------------------------
            # img_embed = x[:, :, 0, :].unsqueeze(2)
            # img_embed = img_embed.repeat(1, 1, 30, 1)
            # obj_embed = x[:, :, 1:, :]  # TO-DO: DSA --> try spatial attention
            # -----------------------------------------
            x_t = torch.cat([obj_embed, img_embed], dim=-1)  # 1 x 30 x 512

            h_all_out = {}
            frame_outputs = []
            frame_labels = []
            # frame_loss = []
            for bbox in range(30):
                if y[0][t][bbox][0] == 0:
                    continue
                else:
                    track_id = str(y[0][t][bbox][0].cpu().detach().numpy())
                    if track_id in h_all_in:
                        h_in = h_all_in[track_id]  # 1x1x256
                        # x_obj = x_t[0][t][bbox]  # 4096 # x_t[batch][frame][bbox]
                        x_obj = x_t[0][bbox]  # 4096 # x_t[batch][frame][bbox]
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512
                        output, h_out = self.gru_net(x_obj, h_in)  # 1x1x256
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        # frame_loss.append(loss)
                        # print('tracked output: ', output)
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  # storing in a dictionary
                    else:  # If object was not found in the previous frame
                        h_in = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim)
                                        )  # TO-DO: hidden_in like dsta
                        h_in = h_in.to(x.device)
                        # x_obj = x_t[0][t][bbox]  # 4096
                        x_obj = x_t[0][bbox]  # 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512
                        output, h_out = self.gru_net(x_obj, h_in)  # 1x1x256
                        # print('output : ', output)

                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))
                        # target = target.squeeze()
                        # print('target : ', target)
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        # frame_loss.append(loss)
                        frame_outputs.append(output.detach().cpu().numpy())
                        # print('labels: ', (y[0][t][bbox][5].detach().cpu().numpy()))
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  # storing in a dictionary
            # print('=================')ss
            # print('frame  :', t)
            # # print('all labels: ', frame_labels)
            # # print('--')
            # # print('all outputs: ', frame_outputs)
            # print('--')
            # print('loss : ', losses['cross_entropy'])
            # if len(frame_loss) == 0:
            #     frame_loss = torch.zeros(1).to(x.device)
            #     print('frame loss: ', frame_loss)
            # else:
            #     l = sum(frame_loss)
            #     print('frame loss: ', l/len(frame_loss))
            # loss = self.bce_loss(frame_outputs[:], frame_labels[:])

            # print('=================')
            all_outputs.append(frame_outputs)
            all_labels.append(frame_labels)
            h_all_in = {}
            h_all_in = h_all_out.copy()
            # if t == 99:
            #     print('break')
            #     sys.exit(0)

        return losses, all_outputs, all_labels
