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
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, output_cor_dim):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = [0, 0]
        self.output_cor_dim = output_cor_dim
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        self.dense1 = torch.nn.Linear(hidden_dim+output_cor_dim, 256)
        self.dense2 = torch.nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h, output_cor):

        out, h = self.gru(x, h)
        out = torch.cat([out, output_cor], dim=-1)
        out = F.dropout(out[:, -1], self.dropout[0])  # optional
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)

        return out, h


class CorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(CorGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class flow_GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(flow_GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class SpatialAttention(nn.Module):
    """
    Applied soft attention on the hidden representation of all the objects in a frame.
    """

    def __init__(self, h_dim):
        super(SpatialAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_dim, 1))  # hidden representation dimension is 256
        self.softmax = nn.Softmax(dim=1)
        import math
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, h_all_in):
        """
        :h_all_in - dictionary containing object tracking id and hidden representation of size 2 x 1 x 256 each
        :output - dictionary with the same shape of h_all_in
        """
        k = []
        v = []
        for key in h_all_in:
            v.append(h_all_in[key])
            k.append(key)

        if len(v) != 0:
            h_in = torch.cat([element for element in v], dim=1)
            m = torch.tanh(h_in)
            alpha = torch.softmax(torch.matmul(m, self.weight), 1)
            roh = torch.mul(h_in, alpha)
            list_roh = []
            for i in range(roh.size(1)):
                list_roh.append(roh[:, i, :].unsqueeze(1).contiguous())

            h_all_in = {}
            for ke, value in zip(k, list_roh):
                h_all_in[ke] = value

        return h_all_in


class RiskyObject(nn.Module):
    def __init__(self, x_dim, h_dim, n_frames=100, fps=20.0):
        super(RiskyObject, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.fps = fps
        self.n_frames = n_frames
        self.n_layers = 2
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())  # rgb

        # for secondary GRU
        self.n_layers_cor = 1
        self.h_dim_cor = 32
        self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2, self.n_layers, self.h_dim_cor)
        self.weight = torch.Tensor([0.25, 1]).cuda()  # TO-DO: find the correct weight

        # input dim 4
        self.gru_net_cor = CorGRU(4, self.h_dim_cor, self.n_layers_cor)
        self.soft_attention = SpatialAttention(h_dim)
        self.soft_attention_cor = SpatialAttention(self.h_dim_cor)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='mean')

    def forward(self, x, y, toa, flow, hidden_in=None, testing=False):
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

        # hidden representation for secondary gru
        h_all_in_cor = {}
        h_all_out_cor = {}
        # h_all_in_flow = {}
        # h_all_out_flow = {}

        all_outputs = []
        all_labels = []

        for t in range(x.size(1)):
            # projecting to a lower dimensional space
            inp = flow[:, t]  # 1 x31 x2048

            # Flow----------------
            x_val = self.phi_x(inp)  # 1 x 31 x 256  #rgb_d
            img_embed = x_val[:, 0, :].unsqueeze(1)  # 1 x 1 x 256
            img_embed = img_embed.repeat(1, 30, 1)  # 1 x 30 x 256
            obj_embed = x_val[:, 1:, :]   # 1 x 30 x 256
            x_t = torch.cat([obj_embed, img_embed], dim=-1)  # 1 x 30 x 512

            h_all_out = {}
            h_all_out_cor = {}
            h_all_out_flow = {}
            frame_outputs = []
            frame_labels = []
            for bbox in range(30):
                if y[0][t][bbox][0] == 0:  # ignore if there is no bounding box
                    continue
                else:
                    track_id = str(y[0][t][bbox][0].cpu().detach().numpy())
                    if track_id in h_all_in:

                        # secondary GRU-----------------------------------
                        # decoding the coordinate with a secondary GRU model
                        unnormalized_cor = y[0][t][bbox]  # unnormalized coordinate (1080,720)scale
                        # print(d[1]/1080)
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3] /
                                                 1080, unnormalized_cor[4]/720])  # normalized coordinate

                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)

                        # hidden representation for coordinate gru
                        h_in_cor = h_all_in_cor[track_id]
                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)

                        h_all_out_cor[track_id] = h_out_cor

                        # base GRU---------------------------------------
                        h_in = h_all_in[track_id]  # 1x1x256

                        x_obj = x_t[0][bbox]  # 4096 # x_t[batch][frame][bbox]
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512

                        output, h_out = self.gru_net(
                            x_obj, h_in, output_cor)  # 1x1x256
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))

                        # compute error per object
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  # storing in a dictionary

                    else:  # If object was not found in the previous frame

                        # secondary GRU --------------------------------------
                        unnormalized_cor = y[0][t][bbox]  # unnormalized coordinate (1080,720)scale
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3] /
                                                 1080, unnormalized_cor[4]/720])  # normalized coordinate

                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)

                        # hidden representation for coordinate gru
                        h_in_cor = Variable(torch.zeros(
                            self.n_layers_cor, x.size(0),  self.h_dim_cor))

                        h_in_cor = h_in_cor.to(x.device)

                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)
                        # Base GRU------------------------------------------
                        h_in = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim)
                                        )  # TO-DO: hidden_in like dsta
                        h_in = h_in.to(x.device)
                        x_obj = x_t[0][bbox]  # 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512

                        output, h_out = self.gru_net(
                            x_obj, h_in, output_cor)  # 1x1x256
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  # storing in a dictionary
                        h_all_out_cor[track_id] = h_out_cor

            all_outputs.append(frame_outputs)
            all_labels.append(frame_labels)
            h_all_in = {}
            h_all_in = h_all_out.copy()

            h_all_in = self.soft_attention(h_all_in)

            h_all_in_cor = {}
            h_all_in_cor = h_all_out_cor.copy()
            h_all_in_cor = self.soft_attention_cor(h_all_in_cor)

        return losses, all_outputs, all_labels
