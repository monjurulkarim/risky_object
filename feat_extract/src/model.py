# import torch
import torch.nn as nn
import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")


# Feature extract
# class FeatureExtractor(nn.Module):
#     def __init__(self): # Let's say output dim = 256+256
#         super(FeatureExtractor,self).__init__()
#         self.resnet = models.resnet50(pretrained=True) #for transfer learning
#         self.resnet.fc = nn.Sequential(
#                            nn.Linear(2048, out_features= 2048))
#         # self.feat_extractor = extractor
#         # if self.feat_extractor =='resnet50':
#         #     self.resnet = models.resnet50(pretrained=True) #for transfer learning
#         #     self.resnet.fc = nn.Sequential(
#         #                    nn.Linear(2048, out_features= 2048))
#         # elif self.feat_extractor == 'vgg16':
#         #     self.resnet = models.vgg16(pretrained=True) #for transfer learning
#         #     self.resnet.classifier[6] = nn.Linear(in_features=4096,out_features=512)
#         # else:
#         #     raise NotImplementedError
#
#     def forward(self,x):
#         x = self.resnet(x)
#         return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.dim_feat = 2048

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output
