from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['ResNet50TP', 'ResNet50TA', 'ResNet50RNN']


def Inter_frame_feature_reorganization(x, t):
    vs = list()
    M1 = Variable(torch.eye(t, t)).to(x.device)

    for v in x:
        l2 = torch.norm(v, p=2, dim=1, keepdim=True)
        metric_matrix = v / l2

        metric_matrix2 = metric_matrix * metric_matrix
        spV = metric_matrix2.sum(1) / 2
        spV = spV.unsqueeze(0)
        spM = spV + spV.transpose(1, 0)

        mpM = torch.matmul(metric_matrix, metric_matrix.transpose(1, 0))

        dM = M1 + spM - mpM
        # dM = F.sigmoid(dM)
        dM = F.softmax(dM, dim=1)
        v = torch.matmul(dM, v)
        vs.append(v.unsqueeze(0))

    x = torch.cat(vs, dim=0)
    return x


class ResNet50TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        self.iffr = kwargs['iffr']
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)

        if self.iffr:
            x = Inter_frame_feature_reorganization(x, t)

        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        self.iffr = kwargs['iffr']
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)

        if self.iffr:
            att_x = Inter_frame_feature_reorganization(att_x, t)

        att_x = torch.sum(att_x,1)
        
        f = att_x.view(b,self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50RNN, self).__init__()
        self.loss = loss
        self.iffr = kwargs['iffr']
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        if self.iffr:
            x = Inter_frame_feature_reorganization(x, t)


        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

