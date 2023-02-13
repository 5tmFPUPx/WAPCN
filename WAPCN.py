# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from DF_Model import DFNet
import lmmd
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class WAPCN(nn.Module):

    def __init__(self, quic_num_classes, tcp_num_classes=101):
        super(WAPCN, self).__init__()
        self.model = DFNet(tcp_num_classes)
        self.model.load_state_dict(torch.load('DF_v3_tcp_trained_pytorch.pkl'))
        self.feature_layers = self.model.feature_layers
        self.fc = self.model.fc
        self.lmmd_loss = lmmd.LMMD_loss(class_num=tcp_num_classes)

        self.domain_classifier = nn.Sequential() # protocol discriminator
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 128))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(128))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(128, 2))
        #self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, source, target, s_label, t_label, alpha):
        #s_pred = self.model(source)
        source_feature = self.feature_layers(source) # representation
        s_pred = self.fc(source_feature)
        reverse_source_feature = ReverseLayerF.apply(source_feature, alpha)
        source_domain_pred = self.domain_classifier(reverse_source_feature)

        #t_pred = self.model(target)
        target_feature = self.feature_layers(target) 
        t_pred = self.fc(target_feature)
        reverse_target_feature = ReverseLayerF.apply(target_feature, alpha)
        target_domain_pred = self.domain_classifier(reverse_target_feature)

        loss_lmmd = self.lmmd_loss.get_loss(source_feature, target_feature, s_label, t_label) # website-aware adaptor
        return s_pred, loss_lmmd, t_pred, source_domain_pred, target_domain_pred

    def predict(self, x):
        x = self.model(x)
        return x