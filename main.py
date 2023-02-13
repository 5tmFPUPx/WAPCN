# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.optim as optim
from DF_Model import DFNet
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from WAPCN import WAPCN
import math

# Assuming that the data preprocessing has been completed, 4 numpy arrays are obtained
# raw_tcp_traces
# raw_tcp_traces_labels
# raw_quic_traces
# raw_quic_traces_labels
    
class TCP_dataset(Dataset):
    def __init__(self, tcp_traces, tcp_traces_labels):
        self.tcp_traces = tcp_traces
        self.tcp_traces_labels = tcp_traces_labels
        self.tcp_protocol_labels = torch.from_numpy(np.zeros([len(tcp_traces_labels)], dtype = int)) # add protocol label. use 0 to denote TCP.
 
    def __len__(self):
        return len(self.tcp_traces)
 
    def __getitem__(self, idx):
        tcp_traces = self.tcp_traces[idx]
        tcp_traces_labels = self.tcp_traces_labels[idx]
        tcp_protocol_labels = self.tcp_protocol_labels[idx]
        return tcp_traces, tcp_traces_labels, tcp_protocol_labels

class QUIC_dataset(Dataset):
    def __init__(self, quic_traces, quic_traces_labels):
        self.quic_traces = quic_traces
        self.quic_traces_labels = quic_traces_labels
        self.quic_protocol_labels = torch.from_numpy(np.ones([len(quic_traces_labels)], dtype = int)) # add protocol label. use 1 to denote TCP.

    def __len__(self):
        return len(self.quic_traces)

    def __getitem__(self, idx):
        quic_traces = self.quic_traces[idx]
        quic_traces_labels = self.quic_traces_labels[idx]
        quic_protocol_labels = self.quic_protocol_labels[idx]
        return quic_traces, quic_traces_labels, quic_protocol_labels


if __name__ == '__main__':
    tcp_num_classes = 101
    quic_num_classes = 100
    model = WAPCN(quic_num_classes = quic_num_classes, tcp_num_classes=tcp_num_classes)
    
    lr=[0.0005, 0.001]
    #optimizer = torch.optim.SGD([{'params': model.feature_layers.parameters()},
    #        {'params': model.fc.parameters(), 'lr': lr[1]},
    #    ], lr=lr[0], momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adamax([{'params': model.feature_layers.parameters()},
            {'params': model.fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], weight_decay=0)
    nepoch = 100
    weight = 0.5

    tcp_traces_per_class = 50
    quic_traces_per_class = 5

    tcp_train_dataset = TCP_dataset(raw_tcp_traces, raw_tcp_traces_labels)
    quic_train_dataset = QUIC_dataset(raw_quic_traces, raw_quic_traces_labels)
    tcp_train_dataloader = DataLoader(tcp_train_dataset, batch_size=32)
    quic_train_dataloader = DataLoader(quic_train_dataset, batch_size=32) 

    criterion = nn.CrossEntropyLoss()
    len_dataloader = min(len(tcp_train_dataloader), len(quic_train_dataloader))

    for epoch in range(100):
        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (epoch - 1) / nepoch), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (epoch - 1) / nepoch), 0.75)
        model.train()
        iter_tcp_train = iter(tcp_train_dataloader)
        iter_quic_train = iter(quic_train_dataloader)
        num_iter_tcp_train = len(tcp_train_dataloader)
        for i in range(1, num_iter_tcp_train): 
            tcp_traces, tcp_traces_labels, tcp_protocol_labels = iter_tcp_train.next()
            quic_traces, quic_traces_labels, quic_protocol_labels  = iter_quic_train.next()
            if i % len(quic_train_dataloader) == 0:
                iter_quic_train = iter(quic_train_dataloader)

            tcp_traces = tcp_traces.type(torch.FloatTensor) # IF GPU, torch.cuda.FloatTensor
            tcp_traces_labels = tcp_traces_labels.type(torch.LongTensor)
            quic_traces = quic_traces.type(torch.FloatTensor) 
            quic_traces_labels = quic_traces_labels.type(torch.LongTensor)
            tcp_protocol_labels = tcp_protocol_labels.type(torch.LongTensor)
            quic_protocol_labels = quic_protocol_labels.type(torch.LongTensor)

            p = float(i + epoch * len_dataloader) / nepoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()
            label_source_pred, loss_lmmd, label_target_pred, source_domain_pred, target_domain_pred = model(tcp_traces, quic_traces, tcp_traces_labels, quic_traces_labels, alpha)
            loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), tcp_traces_labels)
            lambd = 2 / (1 + math.exp(-10 * (epoch) / nepoch)) - 1

            loss_quic = criterion(label_target_pred, quic_traces_labels)
            loss_source_domain = criterion(source_domain_pred, (torch.from_numpy(np.zeros([len(source_domain_pred)], dtype = int))).type(torch.LongTensor)) # tcp domain label 0
            loss_target_domain = criterion(target_domain_pred, (torch.from_numpy(np.ones([len(target_domain_pred)], dtype = int))).type(torch.LongTensor)) # quic domain label 1

            loss = 1 * loss_cls + 1 * loss_quic + weight * lambd * (loss_lmmd + loss_source_domain + loss_target_domain)

            loss.backward()
            optimizer.step()
        print('Epoch: ', epoch+1, 'Loss: ', loss.item())
        #torch.save(model.state_dict(), 'model.pkl')
