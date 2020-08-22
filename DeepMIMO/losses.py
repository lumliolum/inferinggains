import torch
import torch.nn as nn
import numpy as np

class MSELoss(nn.MSELoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y):
        return super(MSELoss, self).forward(y_pred, y)


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, y_pred, y):
        return torch.mean(torch.norm(y-y_pred, 2, dim=1)/torch.norm(y, 2, dim=1))


class CrossEntropyLoss(nn.Module):
    def __init__(self, method, eps=1e-2):
        super(CrossEntropyLoss, self).__init__()
        self.method = method
        self.eps = eps

    def forward(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, self.eps, 1-self.eps)

        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, 1)
        if len(y_pred.shape) == 1:
            y_pred = torch.unsqueeze(y_pred, 1)

        if self.method=='one_hot':
            loss = -torch.sum(y_true*torch.log(y_pred), 1)
        elif self.method=='bin_seq':
            loss = -torch.sum(y_true*torch.log(y_pred) + (1-y_true)*torch.log(1-y_pred), 1)
        loss = torch.sum(loss)/y_true.shape[0]
        return loss
