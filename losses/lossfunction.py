import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import lambertw
from scipy.stats import kendalltau


class Criterion(nn.Module):
    def __init__(self, n_classes, **kwargs):
        super(Criterion, self).__init__()

        self.nClasses = n_classes
        self.criterion = nn.CrossEntropyLoss()

        print('Initialised losses function orgin Criterion')

    def forward(self, x, labels, out_dict=None, epoch=None, flag='train', **kwargs):

        logits = x
        loss = self.criterion(logits, labels)

        if flag == 'train':
            self.train()
            return loss, None
        else:
            self.eval()
            return loss, None


