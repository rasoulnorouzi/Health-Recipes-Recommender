import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TransformerBlocks import *


class SetTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_seeds, induced_dim, d_output, mask=None):
        super(SetTransformer, self).__init__()
        
        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads
            n_seeds: the number of seed vectors
            induced_dim: the dimension of induced vector
            d_output: the dimension of output
        inputs:
            X: the Embedding tensor of shape [batch_size, seq_len_X, d_model]
        returns:
            a float tensor of shape [batch_size, n_seeds, d_output]

        '''

        self.enc = nn.Sequential(
            ISAB(d_model, n_heads, induced_dim),
            ISAB(d_model, n_heads, induced_dim),
        )
        self.dec = nn.Sequential(
            PMA(d_model, n_heads, n_seeds),
            SAB(d_model, n_heads),
            SAB(d_model, n_heads),
            nn.Linear(d_model, d_output)
            )
    def forward(self, X):
        X = self.enc(X)
        X = self.dec(X)
        return X


