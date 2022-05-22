import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#######################################################################################

# Multi Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask=None):
        super(MultiHeadAttention, self).__init__()

        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads
        inputs:
            q: the query tensor of shape [batch_size, set_len, d_model]
            k: the key tensor of shape [batch_size, set_len, d_model]
            v: the value tensor of shape [batch_size, set_len, d_model]
            padding_mask: the masking tensor of shape [batch_size, set_len, set_len],
            it is used to mask out the padding tokens in the input
        returns:
            a float tensor of shape [batch_size, set_len_q, d_model]
        notes:
            In all situation:
                dk = dv 
                set_len_k = set_len_v
        '''

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_q = d_model // n_heads

        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.linear_q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.linear_k = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.linear_v = nn.Linear(self.d_model, self.d_v * self.n_heads)
        self.linear_o = nn.Linear( self.d_v * self.n_heads, self.d_model)

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)
        q = self.linear_q(q) # [batch_size, set_len_q, d_k * n_heads]
        k = self.linear_k(k) # [batch_size, set_len_v, d_k * n_heads]
        v = self.linear_v(v) # [batch_size, set_len_v, d_v * n_heads]

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, set_len_q, d_k]
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, set_len_v, d_k]
        v = v.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # [batch_size, n_heads, set_len_v, d_v]

        energy = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.d_k)
        # k.transpose(2, 3) means transpose [batch_size, n_heads, set_len_v, d_k] to [batch_size, n_heads, d_k, se_len_v]
        # [batch_size, n_heads, set_len_q, set_len_v]

        if mask is not None:
            energy = energy.masked_fill(mask, float('-inf'))

        attention = F.softmax(energy, dim=-1)
        # [batch_size, n_heads, set_len_q, set_len_v]

        context = torch.matmul(attention, v)
        # [batch_size, n_heads, set_len_q, d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        # [batch_size, set_len_q, n_heads * d_v = d_model]
        
        output = self.linear_o(context)
        # [batch_size, set_len_q, d_model]
        return output

#######################################################################################

# Row-wise Feed Forward Network
class RFF(nn.Module):
    def __init__(self, d_model, forward_exapnsion):
        super(RFF, self).__init__()
        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            forward_exapnsion: the number of forward expansion

        inputs:
            x: the input tensor of shape [batch_size, set_len, d_model]
        returns:
            a float tensor of shape [batch_size, set_len, d_model]
        '''
        self.d_model = d_model
        self.linear_r = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*forward_exapnsion),
            nn.ReLU(),
            nn.Linear(self.d_model*forward_exapnsion, self.d_model)
        )
    def forward(self, x):
        return self.linear_r(x)
#######################################################################################

class MAB(nn.Module):
    def __init__(self, d_model, n_heads, forward_exapnsion):
        super(MAB, self).__init__()
        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads
        inputs:
            X: the query tensor of shape [batch_size, set_len_X, d_model]
            Y: the key tensor of shape [batch_size, set_len_Y, d_model]
        returns:
            a float tensor of shape [batch_size, set_len_X, d_model]
        '''

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.rFF = RFF(d_model, forward_exapnsion)

    def forward(self, X, Y):
        H = self.norm1(X + self.attention(X, Y, Y))
        _MAB = self.norm2(H + self.rFF(H))
        return _MAB

#######################################################################################

# Set Attention Block (SAB)
class SAB(nn.Module):
    def __init__(self, d_model, n_heads, forward_exapnsion):
        super (SAB, self).__init__()
        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads
        inputs:
            X: the Embedding tensor of shape [batch_size, set_len, d_model]
        returns:
            a float tensor of shape [batch_size, set_len, d_model]
        '''
        self.d_model = d_model
        self.n_heads = n_heads

        self.mab = MAB(self.d_model, self.n_heads, forward_exapnsion)

    def forward(self, X):

        batch_size = X.size(0)
        X = X.view(batch_size, -1, self.d_model)
        # [batch_size, set_len, d_model]
        X = self.mab(X, X)
        # [batch_size, set_len, d_model]

        return X

#######################################################################################

# Induced Set-Attention Block
class ISAB(nn.Module):
    def __init__(self, d_model, n_heads, induced_dim, forward_exapnsion):
        super(ISAB, self).__init__()
        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads
            induced_dim: the dimension of induced vector
        inputs:
            X: the Embedding tensor of shape [batch_size, set_len_X, d_model]
        
        returns:
            a float tensor of shape [batch_size, set_len_X, d_model]
        notes:
            induce_dim = id
        '''
        self.d_model = d_model

        self.n_heads = n_heads

        self.induced_dim = induced_dim

        self.I = nn.Parameter(torch.Tensor(1, self.induced_dim, d_model))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(self.d_model, self.n_heads, forward_exapnsion)
        # [batch, set_len_id, d_model]
        self.mab1 = MAB(self.d_model, self.n_heads, forward_exapnsion)
        # [batch, set_len_X, d_model]

    def forward(self, X):

        H = self.mab0(self.I.repeat(X.size(0),1,1) , X)
        _ISAB = self.mab1(X,H)

        return _ISAB


#######################################################################################

# Pooling by Multihead Attention
class PMA(nn.Module):
    def __init__(self, d_model, n_heads, n_seeds, forward_exapnsion):
        super(PMA, self).__init__()
        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads
            n_seeds: the number of seed vectors
        inputs:
            X: the Embedding tensor of shape [batch_size, set_len_X, d_model]
        returns:
            a float tensor of shape [batch_size, n_seeds, d_model]
        '''

        self.S = nn.Parameter(torch.Tensor(1, n_seeds, d_model))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(d_model, n_heads, forward_exapnsion)
        self.rFF = RFF(d_model, forward_exapnsion)
    
    def forward(self, X):
        X = self.rFF(X)
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    
#######################################################################################
