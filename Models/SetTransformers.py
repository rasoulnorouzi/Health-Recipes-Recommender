import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

    '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads

        inputs:
            q: the query tensor of shape [batch_size, seq_len, d_model]
            k: the key tensor of shape [batch_size, seq_len, d_model]
            v: the value tensor of shape [batch_size, seq_len, d_model]
            padding_mask: the masking tensor of shape [batch_size, seq_len, seq_len],
            it is used to mask out the padding tokens in the input

        returns:
            a float tensor of shape [batch_size, seq_len_q, d_model]

        notes:
            In all situation:
                dk = dv 
                seq_k = seq_v
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
        q = self.linear_q(q) # [batch_size, seq_len_q, d_k * n_heads]
        k = self.linear_k(k) # [batch_size, seq_len_v, d_k * n_heads]
        v = self.linear_v(v) # [batch_size, seq_len_v, d_v * n_heads]

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, seq_len_q, d_k]
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, seq_len_v, d_k]
        v = v.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # [batch_size, n_heads, seq_len_v, d_v]

        energy = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.d_k)
        # k.transpose(2, 3) means transpose [batch_size, n_heads, seq_len_v, d_k] to [batch_size, n_heads, d_k, seq_len_v]
        # [batch_size, n_heads, seq_len_q, seq_len_v]

        if mask is not None:
            energy = energy.masked_fill(mask, float('-inf'))

        attention = F.softmax(energy, dim=-1)
        # [batch_size, n_heads, seq_len_q, seq_len_v]

        context = torch.matmul(attention, v)
        # [batch_size, n_heads, seq_len_q, d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        # [batch_size, seq_len_q, n_heads * d_v = d_model]
        
        output = self.linear_o(context)
        # [batch_size, seq_len_q, d_model]
        return output


class MAB(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MAB, self).__init__()
        '''
        Arguments:
            d_model: the dimension of Embedding Layer
            n_heads: the number of heads
        inputs:
            X: the query tensor of shape [batch_size, seq_len_X, d_model]
            Y: the key tensor of shape [batch_size, seq_len_Y, d_model]
        returns:
            a float tensor of shape [batch_size, seq_len_X, d_model]
        '''

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.rFF = nn.Linear(d_model, d_model)

    def forward(self, X, Y):
        H = self.norm1(X + self.attention(X, Y, Y))
        _MAB = self.norm2(H + self.rFF(H))
        return _MAB