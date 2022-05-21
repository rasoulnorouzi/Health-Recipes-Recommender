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
            a float tensor of shape [batch_size, seq_len, d_model]
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
        q = self.linear_q(q) # [batch_size, seq_len, d_k * n_heads]
        k = self.linear_k(k) # [batch_size, seq_len, d_k * n_heads]
        v = self.linear_v(v) # [batch_size, seq_len, d_v * n_heads]

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, seq_len, d_k]
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, seq_len, d_k]
        v = v.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # [batch_size, n_heads, seq_len, d_v]

        energy = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.d_k)
        # [batch_size, n_heads, seq_len, seq_len]

        if mask is not None:
            energy = energy.masked_fill(mask, -1e10)

        attention = F.softmax(energy, dim=-1)
        # [batch_size, n_heads, seq_len, seq_len]

        context = torch.matmul(attention, v)
        # [batch_size, n_heads, seq_len, d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        # [batch_size, seq_len, n_heads * d_v]
        
        output = self.linear_o(context)
        # [batch_size, seq_len, d_model]
        return output


