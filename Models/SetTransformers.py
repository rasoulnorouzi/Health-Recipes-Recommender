import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class SetTransformer(nn.Module):
    def __init__(self):
        super(SetTransformer, self).__init__()
        pass
    def forward(self):
        pass



# Multi Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()

        '''
        Arguments:
            d_model: the dimension of the model
            d_k: the dimension of the key
            d_v: the dimension of the value
            n_heads: the number of heads

        input:
            q: the query tensor of shape [batch_size, seq_len, d_model]
            k: the key tensor of shape [batch_size, seq_len, d_model]
            v: the value tensor of shape [batch_size, seq_len, d_model]
            
            padding_mask: the masking tensor of shape [batch_size, seq_len, seq_len],
            it is used to mask out the padding tokens in the input

        returns:
            a float tensor of shape [batch_size, seq_len, d_model]
        '''

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.linear_q = nn.Linear(d_model, d_k * n_heads) 
        self.linear_k = nn.Linear(d_model, d_k * n_heads)
        self.linear_v = nn.Linear(d_model, d_v * n_heads)
        self.linear_o = nn.Linear(d_v * n_heads, d_model)

    def forward(self, q, k, v, padding_mask):
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

        if padding_mask is not None:
            # mask out the padding tokens
            energy = energy.masked_fill(padding_mask, -1e10)
        
        attention = F.softmax(energy, dim=-1)
        # [batch_size, n_heads, seq_len, seq_len]

        context = torch.matmul(attention, v)
        # [batch_size, n_heads, seq_len, d_v]

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        # [batch_size, seq_len, n_heads * d_v]

        output = self.linear_o(context)
        # [batch_size, seq_len, d_model]

        return output