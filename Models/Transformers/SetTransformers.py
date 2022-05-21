import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TransformerBlocks import *


class SetTransformer(nn.Module):
    def __init__(self, n_layers, d_model, heads, d_ff, dropout):
        super(SetTransformer, self).__init__()
        pass

    def forward(self, src, trg, src_mask, trg_mask):
        pass




