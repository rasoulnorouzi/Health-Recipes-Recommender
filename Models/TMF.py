
#%%
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
#%%
class TMF(nn.Module):
    
    def __init__(self, n_users, n_items, n_taggs, n_factors):
        super(TMF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.user_tagg_embedding = nn.Embedding(n_taggs, n_factors)
        self.item_tagg_embedding = nn.Embedding(n_taggs, n_factors)
        self.init_weights()

    def forward(self, user_id, item_id,user_taggs, item_taggs):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        user_tagg_embedding = self.user_tagg_embedding(user_taggs).sum(dim=1)
        item_tagg_embedding = self.item_tagg_embedding(item_taggs).sum(dim=1)
        r_u = user_embedding + user_tagg_embedding/user_taggs.shape[1]
        r_i = item_embedding + item_tagg_embedding/item_taggs.shape[1]
        r_u_i = torch.sum(r_u* r_i, dim=1)   
        return r_u_i
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.user_tagg_embedding.weight)
        nn.init.xavier_uniform_(self.item_tagg_embedding.weight)


#%%
df = pd.read_pickle(r'C:\Users\rasou\Desktop\Paper_Food_Recepie\Datasets\universal_df.pkl')
df.head()
# %%
class TMFCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        
        user_id, recipe_id, user_taggs, recipe_taggs, rating = zip(*batch)
        
        user_taggs = pad_sequence(user_taggs, batch_first=True, padding_value=self.pad_idx)
        recipe_taggs = pad_sequence(recipe_taggs, batch_first=True, padding_value=self.pad_idx)
        
        user_id = torch.stack(user_id)
        recipe_id = torch.stack(recipe_id)
        rating = torch.stack(rating)
        
        return user_id, recipe_id, user_taggs, recipe_taggs, rating
# %%
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
# %%
class TMFDataset(Dataset):
    def __init__(self, df, pad_idx):
        self.df = df
        self.pad_idx = pad_idx
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = self.df.iloc[idx]['user_id']
        recipe_id = self.df.iloc[idx]['recipe_id']
        user_taggs = self.df.iloc[idx]['user_taggs']
        recipe_taggs = self.df.iloc[idx]['recipe_taggs']
        rating = self.df.iloc[idx]['rating']
        return user_id, recipe_id, user_taggs, recipe_taggs, rating
# %%
