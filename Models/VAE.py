#%%
from audioop import rms
from cgi import test
from multiprocessing import reduction
from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%%
def visualize_loss(train_loss_epoch, val_loss_epoch):

    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["figure.dpi"] = 300

    plt.plot(train_loss_epoch)
    plt.plot(val_loss_epoch)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

#%%
def val_evaluator(tarin_test_loader, test_loader):
    model.eval()
    v_loss = []
    rm_loss = []
    train_x = [tr for tr in tarin_test_loader]
    test_x = [te for te in test_loader]
    with torch.no_grad():
        for i in range(len(train_x)):
            x = train_x[i].to(device)
            y = test_x[i].to(device)
            x_recon, z_mean, z_log_var = model(x)
            loss = vae_loss(y, x_recon, z_mean, z_log_var)
            rmse = torch.sqrt(mse(y, x_recon))
            v_loss.append(loss.item())
            rm_loss.append(rmse.item())
    print(f'*** VALIDATION ***')
    print(f'Valdiation loss: {np.mean(v_loss)}')
    print(f'Validation RMSE: {np.mean(rm_loss)}')
    return np.mean(v_loss), np.mean(rm_loss)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

class MYMSE(nn.Module):
    def __init__(self):
        super(MYMSE, self).__init__()

    def forward(self, x, x_recon):
        delta_2 = torch.sum((x - x_recon*torch.sign(x))**2)
        scale = torch.sum(torch.sign(x))
        return (delta_2/scale)

#%%
# VAE Loss CE+KL
class VAE_Loss(nn.Module):
    def __init__(self):
        super(VAE_Loss, self).__init__()

        '''
            arguments:
                None
            input:
                x: input data, shape: [batch_size, input_dim]
                x_recon: reconstructed data, shape: [batch_size, input_dim]
                z_mean: mean of latent variable, shape: [batch_size, latent_dim]
                z_log_var: log variance of latent variable, shape: [batch_size, latent_dim]
            output:
                loss: loss of VAE, shape: [1]
        '''
        self.mse = MYMSE()
    def forward(self, x, x_recon, z_mean, z_log_var):
        # BCE = F.cross_entropy(x_recon, x)
        MSE = self.mse(x, x_recon)
        KLD = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(),dim=1), dim=0)
        # return BCE + KLD
        return MSE + KLD


#%%
# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, epsilon_std=1.0):
        super(VAE, self).__init__()
        '''
            arguments:
                input_dim: dimension of input data
                hidden_dim: dimension of hidden layer
                latent_dim: dimension of latent variable
                epsilon_std: standard deviation of epsilon
            input:
                x: input data 
                shape: [batch_size, input_dim]
            output:
                x_bar: reconstructed data, shape: [batch_size, input_dim]
                z_mean: mean of latent variable, shape: [batch_size, latent_dim]
                z_log_var: log variance of latent variable, shape: [batch_size, latent_dim]
            
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.weight_init()
  
        
        self.h = nn.Linear(input_dim, hidden_dim)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.h_decoder = nn.Linear(latent_dim, hidden_dim)
        self.x_bar = nn.Linear(hidden_dim, input_dim)


    def forward(self, x):
        h = self.tanh(self.h(x))
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.sampling(z_mean, z_log_var)
        h_decoder = self.tanh(self.h_decoder(z))
        # x_bar = self.sigmoid(self.x_bar(h_decoder))
        # x_bar = self.relu(self.x_bar(h_decoder))
        x_bar = self.x_bar(h_decoder)
        return x_bar, z_mean, z_log_var

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def sampling(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.size()).to(device)

        return z_mean + epsilon * torch.exp(z_log_var / 2) * self.epsilon_std

#%%
df = pd.read_pickle(r'C:\Users\rasou\Desktop\Paper_Recipes\Health-Recipes-Recommender\universal_df.pkl')
df.head()
# %%
n_unique_users = df.user_id.nunique()
n_unique_recipes = df.recipe_id.nunique()
# %%
n_unique_recipes
# %%
n_unique_users
# %%

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
# %%

# def interaction_matrix(df, n_unique_users, n_unique_recipes, score_col='rating'):
#     inter_matrix = np.zeros((n_unique_users, n_unique_recipes))
#     for i in range(len(df)):
#         user_id = df.iloc[i].user_id
#         recipe_id = df.iloc[i].recipe_id
#         inter_matrix[user_id - 1, recipe_id - 1] = df.iloc[i][score_col]
#     return inter_matrix

def interaction_matrix(df, n_unique_users, n_unique_recipes, score_col='rating'):
    inter_matrix = np.zeros((n_unique_recipes, n_unique_users))
    for i in range(len(df)):
        user_id = df.iloc[i].user_id
        recipe_id = df.iloc[i].recipe_id
        inter_matrix[recipe_id - 1, user_id-1] = df.iloc[i][score_col]
    return inter_matrix

# %%
train_inter = interaction_matrix(train_data, n_unique_users, n_unique_recipes)
test_inter = interaction_matrix(test_data, n_unique_users, n_unique_recipes)
# %%
class VAE_Dataset(Dataset):
    def __init__(self, inter_matrix):
        self.inter_matrix = inter_matrix
        self.len = len(inter_matrix)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor(self.inter_matrix[index], dtype=torch.float)
    
# %%
train_loader = DataLoader(VAE_Dataset(train_inter), batch_size=128, shuffle=True)
train_test_loader = DataLoader(VAE_Dataset(train_inter), batch_size=128, shuffle=False)
test_loader = DataLoader(VAE_Dataset(test_inter), batch_size=128, shuffle=False)
#%%
input_dim = train_inter.shape[1]
hidden_dim = 128
latent_dim = 64
epsilon_std = 1
# %%
model = VAE(input_dim, hidden_dim, latent_dim, epsilon_std).to(device)
# %%
epochs = 100
learning_rate = 1e-3
# %%
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
vae_loss = VAE_Loss()
mse = MYMSE()
# %%
train_loss_epoc = []
train_mse_epoc = []
val_loss_epoc = []
val_mse_epoc = []

for epoch in range(epochs):
    train_loss = []
    rmse_loss = []
    model.train()
    for i, data in enumerate(train_loader):
        x = data.to(device)
        
        x_recon, z_mean, z_log_var = model(x)

    
        loss = vae_loss(x, x_recon, z_mean, z_log_var)
        rmse = torch.sqrt(mse(x, x_recon))
        train_loss.append(loss.item())
        rmse_loss.append(rmse.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'----------- epoch: {epoch+1}/{epochs} -----------')
    print(f'Train loss: {np.mean(train_loss)}')
    print(f'Train RMSE: {np.mean(rmse_loss)}')
    train_loss_epoc.append(np.mean(train_loss))
    train_mse_epoc.append(np.mean(rmse_loss))
    val_loss, val_rmse = val_evaluator(train_test_loader, test_loader)
    val_loss_epoc.append(val_loss)
    val_mse_epoc.append(val_rmse)

visualize_loss(train_loss_epoc, val_loss_epoc)
visualize_loss(train_mse_epoc, val_mse_epoc)

# %%
