#%%
from cgi import test
from pyexpat import model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from zmq import device
from sklearn.metrics import precision_score, recall_score, f1_score
# %%
data_frame = pd.read_csv('fzf_new.csv')
data_frame.head()
#%%
def u_matrix(data,num_items,num_users):
    inter = np.zeros((num_users, num_items))
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) 
        inter[user_index, item_index] = score
    return inter
#%%
n_unique_users = len(data_frame['user_id'].unique())
n_unique_items = len(data_frame['item_id'].unique())
print(f'Number of unique users: {n_unique_users}')
print(f'Number of unique items: {n_unique_items}')
#%%
# seed = 1234
train_data, val_data = train_test_split(data_frame, test_size=0.1, shuffle=True)
train_data_u = u_matrix(train_data, num_users=n_unique_users, num_items=n_unique_items).astype(np.float32)
val_data_u = u_matrix(val_data, num_users=n_unique_users, num_items=n_unique_items).astype(np.float32)
#%%
# Data Loader
class TourismDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


#%%
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, epsilon_std=1.0):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
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
        x_bar = self.sigmoid(self.x_bar(h_decoder))
        return x_bar, z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.size()).to(device)

        return z_mean + epsilon * torch.exp(z_log_var / 2) * self.epsilon_std



# %%
class VAE_Loss(nn.Module):
    def __init__(self):
        super(VAE_Loss, self).__init__()

    def forward(self, x, x_recon, z_mean, z_log_var):
        BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return BCE + KLD

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(input_dim=n_unique_items, hidden_dim=12, latent_dim=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = VAE_Loss()
# %%
train_loader = DataLoader(TourismDataset(train_data_u), batch_size=32, shuffle=True)
val_loader = DataLoader(TourismDataset(val_data_u), batch_size=32, shuffle=False)
epoch_num = 1000
#%%
for epoch in range(epoch_num):
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, z_mean, z_log_var = model(data)
        loss = criterion(data, x_recon, z_mean, z_log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    model.eval()
    val_loss = 0
    for i, data in enumerate(val_loader):
        data = data.to(device)
        x_recon, z_mean, z_log_var = model(data)
        loss = criterion(data, x_recon, z_mean, z_log_var)
        val_loss += loss.item()
    print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}')

# %%
def compute_precision_recall_f1(prediction, ground_truth, threshold=0.5):
    prediction = prediction.reshape(-1)
    ground_truth = ground_truth.reshape(-1)
    prediction = prediction > threshold
    precision = precision_score(ground_truth, prediction)
    recall = recall_score(ground_truth, prediction)
    f1 = f1_score(ground_truth, prediction)
    return precision, recall, f1
# %%
model.eval()
prediction = model(torch.from_numpy(train_data_u).to(device))

# %%
prediction = prediction[0].detach().cpu().numpy()

#%%
def val_pre_gro(prediction, val_data):
    pred = []
    gro = []
    for i in range(len(prediction)):
        idx = val_data[val_data['user_id'] == i+1]['item_id'].values-1
        val_raing = val_data[val_data['user_id'] == i+1]['rating'].values
        if val_raing.size > 0:
            pred.append(prediction[i][idx].tolist())
            gro.append(val_raing.tolist())
    return pred, gro


# %%
pred, gro = val_pre_gro(prediction, val_data)
# %%
# turn a list of list to a list
pred = np.array([item for sublist in pred for item in sublist])
gro = np.array([item for sublist in gro for item in sublist])

for thr in [0.25,0.3,0.35,0.4,0.45,0.50,0.55,0.60,0.65,0.70]:
    print(f'Threshold: {thr}')
    print(f'Precision: {compute_precision_recall_f1(pred, gro, threshold=thr)[0]}')
    print(f'Recall: {compute_precision_recall_f1(pred, gro, threshold=thr)[1]}')
    print(f'F1: {compute_precision_recall_f1(pred, gro, threshold=thr)[2]}')
    print('\n')



#%%%
# change train_data coulmn names
