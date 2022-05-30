import torch
import torch.nn as nn


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