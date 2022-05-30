import torch
import torch.nn as nn


class MYMSE(nn.Module):
    def __init__(self):
        super(MYMSE, self).__init__()
        '''
            arguments:
                None
            input:
                x: input data, shape: [batch_size, input_dim]
                x_recon: reconstructed data, shape: [batch_size, input_dim]

            output:
                loss: loss of VAE, shape: [1]

            note: this is a custom loss function
            The order of input is very vey important!
            it is very important you ferst input the original data,
            and then input the reconstructed data
        '''

    def forward(self, x, x_recon):
        delta_2 = torch.sum((x - x_recon*torch.sign(x))**2)
        scale = torch.sum(torch.sign(x))
        return (delta_2/scale)


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
        MSE = self.mse(x, x_recon)
        KLD = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(),dim=1), dim=0)
        return MSE + KLD