import normflows as nf
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class realNVP():
    def __init__(self, samples, weights, target_num_params):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.target_num_params = target_num_params
        self.samples = torch.from_numpy(samples).to(self.device)
        self.weights = torch.from_numpy(weights).to(self.device)

        self.base = nf.distributions.base.DiagGaussian(2)
        self.default_num_layers = 32
        self.default_num_params = 141380
        self.num_layers = round(self.default_num_layers*
                                self.target_num_params/self.default_num_params)
        self.flows = []
        for i in range(self.num_layers):
            param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
            self.flows.append(nf.flows.AffineCouplingBlock(param_map))
            self.flows.append(nf.flows.Permute(2, mode='swap'))
        self.model = nf.NormalizingFlow(self.base, self.flows)
        self.model = self.model.to(self.device)
    
    def train(self, max_iter=4000, learning_rate=1e-3, patience=100):
        idx = random.sample(range(len(self.samples)), round(len(self.samples)*0.2))
        idx = np.array(idx)
        test_x = self.samples[idx]
        x = self.samples[~idx]
        test_weights = weights[idx]
        weights = weights[~idx]

        loss_hist = np.array([])
        test_loss_hist = np.array([])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        c = 0
        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            
            # Compute loss
            loss = torch.mean(-self.model.log_prob(x)*weights)
            test_loss = torch.mean(-self.model.log_prob(test_x)*test_weights)
            
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
            
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            test_loss_hist = np.append(test_loss_hist, test_loss.to('cpu').data.numpy())
            
            c += 1
            if it == 0:
                minimum_loss = test_loss_hist[-1]
                minimum_epoch = it
                minimum_model = None
            else:
                if test_loss_hist[-1] < minimum_loss:
                    minimum_loss = test_loss_hist[-1]
                    minimum_epoch = it
                    minimum_model = self.model
                    c = 0
            if minimum_model:
                if c == patience:
                    print('Early stopped. Epochs used = ' + str(it) +
                            '. Minimum at epoch = ' + str(minimum_epoch))
                    self.model = minimum_model
                    break
    
    def kl(self, posterior_probs, weights):
        prob = self.model.log_prob(self.x).to('cpu').data.numpy()

        posterior_probs[np.isnan(prob)] = 0

        KL = np.average(posterior_probs - prob,
                            weights=weights / weights.sum())
        return KL
