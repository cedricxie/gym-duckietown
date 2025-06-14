import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import gym
import gym_duckietown


def generate_duckietown_dataset(env_name='Duckietown-straight_road-v0', seq_len=50, num_sequences=32):
    env = gym.make(env_name, domain_rand=False)
    obs_dim = (3, 64, 64)
    action_dim = 2

    data_obs = []
    data_acts = []
    for _ in tqdm(range(num_sequences)):
        obs = env.reset()
        traj_obs = []
        traj_acts = []
        for _ in range(seq_len):
            action = np.array([0.0, 0.3], dtype=np.float64)
            traj_acts.append(action.copy())
            obs, reward, terminated, info_dict = env.step(action)
            obs = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)
            obs = F.interpolate(obs.float(), size=64, mode='bilinear', align_corners=False) / 255.0
            traj_obs.append(obs.squeeze(0))
            if terminated:
                break
        if len(traj_obs) == seq_len:
            data_obs.append(torch.stack(traj_obs))
            data_acts.append(torch.tensor(traj_acts, dtype=torch.float32))
    env.close()

    obs_tensor = torch.stack(data_obs)
    act_tensor = torch.stack(data_acts)
    return obs_tensor, act_tensor, obs_dim, action_dim

class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim):
        super().__init__()
        C, H, W = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class CNNDecoder(nn.Module):
    def __init__(self, in_dim, obs_shape):
        super().__init__()
        C, H, W = obs_shape
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class RSSM(nn.Module):
    def __init__(self, state_dim=64, hidden_dim=128, obs_shape=(3, 64, 64), action_dim=2):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape, hidden_dim)
        self.decoder = CNNDecoder(state_dim, obs_shape)
        self.rnn = nn.GRUCell(hidden_dim + action_dim + state_dim, hidden_dim)
        self.state_prior = nn.Linear(hidden_dim, state_dim * 2)
        self.state_posterior = nn.Linear(hidden_dim + hidden_dim, state_dim * 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.rnn.hidden_size, device=device)

    def reparameterize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_seq, act_seq):
        B, T, C, H, W = obs_seq.shape
        device = obs_seq.device
        h = self.init_hidden(B, device)
        s = torch.zeros(B, self.state_prior.out_features // 2, device=device)
        decoded, post_means, prior_means = [], [], []

        for t in range(T):
            o_enc = self.encoder(obs_seq[:, t])
            a_t = act_seq[:, t]
            h = self.rnn(torch.cat([o_enc, a_t, s], dim=-1), h)

            prior_stats = self.state_prior(h)
            prior_mu, prior_log_std = torch.chunk(prior_stats, 2, dim=-1)

            post_input = torch.cat([h, o_enc], dim=-1)
            post_stats = self.state_posterior(post_input)
            post_mu, post_log_std = torch.chunk(post_stats, 2, dim=-1)

            s = self.reparameterize(post_mu, post_log_std)
            o_dec = self.decoder(s)
            decoded.append(o_dec)
            post_means.append(post_mu)
            prior_means.append(prior_mu)

        return torch.stack(decoded, dim=1), torch.stack(post_means, dim=1), torch.stack(prior_means, dim=1)

def train_rssm():
    obs_data, act_data, obs_shape, act_dim = generate_duckietown_dataset()
    loader = DataLoader(TensorDataset(obs_data, act_data), batch_size=8, shuffle=True)
    model = RSSM(obs_shape=obs_shape, action_dim=act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    for epoch in tqdm(range(400)):
        total_loss = 0
        total_los_recon = 0
        total_loss_kl = 0
        for obs_seq, act_seq in loader:
            model.train()
            decoded, post_mu, prior_mu = model(obs_seq, act_seq)
            recon_loss = mse_loss(decoded, obs_seq)
            kl_loss = torch.mean((post_mu - prior_mu) ** 2)
            loss = recon_loss + 0.1 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_los_recon += recon_loss.item()
            total_loss_kl += kl_loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, {total_los_recon:.4f}, {total_loss_kl:.4f}")

    return model, obs_data, act_data

def demo():
    model, obs_data, act_data = train_rssm()
    test_obs = obs_data[0:1]
    test_act = act_data[0:1]
    model.eval()
    with torch.no_grad():
        decoded, _, _ = model(test_obs, test_act)
    for t in range(5):
        plt.subplot(2, 5, t + 1)
        plt.imshow(test_obs[0, t].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.subplot(2, 5, 5 + t + 1)
        plt.imshow(decoded[0, t].permute(1, 2, 0).numpy())
        plt.axis('off')
    plt.suptitle("Duckietown Observation Reconstruction")
    # plt.show()
    plt.savefig("demo_reconstruction.png")  # Save the plot to a file

if __name__ == '__main__':
    demo()
