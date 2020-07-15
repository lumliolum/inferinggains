# inspired from https://colab.research.google.com/drive/1HwD1QM7uiSfj9qHbkqMTCVUQ-589DUFU#scrollTo=2u8Ufw5AuwmW
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class AutoEncoder(nn.Module):
    def __init__(self, M, n, device=torch.device('cpu')):
        super(AutoEncoder, self).__init__()
        self.M = M
        self.n = n
        self.device = device

        # encoder layers(transmitter)
        self.encoder_layer1 = nn.Linear(in_features=self.M+1, out_features=self.M)
        self.encodoer_layer2 = nn.Linear(in_features=self.M, out_features=self.n)

        # decoder layers (reciever)
        self.decoder_layer1 = nn.Linear(in_features=self.n, out_features=self.M)
        self.decoder_layer2 = nn.Linear(in_features=self.M, out_features=self.M)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batches, h, noise_std, is_noise=True):
        batches = torch.cat((batches, h), dim=1)
        tx = self.relu(self.encoder_layer1(batches))
        tx = self.encodoer_layer2(tx)
        norms = torch.sqrt(torch.mean(tx**2))
        x = tx/norms
        noise = np.random.normal(loc=0, scale=noise_std, size=(x.shape[0], x.shape[1]))
        noise = torch.tensor(noise).float().to(self.device)
        if is_noise:
            y = h*x + noise
        else:
            y = h*x
        # y = torch.cat((y,h),dim=1)
        rx = self.relu(self.decoder_layer1(y))
        m_hat = self.softmax(self.decoder_layer2(rx))
        return m_hat


class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, self.eps, 1-self.eps)

        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, 1)
        if len(y_pred.shape) == 1:
            y_pred = torch.unsqueeze(y_pred, 1)

        loss = -torch.sum(y_true*torch.log(y_pred), 1)
        loss = torch.sum(loss)/y_true.shape[0]
        return loss


def EbNo2Sigma(ebnodb, k, n):
    ebno = 10**(ebnodb/10)
    bits_per_complex_symbol = k/(n/2)
    return 1.0/np.sqrt(bits_per_complex_symbol*ebno)

def one_hot(s, depth):
    if isinstance(s, int):
        s = np.array([s])
    b = np.zeros((len(s),depth))
    b[np.arange(len(s)),s] = 1
    return b

def set_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# main parameters of the autoencoder
k = 4          # Number of information bits per message, i.e., M=2**k
n = 4          # Number of real channel uses per message
M = 2**k         # Number of messages
seed = 42
device = torch.device('cpu')
set_seed(seed, device)

loss_fn = CrossEntropyLoss()
model = AutoEncoder(M, n, device)
# training
model.train()
for i in tqdm.tqdm(range(1000)):
    batch_size = 100
    lr = 1e-3
    training_ebnodb = 25
    noise_std = EbNo2Sigma(training_ebnodb, k, n)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    # message
    m = np.random.randint(low=0, high=M, size=(batch_size,))
    m = one_hot(m, depth=M)
    # channel
    h = 1/np.sqrt(2)*np.random.normal(0, 1, size=(batch_size, 1))
    m = torch.tensor(m).float().to(device)
    h = torch.tensor(h).float().to(device)
    m_hat = model.forward(m, h, noise_std, is_noise=True)
    loss = loss_fn.forward(m, m_hat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

for i in tqdm.tqdm(range(10000)):
    batch_size = 100
    lr = 1e-4
    training_ebnodb = 25
    noise_std = EbNo2Sigma(training_ebnodb, k, n)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    # message
    m = np.random.randint(low=0, high=M, size=(batch_size,))
    m = one_hot(m, depth=M)
    # channel
    h = 1/np.sqrt(2)*np.random.normal(0, 1, size=(batch_size, 1))
    m = torch.tensor(m).float().to(device)
    h = torch.tensor(h).float().to(device)
    m_hat = model.forward(m, h, noise_std, is_noise=True)
    loss = loss_fn.forward(m, m_hat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

for i in tqdm.tqdm(range(10000)):
    batch_size = 1000
    lr = 1e-4
    training_ebnodb = 25
    noise_std = EbNo2Sigma(training_ebnodb, k, n)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    # message
    m = np.random.randint(low=0, high=M, size=(batch_size,))
    m = one_hot(m, depth=M)
    # channel
    h = 1/np.sqrt(2)*np.random.normal(0, 1, size=(batch_size, 1))
    m = torch.tensor(m).float().to(device)
    h = torch.tensor(h).float().to(device)
    m_hat = model.forward(m, h, noise_std, is_noise=True)
    loss = loss_fn.forward(m, m_hat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(loss)
# testing (for various range of SNR)
model.eval()
snr_range = np.linspace(0, 50, 51)
# one is bit error rate and one is block error rate
monte_carlo_bler = np.zeros((len(snr_range),))
monte_carlo_ber = np.zeros((len(snr_range),))
for i in tqdm.tqdm(range(len(snr_range))):
    iterations = 10
    batch_size = 100000
    for j in range(iterations):
        noise_std = EbNo2Sigma(snr_range[i], k, n)
        m = np.random.randint(low=0, high=M, size=(batch_size,))
        m = one_hot(m, depth=M)
        h = 1/np.sqrt(2)*np.random.normal(0, 1, size=(batch_size, 1))
        with torch.no_grad():
            m = torch.tensor(m).float().to(device)
            h = torch.tensor(h).float().to(device)
            m_hat = model.forward(m, h, noise_std, is_noise=True)

            # converting to one hot vector
            idx = torch.argmax(m_hat, dim=1)
            m_hat = torch.zeros(m_hat.shape)
            m_hat[torch.arange(m_hat.shape[0]), idx] = 1

            accuracy = (torch.argmax(m, dim=1) == torch.argmax(m_hat, dim=1)).float()
            accuracy = torch.mean(accuracy)
            bler = 1.0-accuracy
            bitaccuracy = torch.mean((m == m_hat).float())
            ber = 1.0-bitaccuracy
            monte_carlo_bler[i] = monte_carlo_bler[i] + float(bler)
            monte_carlo_ber[i] = monte_carlo_ber[i] + float(ber)

monte_carlo_bler = monte_carlo_bler/iterations
monte_carlo_ber = monte_carlo_ber/iterations

# calculating the ber and bler at no noise level
iterations = 10
batch_size = 100000
monte_carlo_ber_no_noise = 0
monte_carlo_bler_no_noise = 0
for j in range(iterations):
    noise_std = EbNo2Sigma(10, k, n)
    m = np.random.randint(low=0, high=M, size=(batch_size,))
    m = one_hot(m, depth=M)
    h = 1/np.sqrt(2)*np.random.normal(0, 1, size=(batch_size, 1))
    with torch.no_grad():
        m = torch.tensor(m).float().to(device)
        h = torch.tensor(h).float().to(device)
        m_hat = model.forward(m, h, noise_std, is_noise=False)

        # converting to one hot vector
        idx = torch.argmax(m_hat, dim=1)
        m_hat = torch.zeros(m_hat.shape)
        m_hat[torch.arange(m_hat.shape[0]), idx] = 1

        accuracy = (torch.argmax(m, dim=1) == torch.argmax(m_hat, dim=1)).float()
        accuracy = torch.mean(accuracy)
        bler = 1.0-accuracy
        bitaccuracy = torch.mean((m == m_hat).float())
        ber = 1.0-bitaccuracy
        monte_carlo_bler_no_noise = monte_carlo_bler_no_noise + float(bler)
        monte_carlo_ber_no_noise = monte_carlo_ber_no_noise + float(ber)

monte_carlo_bler_no_noise = monte_carlo_bler_no_noise/iterations
monte_carlo_ber_no_noise = monte_carlo_ber_no_noise/iterations
print("The bler for zero noise is", monte_carlo_bler_no_noise)
print("The ber for zero noise is", monte_carlo_ber_no_noise)

results = pd.DataFrame()
results['snr'] = snr_range
results['bler'] = monte_carlo_bler
results['ber'] = monte_carlo_ber
print(results)

# name = "rayleigh_channel_rate_4_by_4_training_at_25_ebno_h_at_encoder"
name = "results/testing_detach_code"
plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.plot(snr_range, monte_carlo_bler, linewidth=2.0)
plt.legend(['Autoencoder'], prop={'size': 10}, loc='upper right')
plt.yscale('log')
plt.xlabel('EbNo (dB)', fontsize=12)
plt.ylabel('Block-error rate', fontsize=12)
plt.grid(True)

plt.subplot(122)
plt.plot(snr_range, monte_carlo_ber, linewidth=2.0)
plt.legend(['Autoencoder'], prop={'size': 10}, loc='upper right')
plt.yscale('log')
plt.xlabel('EbNo (dB)', fontsize=12)
plt.ylabel('Bit-error rate', fontsize=12)
plt.grid(True)
plt.savefig(name)
plt.show()
