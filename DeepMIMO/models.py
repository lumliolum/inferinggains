import torch
import numpy as np
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        params: input_dim : input dimension
                hidden_dim : hidden layer dimension
                output_dim : output dimension
        """
        super(FNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if isinstance(self.hidden_dim, int):
            self.hidden_dim = list(self.hidden_dim)
        elif not isinstance(self.hidden_dim, list):
            raise Exception("list or int is expected for hidden_dim but got {}".format(type(self.hidden_dim)))

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim[0])

        self.hidden_layers = nn.ModuleList([])
        for i in range(1, len(self.hidden_dim)):
            self.hidden_layers.append(nn.Linear(in_features=self.hidden_dim[i-1], out_features=self.hidden_dim[i]))

        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim[-1], self.output_dim)

    def forward(self, X):
        h = self.relu(self.input_layer(X))
        for hidden_layer in self.hidden_layers:
            h = self.relu(hidden_layer(h))
        out = self.output_layer(h)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, M, n, num_ant, input_dim, device=torch.device('cpu')):
        """
            M: length of message vector
            n: number of channel uses
            num_ant: number of antennas
            input_dim : dimension of the input(for encoder)
            device: torch.device on which model runs
        """
        super(AutoEncoder, self).__init__()
        self.M = M
        self.n = n
        self.num_ant = num_ant
        self.input_dim = input_dim
        self.device = device

        # encoder layer(transmitter)
        self.encoder_layer1 = nn.Linear(in_features=self.input_dim, out_features=2*self.num_ant*self.n)
        self.encoder_layer2 = nn.Linear(in_features=2*self.num_ant*self.n, out_features=2*self.num_ant*self.n)

        # decoder layer(receiver)
        self.decoder_layer1 = nn.Linear(in_features=2*self.n, out_features=self.M)
        self.decoder_layer2 = nn.Linear(in_features=self.M, out_features=self.M)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data, downlink_data, noise_std):
        h_encoder = self.relu(self.encoder_layer1(input_data))
        h_encoder = self.encoder_layer2(h_encoder)
        # normalization
        norms = torch.sqrt(torch.sum(h_encoder**2)/(h_encoder.shape[0]*self.n))
        h_encoder = h_encoder/norms
        for i in range(self.n):
            h = h_encoder[:,2*self.num_ant*i:2*self.num_ant*(i+1)]
            _,size = downlink_data.shape

            rxr = torch.sum(downlink_data[:, :size//2]*h[:, :size//2], dim=1)
            rxi = torch.sum(downlink_data[:, size//2:]*h[:, size//2:], dim=1)

            ryr = torch.sum(downlink_data[:, :size//2]*h[:, size//2:], dim=1)
            ryi = torch.sum(downlink_data[:, size//2:]*h[:, :size//2], dim=1)

            rx = rxr - rxi
            ry = ryr + ryi
            r = torch.cat([torch.unsqueeze(rx,-1), torch.unsqueeze(ry,-1)], dim=1)
            noise = np.random.normal(loc=0,scale=noise_std, size=(r.shape[0], r.shape[1]))
            noise = torch.tensor(noise).float().to(self.device)
            r = r + noise
            if i==0:
                h_decoder = r
            else:
                h_decoder = torch.cat([h_decoder, r], dim=1)

        h_decoder = self.relu(self.decoder_layer1(h_decoder))
        h_decoder = self.softmax(self.decoder_layer2(h_decoder))
        return h_decoder
