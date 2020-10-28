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
    def __init__(self, M, n, num_ant, input_dim, method, device=torch.device('cpu')):
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
        self.method = method
        self.device = device

        # encoder layer(transmitter)
        self.encoder_layer1 = nn.Linear(in_features=self.input_dim, out_features=2*self.num_ant*self.n)
        self.encoder_layer2 = nn.Linear(in_features=2*self.num_ant*self.n, out_features=4*self.num_ant*self.n)
        self.encoder_layer3 = nn.Linear(in_features=4*self.num_ant*self.n, out_features=4*self.num_ant*self.n)
        self.encoder_layer4 = nn.Linear(in_features=4*self.num_ant*self.n, out_features=2*self.num_ant*self.n)

        # decoder layer(receiver)
        self.decoder_layer1 = nn.Linear(in_features=2*self.n, out_features=self.M)
        self.decoder_layer2 = nn.Linear(in_features=self.M, out_features=self.M)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data, downlink_data, noise_std):
        h_encoder = self.relu(self.encoder_layer2(self.relu(self.encoder_layer1(input_data))))
        h_encoder = self.encoder_layer4(self.relu(self.encoder_layer3(h_encoder)))

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
        if self.method=='one_hot':
            h_decoder = self.softmax(self.decoder_layer2(h_decoder))
        elif self.method=='bin_seq':
            h_decoder = self.sigmoid(self.decoder_layer2(h_decoder))
        return h_decoder

class AutoEncoder_v2(nn.Module):
    def __init__(self, M, n, num_ant, feats_dim, method, scheme, device=torch.device('cpu')):
        """
            .....
        """
        super(AutoEncoder_v2, self).__init__()
        self.M = M
        self.n = n
        self.num_ant = num_ant
        self.feats_dim = feats_dim
        self.method = method
        self.scheme = scheme
        self.device = device

        # selected schemes for pred network
        self.sel_schemes = [4,5]

        if self.scheme in self.sel_schemes:
            # prediction network
            self.prednetwork = nn.Sequential(
                                                nn.Linear(2*self.num_ant+self.feats_dim, 2*self.num_ant),
                                                nn.ReLU(),
                                                nn.Linear(2*num_ant, 4*self.num_ant),
                                                nn.ReLU(),
                                                nn.Linear(4*self.num_ant, 4*self.num_ant),
                                                nn.ReLU(),
                                                nn.Linear(4*self.num_ant, 2*self.num_ant),
                                                )
        # encoder network
        self.encodernetwork = nn.Sequential(
                                                nn.Linear(self.M, self.M//2),
                                                nn.ReLU(),
                                                nn.Linear(self.M//2, 2*self.n),
                                            )

        # decoder network
        self.decodernetwork = nn.Sequential(
                                                nn.Linear(2*self.n, self.M),
                                                nn.ReLU(),
                                                nn.Linear(self.M, self.M),
                                                nn.Softmax(dim=1),
                                            )
    
    def forward(self, channel, message, downlink_data, noise_std, hhat):
        """
            channel : channel vector + feats(optional)
            message : one hot message vector
            downlink: downlink_data
        """
        if self.scheme in self.sel_schemes:
            hhat = self.prednetwork(channel)
        else:
            hhat = hhat

        x = self.encodernetwork(message)
        for i in range(self.n):
            _,size = hhat.shape
            x_i = x[:,2*i:2*(i+1)]
            sr = hhat[:,:size//2]*x_i[:,[0]] - hhat[:,size//2:]*x_i[:,[1]]
            si = hhat[:,:size//2]*x_i[:,[1]] + hhat[:,size//2:]*x_i[:,[0]]
            s = torch.cat([sr,si], -1)
            if i==0:
                h_encoder = s
            else:
                h_encoder = torch.cat((h_encoder, s), dim=1)
        # normalization
        norms = torch.sqrt(torch.sum(h_encoder**2)/(h_encoder.shape[0]*self.n))
        h_encoder = h_encoder/norms
        for i in range(self.n):
            h = h_encoder[:,2*self.num_ant*i:2*self.num_ant*(i+1)]
            _,size = h.shape
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

        h_decoder = self.decodernetwork(h_decoder)
        return hhat, h_decoder