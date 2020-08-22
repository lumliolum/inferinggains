import numpy as np
from torch.utils.data import Dataset
from utils import one_hot, bin_seq


class DeepMIMODataset(Dataset):
    def __init__(self, input_data, feats, output_data,
                 input_mean, input_std, feats_mean,
                 feats_std, output_mean, output_std,
                 user_indices=None, add_noise=False, snr=np.inf):
        """
        params: input_data : input data numpy array
                feats : extra features like distance, coordinates. If no feats, pass None.
                output_data : output data numpy array
                input_mean : mean of the input data
                input_std : standard deviation of the input data
                feats_mean : mean of the feats data. If no feats, pass None
                feats_std : standard deviation of the feats data. If no feats, pass None
                output_mean : mean of the output data
                output_std : standard deviation of output data
                user_indices : indices of users that will be included
                                None will mean all users.
                add_noise : Boolean, If True, will add the noise to channel gains
                snr : if add_noise is True, then this will determine the varaince of noise added.
        """
        self.input_data = input_data
        self.feats = feats
        self.output_data = output_data
        self.input_mean = input_mean
        self.input_std = input_std
        self.feats_mean = feats_mean
        self.feats_std = feats_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.user_indices = user_indices
        self.add_noise = add_noise
        self.snr = snr

    def __len__(self):
        if self.user_indices is None:
            return self.input_data.shape[0]
        else:
            return len(self.user_indices)

    def __getitem__(self, index):
        if self.user_indices is None:
            input_data = self.input_data[index]
            output_data = self.output_data[index]
        else:
            input_data = self.input_data[self.user_indices[index]]
            output_data = self.output_data[self.user_indices[index]]
        if self.feats is not None:
            if self.user_indices is None:
                feats_data = self.feats[index]
            else:
                feats_data = self.feats[self.user_indices[index]]
            feats_data = (feats_data-self.feats_mean)/self.feats_std
        if self.add_noise:
            # input_data = input_data + np.random.normal(0, self.noise_std, size=input_data.shape)
            # add noise according to user's channel gains
            self.snr_linear = np.power(10, self.snr/10)
            nstd = np.sqrt((2*np.mean(input_data**2)/(2*self.snr_linear)))
            input_data = input_data + np.random.normal(0, nstd, size=input_data.shape)
        # centralizing the data
        input_data = (input_data-self.input_mean)/self.input_std
        output_data = (output_data-self.output_mean)/self.output_std
        if self.feats is not None:
            input_data = np.append(input_data, feats_data)
        return {"uplink": input_data, "downlink": output_data}


class AutoEncoderDataset(Dataset):
    def __init__(self, input_data, feats,
                 downlink_data, k, M, input_mean, input_std, feats_mean,
                 feats_std, downlink_mean, downlink_std, method, user_indices=None, 
                 generate_messages=True, message_matrix=None):
        self.input_data = input_data
        self.feats = feats
        self.downlink_data = downlink_data
        self.k = k
        self.M = M
        self.input_mean = input_mean
        self.input_std = input_std
        self.feats_mean = feats_mean
        self.feats_std = feats_std
        self.downlink_mean = downlink_mean
        self.downlink_std = downlink_std
        self.method = method
        self.user_indices = user_indices
        self.generate_messages = generate_messages
        self.message_matrix = message_matrix

    def __len__(self):
        if self.user_indices is None:
            return self.input_data.shape[0]
        else:
            return len(self.user_indices)

    def __getitem__(self, index):
        if self.user_indices is None:
            input_data = self.input_data[index]
            downlink_data = self.downlink_data[index]
        else:
            input_data = self.input_data[self.user_indices[index]]
            downlink_data = self.downlink_data[self.user_indices[index]]
        if self.feats is not None:
            if self.user_indices is None:
                feats_data = self.feats[index]
            else:
                feats_data = self.feats[self.user_indices[index]]
            feats_data = (feats_data - self.feats_mean)/self.feats_std
        # centralizing the data
        input_data = (input_data-self.input_mean)/(self.input_std)
        downlink_data = (downlink_data-self.downlink_mean)/self.downlink_std
        if self.generate_messages:
            # creating the message signal (one hot vector)
            message = np.random.randint(low=0, high=2**self.k)
            if self.method=='one_hot':
                message = np.squeeze(one_hot(message, d=self.M))
            elif self.method=='bin_seq':
                message = np.squeeze(bin_seq(message, d=self.M))
        else:
            message = self.message_matrix[index,:]
        if self.feats is not None:
            input_data = np.append(input_data, feats_data)
        input_data = np.append(input_data, message)
        return {"input": input_data,
                'message': message,
                'downlink': downlink_data}
