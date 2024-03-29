import os
import yaml
import tqdm
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .models import AutoEncoder_v2
from .losses import CrossEntropyLoss
from .datasets import AutoEncoderDataset
from .utils import set_seed, construction, crossentropy, \
                  accuracy, read_yaml, EbNo2Sigma, one_hot, \
                  bin_seq, set_params_autoencoder, nmse


def main():
    x = datetime.datetime.now()
    # inputs
    filename = "config.yaml"
    config = read_yaml(filename)
    params = config['autoencoder']

    params['version'] = 2
    scheme = params['scheme']
    mode = params['mode']
    method = params['method']
    num_ant = params['num_ant']
    params = set_params_autoencoder(params)

    # setting up the directory for results
    if not os.path.isdir(params['results_dir']):
        os.makedirs(params['results_dir'])

    feats_to_include = params['feats_to_include']
    k = params['k']
    n = params['n']

    # assigning value M
    if method=="one_hot":
        M = 2**k
    elif method=='bin_seq':
        M = k
    else:
        raise Exception("{} method not available".format(method))

    training_ebnodb = params['training_ebnodb']
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']
    epochs = params['epochs']
    seed = params['seed']
    device = params['device']

    set_seed(seed, device)
    input_data = loadmat(params['input_path'])['channelgains']
    downlink_channel = loadmat(params['downlink_path'])['channelgains']
    locations = loadmat(params['locations_path'])['locations']
    # reshaping the array as (m,*)
    input_data = input_data.reshape((input_data.shape[0], -1))
    downlink_channel = downlink_channel.reshape((downlink_channel.shape[0], -1))

    input_data = construction(input_data)
    downlink_channel = construction(downlink_channel)
    feats = locations[:, feats_to_include]
    num_rows, _ = input_data.shape
    _, feats_dim = feats.shape
    _, output_dim = downlink_channel.shape

    row_indices = np.arange(num_rows)
    # shuffling the indices
    np.random.shuffle(row_indices)
    train_size, val_size, test_size = params['train_val_test_split']
    train_num_rows = int(num_rows*train_size)
    val_num_rows = int(num_rows*val_size)
    test_num_rows = num_rows - train_num_rows - val_num_rows
    train_row_indices = row_indices[:train_num_rows]
    val_row_indices = row_indices[train_num_rows:-test_num_rows]
    test_row_indices = row_indices[-test_num_rows:]

    print("scheme = {0}, method = {1}, mode={2}".format(scheme, method, mode))
    print("train size = {} val size = {} test size = {}".format(train_num_rows, val_num_rows, test_num_rows))

    # calculating mean and std over training set
    input_mean = np.mean(input_data[train_row_indices, :], axis=0)
    input_std = np.std(input_data[train_row_indices, :], axis=0)

    if feats_dim != 0:
        feats_mean = np.mean(feats[train_row_indices, :], axis=0)
        feats_std = np.std(feats[train_row_indices, :], axis=0)
    else:
        feats, feats_mean, feats_std = None, None, None

    downlink_mean = np.mean(downlink_channel[train_row_indices, :], axis=0)
    downlink_std = np.std(downlink_channel[train_row_indices, :], axis=0)

    # generating message vectors for test data.
    test_message_matrix = np.random.randint(low=0, high=2**k,
                                                     size=(test_num_rows, ))
    if method=='one_hot':
        test_message_matrix = one_hot(test_message_matrix, d=M)
    elif method=='bin_seq':
        test_message_matrix = bin_seq(test_message_matrix, d=M)

    train_dataset = AutoEncoderDataset(input_data, feats, downlink_channel, k, M,
                                       input_mean, input_std, feats_mean,
                                       feats_std, downlink_mean, downlink_std, method=method,
                                       indices=train_row_indices, generate_messages=True)

    val_dataset = AutoEncoderDataset(input_data, feats, downlink_channel, k, M,
                                     input_mean, input_std, feats_mean,
                                     feats_std, downlink_mean, downlink_std, method=method,
                                     indices=val_row_indices, generate_messages=True)

    test_dataset = AutoEncoderDataset(input_data, feats, downlink_channel, k, M,
                                      input_mean, input_std, feats_mean,
                                      feats_std, downlink_mean, downlink_std, method=method,
                                      indices=test_row_indices, generate_messages=False,
                                      message_matrix=test_message_matrix)

    print("channel_dim = {0}, feats_dim = {1}, message length = {2}, number of channel uses = {3}".format(2*num_ant, feats_dim, M, n))
    print("learning rate = {0}, batch size = {1}, epochs = {2}".format(lr, batch_size, epochs))
    # model initialization
    model = AutoEncoder_v2(M=M,
                           n=n,
                           num_ant=num_ant,
                           feats_dim=feats_dim,
                           method=method,
                           scheme=scheme,
                           mode=mode,
                           device=device)
    model.float()
    model.to(device)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size, shuffle=False)

    # loss function and optimizer
    loss_fn = CrossEntropyLoss(method)

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    optimizer.zero_grad()
    train_loss = []
    val_loss = []
    val_error = []
    best_cross_entropy = +np.inf
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        t1 = datetime.datetime.now()
        noise_std = EbNo2Sigma(training_ebnodb, k, n)

        # training
        train_preds = np.zeros((train_num_rows, M))
        train_targets = np.zeros((train_num_rows, M))
        index = 0
        model.train()
        for batch, inputs in enumerate(train_loader):
            channel = inputs['channel'].float().to(device)
            channel_only = inputs['channel_only'].float().to(device)
            m = inputs['message'].float().to(device)
            downlink_data = inputs['downlink'].float().to(device)

            _, m_hat = model.forward(channel, m, downlink_data, noise_std, channel_only)
            loss = loss_fn.forward(m, m_hat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_preds[index:index+channel.shape[0], :] = m_hat.cpu().detach().squeeze().numpy()
            train_targets[index:index+channel.shape[0], :] = m.cpu().detach().squeeze().numpy()

            index = index + channel.shape[0]
            logger = str(index)+'/'+str(len(train_dataset))
            if batch == len(train_loader)-1:
                print(logger)
            else:
                print(logger, end='\r')
        
        # validation
        val_channel_preds = np.zeros((val_num_rows, 2*params['num_ant']))
        val_preds = np.zeros((val_num_rows, M))
        val_targets = np.zeros((val_num_rows, M))
        index = 0
        with torch.no_grad():
            model.eval()
            for batch, inputs in enumerate(val_loader):
                channel = inputs['channel'].float().to(device)
                m = inputs['message'].float().to(device)
                channel_only = inputs['channel_only'].float().to(device)
                downlink_data = inputs['downlink'].float().to(device)

                hhat, m_hat = model.forward(channel, m, downlink_data, noise_std, channel_only)

                val_channel_preds[index:index+channel.shape[0], :] = hhat.cpu().detach().squeeze().numpy()
                val_preds[index:index+channel.shape[0], :] = m_hat.cpu().detach().squeeze().numpy()
                val_targets[index:index+channel.shape[0], :] = m.cpu().detach().squeeze().numpy()

                index = index + channel.shape[0]
                logger = str(index)+'/'+str(len(val_dataset))
                if batch == len(val_loader)-1:
                    print(logger)
                else:
                    print(logger, end='\r')
        train_cross_entropy = crossentropy(train_targets, train_preds)
        val_cross_entropy = crossentropy(val_targets, val_preds)
        val_nmse = nmse(downlink_channel[val_row_indices], downlink_std*val_channel_preds + downlink_mean)

        train_loss.append(train_cross_entropy)
        val_loss.append(val_cross_entropy)
        val_error.append(val_nmse)

        t2 = datetime.datetime.now()
        print("Seconds = {} train loss = {} val loss = {}".format(
            round((t2-t1).total_seconds()), round(train_cross_entropy, 5),
            round(val_cross_entropy, 5)))
        if val_cross_entropy < best_cross_entropy:
            print("val cross entropy decreases from {} to {}. Saving the model at {}".format(round(best_cross_entropy, 5), round(val_cross_entropy, 5), params['model_save_path']))
            torch.save(model.state_dict(), params['model_save_path'])
            best_cross_entropy = val_cross_entropy
            best_nmse = val_nmse

    print("The validation entropy is {}".format(round(best_cross_entropy, 5)))
    print("The validataion nmse is {}".format(round(best_nmse, 5)))
    # loading the model
    model = AutoEncoder_v2(M=M,
                           n=n,
                           num_ant=num_ant,
                           feats_dim=feats_dim,
                           method=method,
                           scheme=scheme,
                           mode=mode,
                           device=device)
    model.load_state_dict(torch.load(params['model_save_path']))
    model.eval()

    # testing(for various ranges of SNR)
    lb,ub = params['testing_ebnodb'][0],params['testing_ebnodb'][1]
    snr_range = np.linspace(lb,ub,ub-lb+1)
    bler = np.zeros((len(snr_range),))
    ber = np.zeros((len(snr_range),))
    lossvalues = np.zeros((len(snr_range),))

    for i in tqdm.tqdm(range(len(snr_range))):
        noise_std = EbNo2Sigma(snr_range[i], k, n)
        test_preds = np.zeros((test_num_rows, M))
        test_targets = np.zeros((test_num_rows, M))
        index = 0
        with torch.no_grad():
            for batch, inputs in enumerate(test_loader):
                channel = inputs['channel'].float().to(device)
                channel_only = inputs['channel_only'].float().to(device)
                m = inputs['message'].float().to(device)
                downlink_data = inputs['downlink'].float().to(device)

                _, m_hat = model.forward(channel, m, downlink_data, noise_std, channel_only)

                test_preds[index:index+channel.shape[0]] = m_hat.cpu().detach().squeeze().numpy()
                test_targets[index:index+channel.shape[0]] = m.cpu().detach().squeeze().numpy()
                index = index + channel.shape[0]
        lossvalues[i] = crossentropy(test_targets, test_preds)
        if method=='one_hot':
            # converting to one hot vector
            idx = np.argmax(test_preds, axis=1)
            test_preds = np.zeros(test_preds.shape)
            test_preds[np.arange(test_preds.shape[0]), idx] = 1
            accuracy = np.mean((np.sum((test_targets == test_preds), axis=1)) == M)
            block_error_rate = 1.0 - accuracy
            bitaccuracy = np.mean((test_targets == test_preds))
            bit_error_rate = 1.0 - bitaccuracy
            bler[i] = block_error_rate
            ber[i] = bit_error_rate
        elif method=='bin_seq':
            test_preds = (test_preds>0.5).astype(int)
            accuracy = np.mean((np.sum((test_targets == test_preds), axis=1)) == M)
            block_error_rate = 1.0 - accuracy
            bitaccuracy = np.mean((test_targets == test_preds))
            bit_error_rate = 1.0 - bitaccuracy
            bler[i] = block_error_rate
            ber[i] = bit_error_rate

    # testing for no noise
    noise_std = 0
    test_preds = np.zeros((test_num_rows, M))
    test_targets = np.zeros((test_num_rows, M))
    index = 0
    with torch.no_grad():
        model.eval()
        for batch,inputs in enumerate(test_loader):
            channel = inputs['channel'].float().to(device)
            channel_only = inputs['channel_only'].float().to(device)
            m = inputs['message'].float().to(device)
            downlink_data = inputs['downlink'].float().to(device)

            _, m_hat = model.forward(channel, m, downlink_data, noise_std, channel_only)

            test_preds[index:index+channel.shape[0]] = m_hat.cpu().detach().squeeze().numpy()
            test_targets[index:index+channel.shape[0]] = m.cpu().detach().squeeze().numpy()
            index = index + channel.shape[0]
            logger = str(index)+'/'+str(len(test_dataset))
            print(logger, end='\r')

    if method=='one_hot':
        # converting to one hot vector
        idx = np.argmax(test_preds, axis=1)
        test_preds = np.zeros(test_preds.shape)
        test_preds[np.arange(test_preds.shape[0]), idx] = 1

        accuracy = np.mean((np.sum((test_targets == test_preds), axis=1)) == M)
        bler_no_noise = 1.0 - accuracy
        bitaccuracy = np.mean((test_targets == test_preds))
        ber_no_noise = 1.0 - bitaccuracy
    elif method=='bin_seq':
        test_preds = (test_preds>0.5).astype(int)
        accuracy = np.mean((np.sum((test_targets == test_preds), axis=1)) == M)
        bler_no_noise = 1.0 - accuracy
        bitaccuracy = np.mean((test_targets == test_preds))
        ber_no_noise = 1.0 - bitaccuracy

    print("The bler for zero noise is", round(bler_no_noise,7))
    print("The ber for zero noise is", round(ber_no_noise, 7))

    # loss values
    loss = pd.DataFrame()
    loss['epoch'] = np.arange(1, params['epochs']+1)
    loss['train_cross_entropy'] = train_loss
    loss['val_cross_entropy'] = val_loss
    loss['val_nmse'] = val_error
    path = os.path.join(params['results_dir'], params['loss_file_name'])
    loss.to_csv(path, index=False)
    print(loss.head())

    # results
    results = pd.DataFrame()
    results['snr'] = snr_range
    results['bler'] = bler
    results['ber'] = ber
    results['loss'] = lossvalues
    path = os.path.join(params['results_dir'], params['blerber_file_name'])
    results.to_csv(path, index=False)
    print(results.head())

    # saving the config used to the folder
    params['config_filename'] = "config.yaml"
    yaml.dump(params, open(os.path.join(params['results_dir'], params['config_filename']), 'w'))

    y = datetime.datetime.now()
    print("Completed in {} seconds".format(round((y-x).total_seconds(), 5)))

if __name__ == "__main__":
    main()
