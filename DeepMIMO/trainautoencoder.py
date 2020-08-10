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

from models import AutoEncoder
from losses import CrossEntropyLoss
from datasets import AutoEncoderDataset
from utils import set_seed, construction, crossentropy, \
                  accuracy, read_yaml, EbNo2Sigma, one_hot


def main():
    x = datetime.datetime.now()
    # inputs
    filename = "config.yaml"
    config = read_yaml(filename)
    params = config['autoencoder']

    fname = params['model_save_path']
    feats_to_include = params['feats_to_include']
    k = params['k']
    n = params['n']
    M = 2**k
    training_ebnodb = params['training_ebnodb']
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']
    epochs = params['epochs']
    seed = params['seed']
    device = params['device']

    set_seed(seed, device)
    input_data = loadmat(params['input_path'])['channelgains']
    downlink_data = loadmat(params['downlink_path'])['channelgains']
    locations = loadmat(params['locations_path'])['locations']
    # reshaping the array as (m,*)
    input_data = input_data.reshape((input_data.shape[0], -1))
    downlink_data = downlink_data.reshape((downlink_data.shape[0], -1))

    input_data = construction(input_data)
    downlink_data = construction(downlink_data)
    feats = locations[:, feats_to_include]
    num_rows, num_ant = input_data.shape
    num_ant = num_ant//2
    _, feats_dim = feats.shape
    _, output_dim = downlink_data.shape

    row_indices = np.arange(num_rows)
    # shuffling the indices
    np.random.shuffle(row_indices)
    # train,val and test split is 3:1:1 implies train is 60% of total
    train_size, val_size, test_size = params['train_val_test_split']
    train_num_rows = int(num_rows*train_size)
    val_num_rows = int(num_rows*val_size)
    test_num_rows = num_rows - train_num_rows - val_num_rows
    train_row_indices = row_indices[:train_num_rows]
    val_row_indices = row_indices[train_num_rows:-test_num_rows]
    test_row_indices = row_indices[-test_num_rows:]
    print("train size = {} val size = {} test size = {}".format(train_num_rows, val_num_rows, test_num_rows))

    # calculating mean and std over training set
    input_mean = np.mean(input_data[train_row_indices, :], axis=0)
    input_std = np.std(input_data[train_row_indices, :], axis=0)

    if feats_dim != 0:
        feats_mean = np.mean(feats[train_row_indices, :], axis=0)
        feats_std = np.std(feats[train_row_indices, :], axis=0)
    else:
        feats, feats_mean, feats_std = None, None, None
    downlink_mean = np.mean(downlink_data[train_row_indices, :], axis=0)
    downlink_std = np.std(downlink_data[train_row_indices, :], axis=0)

    # generating message vectors for test data.
    test_message_matrix = one_hot(np.random.randint(low=0, high=M, 
                                                     size=(test_num_rows, )), d=M)
    
    train_dataset = AutoEncoderDataset(input_data, feats, downlink_data, M,
                                       input_mean, input_std, feats_mean,
                                       feats_std, downlink_mean, downlink_std,
                                       user_indices=train_row_indices)

    val_dataset = AutoEncoderDataset(input_data, feats, downlink_data, M,
                                     input_mean, input_std, feats_mean,
                                     feats_std, downlink_mean, downlink_std,
                                     user_indices=val_row_indices)

    test_dataset = AutoEncoderDataset(input_data, feats, downlink_data, M,
                                      input_mean, input_std, feats_mean,
                                      feats_std, downlink_mean, downlink_std,
                                      user_indices=test_row_indices, generate_messages=False,
                                      message_matrix=test_message_matrix)


    # model initialization
    print("input dim = {0}, message length = {1}, number of channel uses = {2}".format(2*num_ant+feats_dim+M,M,n))
    print("learning rate = {0}, batch size = {1}, epochs = {2}".format(lr, batch_size, epochs))
    model = AutoEncoder(M=M,
                        n=n,
                        num_ant=num_ant,
                        input_dim=2*num_ant+feats_dim+M,
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
    loss_fn = CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    optimizer.zero_grad()
    train_loss = []
    val_loss = []
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
            input_data = inputs['input'].float().to(device)
            downlink_data = inputs['downlink'].float().to(device)
            m = inputs['message'].float().to(device)
            m_hat = model.forward(input_data, downlink_data, noise_std)
            loss = loss_fn.forward(m, m_hat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_preds[index:index+input_data.shape[0], :] = m_hat.cpu().detach().squeeze().numpy()
            train_targets[index:index+input_data.shape[0], :] = m.cpu().detach().squeeze().numpy()
            index = index + input_data.shape[0]
            logger = str(index)+'/'+str(len(train_dataset))
            if batch == len(train_loader)-1:
                print(logger)
            else:
                print(logger, end='\r')
        
        # validation
        val_preds = np.zeros((val_num_rows, M))
        val_targets = np.zeros((val_num_rows, M))
        index = 0
        with torch.no_grad():
            model.eval()
            for batch, inputs in enumerate(val_loader):
                input_data = inputs['input'].float().to(device)
                downlink_data = inputs['downlink'].float().to(device)
                m = inputs['message'].float().to(device)
                m_hat = model.forward(input_data, downlink_data, noise_std)
                val_preds[index:index+input_data.shape[0], :] = m_hat.cpu().detach().squeeze().numpy()
                val_targets[index:index+input_data.shape[0], :] = m.cpu().detach().squeeze().numpy()
                index = index + input_data.shape[0]
                logger = str(index)+'/'+str(len(val_dataset))
                if batch == len(val_loader)-1:
                    print(logger)
                else:
                    print(logger, end='\r')
        train_cross_entropy = crossentropy(train_targets, train_preds)
        val_cross_entropy = crossentropy(val_targets, val_preds)
        train_loss.append(train_cross_entropy)
        val_loss.append(val_cross_entropy)
        t2 = datetime.datetime.now()
        print("Seconds = {} train loss = {} val loss = {}".format(
            round((t2-t1).total_seconds()), round(train_cross_entropy, 5),
            round(val_cross_entropy, 5)))
        if val_cross_entropy < best_cross_entropy:
            print("val cross entropy decreases from {} to {}. Saving the model at {}".format(round(best_cross_entropy, 5), round(val_cross_entropy, 5), fname))
            torch.save(model.state_dict(), fname)
            best_cross_entropy = val_cross_entropy
    
    print("The validation entropy is", round(best_cross_entropy), 5)
    # loading the model
    model = AutoEncoder(M=M,
                        n=n,
                        num_ant=num_ant,
                        input_dim=2*num_ant+feats_dim+M,
                        device=device)
    model.load_state_dict(torch.load(fname))
    model.eval()
    
    # testing(for various ranges of SNR)
    lb,ub = params['testing_ebnodb'][0],params['testing_ebnodb'][1]
    snr_range = np.linspace(lb,ub,ub-lb+1)
    bler = np.zeros((len(snr_range),))
    ber = np.zeros((len(snr_range),))
    for i in tqdm.tqdm(range(len(snr_range))):
        noise_std = EbNo2Sigma(snr_range[i], k, n)
        test_preds = np.zeros((test_num_rows, M))
        test_targets = np.zeros((test_num_rows, M))
        index = 0
        with torch.no_grad():
            for batch, inputs in enumerate(test_loader):
                input_data = inputs['input'].float().to(device)
                downlink_data = inputs['downlink'].float().to(device)
                m = inputs['message'].float().to(device)
                m_hat = model.forward(input_data, downlink_data, noise_std)
                test_preds[index:index+input_data.shape[0]] = m_hat.cpu().detach().squeeze().numpy()
                test_targets[index:index+input_data.shape[0]] = m.cpu().detach().squeeze().numpy()
                index = index + input_data.shape[0]

        # converting to one hot vector
        idx = np.argmax(test_preds, axis=1)
        test_preds = np.zeros(test_preds.shape)
        test_preds[np.arange(test_preds.shape[0]), idx] = 1

        accuracy = (np.argmax(test_targets, axis=1) == np.argmax(test_preds, axis=1))
        accuracy = np.mean(accuracy)
        block_error_rate = 1.0 - accuracy
        bitaccuracy = np.mean((test_targets == test_preds))
        bit_error_rate = 1.0-bitaccuracy
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
            input_data = inputs['input'].float().to(device)
            downlink_data = inputs['downlink'].float().to(device)
            m = inputs['message'].float().to(device)
            m_hat = model.forward(input_data, downlink_data, noise_std)
            test_preds[index:index+input_data.shape[0]] = m_hat.cpu().detach().squeeze().numpy()
            test_targets[index:index+input_data.shape[0]] = m.cpu().detach().squeeze().numpy()
            index = index + input_data.shape[0]
            logger = str(index)+'/'+str(len(test_dataset))
            print(logger, end='\r')
    
    # converting to one hot vector
    idx = np.argmax(test_preds, axis=1)
    test_preds = np.zeros(test_preds.shape)
    test_preds[np.arange(test_preds.shape[0]), idx] = 1

    accuracy = (np.argmax(test_targets, axis=1) == np.argmax(test_preds, axis=1))
    accuracy = np.mean(accuracy)
    bler_no_noise = 1.0 - accuracy
    bitaccuracy = np.mean((test_targets == test_preds))
    ber_no_noise = 1.0-bitaccuracy
    print("The bler for zero noise is", round(bler_no_noise), 7)
    print("The ber for zero noise is", round(ber_no_noise), 7)

    # plotting the train loss and validation loss
    plt.plot(list(range(1, epochs+1)), train_loss, label='train')
    plt.plot(list(range(1, epochs+1)), val_loss, label='validation')
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.legend()
    plt.savefig(params['loss_save_path'])
    plt.show()
    plt.close()

    # results
    results = pd.DataFrame()
    results['snr'] = snr_range
    results['bler'] = bler
    results['ber'] = ber
    results.to_csv(params['blerber_save_path'], index=False)
    print(results.head())

    # plotting the figures
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.plot(snr_range, bler, linewidth=2.0)
    plt.legend(['Autoencoder'], prop={'size': 10}, loc='upper right')
    plt.yscale('log')
    plt.xlabel('EbNo (dB)', fontsize=12)
    plt.ylabel('Block-error rate', fontsize=12)
    plt.grid(True)

    plt.subplot(122)
    plt.plot(snr_range, ber, linewidth=2.0)
    plt.legend(['Autoencoder'], prop={'size': 10}, loc='upper right')
    plt.yscale('log')
    plt.xlabel('EbNo (dB)', fontsize=12)
    plt.ylabel('Bit-error rate', fontsize=12)
    plt.grid(True)
    plt.savefig(params['blerbercurve_save_path'])
    plt.show()
    plt.close()

    y = datetime.datetime.now()
    print("Completed in {} seconds".format(round((y-x).total_seconds(), 5)))

if __name__ == "__main__":
    main()
