import yaml
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import FNN
from datasets import DeepMIMODataset
from losses import MSELoss, NMSELoss
from utils import read_yaml, set_seed, construction, mse, nmse, bps, reconstruction


def main():
    # params
    x = datetime.datetime.now()
    filename = "config.yaml"
    config = read_yaml(filename)
    params = config['fnn']
    seed = params['seed']
    device = torch.device(params['device'])
    hidden_dim = params['hidden_dim']
    feats_to_include = params['feats_to_include']
    fname = params['model_save_path']
    lr = params['lr']
    weight_decay = params['weight_decay']
    batch_size = params['batch_size']
    epochs = params['epochs']
    add_noise = params['add_noise']
    snr = params['snr']
    savepath = params['savefig']
    set_seed(seed, device)

    # reading the data
    input_data = loadmat(params['input_path'])['channelgains']
    output_data = loadmat(params['output_path'])['channelgains']
    locations = loadmat(params['locations_path'])['locations']
    # reshaping the array as (m,*)
    input_data = input_data.reshape((input_data.shape[0], -1))
    output_data = output_data.reshape((output_data.shape[0], -1))

    input_data = construction(input_data)
    output_data = construction(output_data)
    feats = locations[:, feats_to_include]
    num_rows, input_dim = input_data.shape
    _, feats_dim = feats.shape
    _, output_dim = output_data.shape

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
    output_mean = np.mean(output_data[train_row_indices, :], axis=0)
    output_std = np.std(output_data[train_row_indices, :], axis=0)
    # output_mean = 0
    # output_std = 1e-5
    train_dataset = DeepMIMODataset(input_data, feats, output_data, input_mean, input_std, feats_mean, feats_std,
                                    output_mean, output_std, user_indices=train_row_indices, add_noise=add_noise, snr=snr)
    val_dataset = DeepMIMODataset(input_data, feats, output_data, input_mean, input_std, feats_mean, feats_std,
                                  output_mean, output_std, user_indices=val_row_indices, add_noise=add_noise, snr=snr)
    test_dataset = DeepMIMODataset(input_data, feats, output_data, input_mean, input_std, feats_mean, feats_std,
                                   output_mean, output_std, user_indices=test_row_indices, add_noise=add_noise, snr=snr)
    # model initialization
    print("input Dimension = {0}, hidden Dimension = {1}, output Dimension = {2}".format(input_dim+feats_dim, hidden_dim, output_dim))
    print("learning rate = {0}, batch size = {1}, epochs = {2}".format(lr, batch_size, epochs))
    model = FNN(input_dim+feats_dim, hidden_dim, output_dim)
    model.float()
    model.to(device)

    # loss function and optimizer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    loss_fn = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    train_loss = []
    val_loss = []
    train_metric = []
    val_metric = []
    best_val_nmse = +np.inf

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        t1 = datetime.datetime.now()

        # training
        train_preds = np.zeros((train_num_rows, output_dim))
        train_targets = np.zeros((train_num_rows, output_dim))
        index = 0
        model.train()
        for batch, inputs in enumerate(train_loader):
            X = inputs['uplink'].float().to(device)
            y = inputs['downlink'].float().to(device)
            y_pred = model.forward(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_preds[index:index+X.shape[0], :] = y_pred.cpu().detach().squeeze().numpy()
            train_targets[index:index+X.shape[0], :] = y.cpu().detach().squeeze().numpy()
            index = index + X.shape[0]
            logger = str(index)+'/'+str(len(train_dataset))
            if batch == len(train_loader)-1:
                print(logger)
            else:
                print(logger, end='\r')
        # validation
        val_preds = np.zeros((val_num_rows, output_dim))
        val_targets = np.zeros((val_num_rows, output_dim))
        index = 0
        with torch.no_grad():
            model.eval()
            for batch, inputs in enumerate(val_loader):
                X = inputs['uplink'].float().to(device)
                y = inputs['downlink'].float().to(device)
                y_pred = model.forward(X)
                val_preds[index:index+X.shape[0], :] = y_pred.cpu().detach().squeeze().numpy()
                val_targets[index:index+X.shape[0], :] = y.cpu().detach().squeeze().numpy()
                index = index + X.shape[0]
                logger = str(index)+'/'+str(len(val_dataset))
                if batch == len(val_loader)-1:
                    print(logger)
                else:
                    print(logger, end='\r')
        train_mse = mse(train_targets, train_preds)
        val_mse = mse(val_targets, val_preds)
        train_nmse = nmse(train_targets, train_preds)
        val_nmse = nmse(val_targets, val_preds)
        train_loss.append(train_mse)
        val_loss.append(val_mse)
        train_metric.append(train_nmse)
        val_metric.append(val_nmse)
        t2 = datetime.datetime.now()
        print("Seconds = {} train mse = {} val mse = {} train nmse = {} val nmse = {}".format(
            round((t2-t1).total_seconds()), round(train_mse, 5), round(val_mse, 5), round(train_nmse, 5), round(val_nmse, 5)))
        if val_nmse < best_val_nmse:
            print("val nmse decreased from {} to {}.Saving the model at {}".format(round(best_val_nmse, 5), round(val_nmse, 5), fname))
            torch.save(model.state_dict(), fname)
            best_val_nmse = val_nmse

    # saving the graph
    plt.plot(list(range(1, epochs+1)), train_metric, label='train')
    plt.plot(list(range(1, epochs+1)), val_metric, label='validation')
    plt.xlabel("epoch")
    plt.ylabel("nmse")
    plt.legend()
    plt.savefig(savepath)
    plt.close()

    print("The nmse on validation is", best_val_nmse)
    # evaluation on test data
    print("Evaluating on test data")
    # loading the saved model
    model = FNN(input_dim+feats_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(fname))
    test_preds = np.zeros((test_num_rows, output_dim))
    test_targets = np.zeros((test_num_rows, output_dim))
    index = 0
    with torch.no_grad():
        model.eval()
        for batch, inputs in enumerate(test_loader):
            X = inputs['uplink'].float().to(device)
            y = inputs['downlink'].float().to(device)
            y_pred = model.forward(X)
            test_preds[index:index+X.shape[0], :] = y_pred.cpu().detach().squeeze().numpy()
            test_targets[index:index+X.shape[0], :] = y.cpu().detach().squeeze().numpy()
            index = index+X.shape[0]
            logger = str(index)+'/'+str(len(test_dataset))
            if batch == len(test_loader)-1:
                print(logger)
            else:
                print(logger, end='\r')
    test_nmse = nmse(test_targets, test_preds)
    print("Test nmse = {}".format(test_nmse))
    # print("Compute BPS", bps(test_targets,  test_preds, mfactor=output_std))

    predict_for_autoencoder = params['predict_for_autoencoder']
    if predict_for_autoencoder:
        print("For autoencoder")
        # new_input_data = loadmat("DeepMIMO Dataset/deepmimo_dataset_I1_2p4_128_xant_1_ofdm_5_paths_second_half.mat")['channelgains']
        # new_output_data = loadmat("DeepMIMO Dataset/deepmimo_dataset_I1_2p5_128_xant_1_ofdm_5_paths_second_half.mat")['channelgains']
        # new_locations = loadmat("DeepMIMO Dataset/locations_second_half.mat")['locations']

        # # reshaping
        # new_input_data = new_input_data.reshape((new_input_data.shape[0], -1))
        # new_output_data = new_output_data.reshape((new_output_data.shape[0], -1))

        # new_input_data = construction(new_input_data)
        # new_output_data = construction(new_output_data)
        # new_feats = new_locations[:,feats_to_include]
        # if feats_dim==0:
        #     new_feats = None
        # prediction on data
        test_dataset = DeepMIMODataset(input_data, feats, output_data, input_mean, input_std, feats_mean, feats_std,
                                       output_mean, output_std, user_indices=np.arange(input_data.shape[0]), add_noise=False, snr=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # will use the loaded model
        test_preds = np.zeros((len(test_dataset), output_dim))
        test_targets = np.zeros((len(test_dataset), output_dim))
        index = 0
        with torch.no_grad():
            model.eval()
            for batch,inputs in enumerate(test_loader):
                X = inputs['uplink'].float().to(device)
                y = inputs['downlink'].float().to(device)
                y_pred = model.forward(X)
                test_preds[index:index+X.shape[0], :] = y_pred.cpu().detach().squeeze().numpy()
                test_targets[index:index+X.shape[0], :] = y.cpu().detach().squeeze().numpy()
                index = index+X.shape[0]
                logger = str(index)+'/'+str(len(test_dataset))
                if batch == len(test_loader)-1:
                    print(logger)
                else:
                    print(logger, end='\r')
        # nmse computation
        test_nmse_autoencoder = nmse(test_targets, test_preds)
        print("Test nmse = {}".format(test_nmse_autoencoder))
        # multiply by std and add mean to predictions
        test_preds = output_std*test_preds + output_mean
        # converting to complex array
        test_preds = reconstruction(test_preds)
        print("Saving the file at",params['autoencoder_save_path'])
        # saving the predictions
        savemat(
                file_name=params['autoencoder_save_path'],
                mdict={"channelgains":test_preds},
                appendmat=False
               )
    
    y = datetime.datetime.now()
    print("Completed in {} seconds".format(round((y-x).total_seconds(), 5)))


if __name__ == '__main__':
    main()
