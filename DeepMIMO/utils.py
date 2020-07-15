import yaml
import torch
import numpy as np
from sklearn.metrics import mean_squared_error


# accuracy
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred)/y_true.shape[0]


# bits per second
def bps(target, pred, mfactor=1, snr=1e10):
    """
        target : true channel gains.
        pred : predicted channel gains.
    """
    target = mfactor*construction(target)
    pred = mfactor*construction(pred)
    upper_bound_examples = np.log2(1+snr*np.linalg.norm(target, axis=1)**2)
    upper_bound = np.mean(upper_bound_examples)
    pred_bps_examples = np.log2(1+snr*np.abs(np.sum(np.conj(target)*pred, axis=1))**2/np.abs(np.sum(np.conj(pred)*pred, axis=1)))
    pred_bps = np.mean(pred_bps_examples)
    return pred_bps, upper_bound


# construction
def construction(data):
    """
    construct the array by concatenating real and imag values.
    params: data: numpy array for reconstruct.
    """
    if data.dtype == 'complex':
        return np.hstack((data.real, data.imag))
    else:
        return data


# cross entropy
def crossentropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1-eps)
    loss = -np.sum(y_true*np.log(y_pred), 1)
    loss = np.sum(loss)/y_true.shape[0]
    return loss


# calucalating noise variance
def EbNo2Sigma(ebnodb, k, n):
    ebno = 10**(ebnodb/10)
    bits_per_complex_symbol = k/(n/2)
    return 1.0/np.sqrt(bits_per_complex_symbol*ebno)


# inverse fourier trasnform
def ifft_transform(data):
    """
    take the ifft and returns the real and imag values.
    params: data: numpy array for ifft. data must have real and imag concatenated
    """
    if data.shape[1] % 2 == 0:
        ifft_data = np.fft.ifft(data, axis=1, norm='ortho')
        return ifft_data
    else:
        raise Exception("Array have {} dimension which is not accepted.".format(data.shape[1]))


# mean squared error.
def mse(y_true, y_pred):
    """
    params: y_true: numpy array of shape (m,n)
            y_pred: numpy array of shape (m,n)
    """
    return mean_squared_error(y_true, y_pred)


# normalized mean squared error.
def nmse(y_true, y_pred):
    """
    params: y_true: numpy array of shape (m,n)
            y_pred: numpy array of shape (m,n)
    """
    return np.mean(np.linalg.norm(y_true-y_pred, 2, axis=1)**2/np.linalg.norm(y_true, 2, axis=1)**2)


# converting array of numbers to one hot vectors of certain depth d
def one_hot(s, d):
    if isinstance(s, int):
        s = np.array([s])
    b = np.zeros((len(s), d))
    b[np.arange(len(s)), s] = 1
    return b


# for reading yaml file
def read_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            f = yaml.safe_load(stream)
            return f
        except Exception as e:
            raise Exception("{} is not present. Check the path".format(filename))


# reconstruction
def reconstruction(data):
    """
    reconstruct the array by combining real and imag values.
    params: data: numpy array for reconstruct.
    """
    if data.shape[1] % 2 == 0:
        cutpoint = data.shape[1]//2
        return data[:, :cutpoint] + data[:, cutpoint:]*1j
    else:
        raise Exception("Array have {} dimension which is not accepted.".format(data.shape[1]))


# set's random seed
def set_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
