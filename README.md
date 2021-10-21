# INTRODUCTION

## Uplink-Downlink Channel Prediction

Scaling the number of antennas up is a key characterstic of current and future wireless
systems. Realizing the multiplexing and beamforming gains of large number of antennas
requires channel knowledge. Usually channel feedback from users are used to get the channel
knowledge. This results in signalling overhead and wastage of radio resources.
Suppose we know the channels between a user and a certain set of antennas at one frequency
band, can we map this knowledge to the channels at a different set of antennas and at
different frequency band? Essentially this mapping means that we can directly predict the
downlink channel gains from the uplink channel gains, eliminating the downlink training
and feedback overhead in co-located/distributed FDD massive MIMO systems. Following
[1], we use deep neural networks (DNN) to approximate the mapping between uplink and
downlink channel gains, and the freely available Deep MIMO Dataset [2] will be used for
training and testing.

## Performance Evaluation of AutoEncoder

Following [3], we present a communication system (transmitter, channel and receiver) as an
autoencoder, and design an end to end reconstruction task that seeks to jointly optimize
transmitter and receiver components in a single process. We compare the performance
of end to end autoencoder for five different schemes. The key idea here is to represent
transmitter, channel, and receiver as one deep neural network (NN) that can be trained as
an autoencoder.

## Code Details

The configuration file is `config.yaml`. It has configurations for DNN (`fnn`), autoencoder and varitional autoencoder. The code uses pytorch for implementation.

To run DNN, use

```bash
python train.py -c config.yaml
```

To run autoencoder, use

```bash
python trainautoencoder.py
```

or (V2 is different architecture of AE we tried)

```bash
python trainautoencoder_v2.py
```

To run Variational Autoencoder, use

```bash
python train_vae.py -c config.yaml
```

## References

[1] M. Alrabeiah and A. Alkhateeb, “Deep learning for tdd and fdd massive mimo: Map-
ping channels in space and frequency,” arXiv preprint arXiv:1905.03761, 2019.

[2] A. Alkhateeb, “Deepmimo: A generic deep learning dataset for millimeter wave and
massive mimo applications,” arXiv preprint arXiv:1902.06435, 2019.

[3] J. Xu, W. Chen, B. Ai, R. He, Y. Li, J. Wang, T. Juhana, and A. Kurniawan, “Perfor-
mance evaluation of autoencoder for coding and modulation in wireless communica-
tions,” in 2019 11th International Conference on Wireless Communications and Signal
Processing (WCSP). IEEE, 2019, pp. 1–6.
