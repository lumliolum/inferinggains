# INTRODUCTION



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

## Thesis Supervisor

You can find about my supervisor Dr. Jobin Francis over [here](https://scholar.google.com/citations?user=a9Mpdm0AAAAJ)
