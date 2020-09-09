# code to the split the dataset in two halfs
# nothing to do with running the model
import numpy as np
from scipy.io import loadmat,savemat

seed = 342
np.random.seed(seed)

# load the 128 ant dataset
uplink = loadmat("DeepMIMODataset/deepmimo_dataset_I1_2p4_128_xant_1_ofdm_5_paths.mat")['channelgains']
downlink = loadmat("DeepMIMODataset/deepmimo_dataset_I1_2p5_128_xant_1_ofdm_5_paths.mat")['channelgains']
locations = loadmat("DeepMIMODataset/locations.mat")['locations']

# split the dataset in half
row_indices = np.arange(uplink.shape[0])
np.random.shuffle(row_indices)
first_half_indices = row_indices[:int(0.5*uplink.shape[0])]
second_half_indices = row_indices[int(0.5*uplink.shape[0]):]

# first half
uplink_first_half = uplink[first_half_indices]
downlink_first_half = downlink[first_half_indices]
locations_first_half = locations[first_half_indices]

# second half
uplink_second_half = uplink[second_half_indices]
downlink_second_half = downlink[second_half_indices]
locations_second_half = locations[second_half_indices]

print("First half shape",uplink_first_half.shape, downlink_first_half.shape, locations_first_half.shape)
print("Seond half shape",uplink_second_half.shape, downlink_second_half.shape, locations_second_half.shape)

# saving the files
# first half
savemat(
        file_name="DeepMIMODataset/deepmimo_dataset_I1_2p4_128_xant_1_ofdm_5_paths_first_half.mat",
        mdict={'channelgains':uplink_first_half},
        appendmat=False
)
savemat(
        file_name="DeepMIMODataset/deepmimo_dataset_I1_2p5_128_xant_1_ofdm_5_paths_first_half.mat",
        mdict={'channelgains':downlink_first_half},
        appendmat=False
)
savemat(
        file_name="DeepMIMODataset/locations_first_half.mat",
        mdict={'locations':locations_first_half},
        appendmat=False
)
# second half
savemat(
        file_name="DeepMIMODataset/deepmimo_dataset_I1_2p4_128_xant_1_ofdm_5_paths_second_half.mat",
        mdict={'channelgains':uplink_second_half},
        appendmat=False
)
savemat(
        file_name="DeepMIMODataset/deepmimo_dataset_I1_2p5_128_xant_1_ofdm_5_paths_second_half.mat",
        mdict={'channelgains':downlink_second_half},
        appendmat=False
)
savemat(
        file_name="DeepMIMODataset/locations_second_half.mat",
        mdict={'locations':locations_second_half},
        appendmat=False
)
