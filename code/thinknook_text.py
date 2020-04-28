import  torch
from torch.utils.data.dataset import Dataset
import h5py
import numpy as np

# DATASET
class ThinkNookText(Dataset):
    def __init__(self):
        print("LOADING")
        self.file = np.load('/home/hfsk44/gensen/data/corpora/tweets.npy')
        self.size = self.file.shape[0]
        print("LOADED {} SAMPLES".format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        tweet = self.file[item]
        return tweet, len(tweet.split())

    def collate(self, samples):
        # Sort batch by sequence length and pack
        inputs, lengths = zip(*samples)

        # input_tensor = torch.stack(list(inputs))
        # lengths = torch.tensor(lengths, dtype=torch.uint8, requires_grad=False)
        #
        # lengths, perm_index = lengths.sort(0, descending=True)
        #
        # input_tensor = input_tensor[perm_index]
        # input_tensor = input_tensor.permute(1, 0).contiguous()

        return inputs, lengths


