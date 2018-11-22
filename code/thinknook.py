import  torch
from torch.utils.data.dataset import Dataset
import numpy as np
# DATASET
class ThinkNook(Dataset):
    def __init__(self, from_cache, name):
        print("LOADING")
        self.file = np.load('tweets.p.npy')
        # self.file = self.load_from_cache()
        # self.samples = self.load_from_cache() if from_cache else self.load()
        self.size = self.file.shape[0]
        # self.size = self.samples.size
        # self.size = self.file['tweets'].shape[0]
        print("LOADED {} SAMPLES".format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        vec = self.file[item]

        tweet = np.zeros((70,))
        tweet[:len(vec)] = vec


        tweet = torch.tensor(tweet, dtype=torch.long)

        length = len(vec)

        return tweet, length

    def collate(self, samples):
        # Sort batch by sequence length and pack
        inputs, lengths = zip(*samples)

        input_tensor = torch.stack(list(inputs))
        lengths = torch.tensor(lengths, dtype=torch.uint8, requires_grad=False)

        lengths, perm_index = lengths.sort(0, descending=True)

        input_tensor = input_tensor[perm_index]
        input_tensor = input_tensor.permute(1, 0).contiguous()

        return input_tensor, lengths


