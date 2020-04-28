import  torch
from torch.utils.data.dataset import Dataset
import h5py
import nltk
import html
import re
import numpy as np
import mmap
import os

# DATASET
class BookCorpus(Dataset):
    def __init__(self):
        print("LOADING")
        self.filename = '/home/hfsk44/bookcorpus/all_filt.txt'
        # self.file = self.load_from_cache()
        # self.samples = self.load_from_cache() if from_cache else self.load()
        f = open(self.filename, 'rb')
        self.size = sum(1 for line in f)
        f.seek(0)
        self.linestart = [0]
        for l in f:
            self.linestart.append(self.linestart[-1] + len(l))
        # self.size = self.samples.size
        # self.size = self.file['tweets'].shape[0]
        print("LOADED {} SAMPLES".format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        f = open(self.filename, 'rb')
        ff = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        t = ff[self.linestart[item]:self.linestart[item+1]]
        t = t.decode("ascii", 'ignore')
        t = t.lower()[:140]
        #t = html.unescape(t)
        #t = t.encode('ascii', 'ignore').decode("utf-8")
        #t = re.sub(r'http[^\s]*($|\s)', r'https ', t)
        #t = re.sub(r'@[^\s]*($|\s)', r'@ ', t)
        #t = re.sub(r'#[^\s]*($|\s)', r'# ', t)
        #t = nltk.casual_tokenize(t, preserve_case=True, reduce_len=True, strip_handles=True)
        #t = ' '.join(t)
        return t, len(t.split())

    def collate(self, samples):
        # Sort batch by sequence length and pack
        inputs, lengths = zip(*samples)

        #input_tensor = torch.stack(list(inputs))
        lengths = torch.tensor(lengths, dtype=torch.uint8, requires_grad=False)
        #lengths, perm_index = lengths.sort(0, descending=True)

        #input_tensor = input_tensor[perm_index]
        # input_tensor = input_tensor.permute(1, 0).contiguous()

        return inputs, lengths