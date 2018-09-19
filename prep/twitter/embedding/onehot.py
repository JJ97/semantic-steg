import torch

from .embedder import Embedder
from constants import *

class OneHot(Embedder):

    def __init__(self, wordlist, device):
        super().__init__()
        self.wordlist = list(wordlist)
        self.word2index = {w : i for (i,w) in enumerate(wordlist)}
        self.device = device
        self.input_size = len(wordlist)
        self.start_tensor = torch.tensor([[self.word2index[START]]], dtype=torch.long, device=self.device).view(-1, 1)
        self.end_tensor = torch.tensor([[self.word2index[END]]], dtype=torch.long, device=self.device).view(-1, 1)

    def embed(self, tweet):
        tweet = ' '.join([START] + tweet + [END])
        indices = [self.word2index[word] if word in self.word2index else self.word2index[UNK] for word in tweet.split(' ')]
        tensor = torch.tensor(indices, dtype=torch.long, device=self.device).view(-1, 1)
        return tensor

    def unembed(self, index):
        return self.wordlist[index]

