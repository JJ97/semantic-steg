import torch

from .embedder import Embedder
from constants import *

class OneHotChar(Embedder):

    def __init__(self, chars, device):
        super().__init__()
        self.chars = list(chars)
        self.char2index = {c : i for (i,c) in enumerate(chars)}
        self.device = device
        self.input_size = len(chars)
        self.start_tensor = torch.tensor([[self.char2index[START]]], dtype=torch.long, device=self.device).view(-1, 1)
        self.end_tensor = torch.tensor([[self.char2index[END]]], dtype=torch.long, device=self.device).view(-1, 1)

    def embed(self, tweet):
        tweet = START + tweet + END
        indices = [self.char2index[c] if c in self.char2index else self.char2index[c] for c in tweet]
        tensor = torch.tensor(indices, dtype=torch.long, device=self.device).view(-1, 1)
        return tensor

    def unembed(self, decoder_outputs):
        return ''.join(self.chars[index] for index in decoder_outputs)

