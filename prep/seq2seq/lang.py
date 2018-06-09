import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from consts import *

class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2 : "UNK"}
        self.n_words = 3  # Count SOS, EOS, UNK

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence2tensor(self, sentence, device):
        indices = [self.word2index[word] if word in self.word2index else UNK_TOKEN for word in sentence.split(' ')]
        indices.append(EOS_TOKEN)
        return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)