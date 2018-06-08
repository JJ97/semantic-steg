from io import open
import unicodedata
import string
import re
import random
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from lang import Language
from input import read_file, readCache

FILE = 'deu-eng/deu.txt'
CACHE = ('deu-am', 'eng-am', 'eng2deu-am')
INCLUDE_PHRASES = ["i am", 'you are', 'we are', 'they are', 'she is', 'he is']

USE_CACHE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if USE_CACHE:
    input_lang, output_lang, pairs = (readCache(c) for c in CACHE)
else:
    input_lang, output_lang, pairs = read_file(FILE, INCLUDE_PHRASES, suffix='am')

print(input_lang.name)




