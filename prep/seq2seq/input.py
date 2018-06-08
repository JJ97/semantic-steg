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

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    # put space in front of any .!?
    s = re.sub(r"([.!?])", r" \1", s)
    # replace any other chars with space
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_file(file, include_phrases, suffix="", reverse=False):
    if suffix:
        suffix = "-" + suffix

    print("Reading {}".format(file))

    # Read the file and split into lines
    lines = open('../../datasets/{}'.format(file), encoding='utf-8')
    lines = lines.read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    lang1, lang2 = tuple(file.split("/")[0].split('-'))

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang1)
        output_lang = Language(lang2)
    else:
        input_lang = Language(lang2)
        output_lang = Language(lang1)

    print("Counting words...")

    if include_phrases:
        filtered_pairs = []
        for pair in pairs:
            for phrase in include_phrases:
                if phrase in pair[0] or phrase in pair[1]:
                    filtered_pairs.append(pair)
        pairs = filtered_pairs

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    print("Dumping")
    pickle.dump(input_lang, open("cache/{}{}.p".format(input_lang.name, suffix), 'wb'))
    pickle.dump(output_lang, open("cache/{}{}.p".format(output_lang.name, suffix), 'wb'))
    pickle.dump(pairs, open("cache/{}2{}{}.p".format(input_lang.name, output_lang.name, suffix), 'wb'))

    return input_lang, output_lang, pairs

def readCache(file):
    return pickle.load(open("cache/{}.p".format(file), 'rb'))
