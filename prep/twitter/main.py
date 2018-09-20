import pandas as pd
import torch

from data.thinknook import ThinkNook
from data.thinknook_char import ThinkNookChar
from embedding.onehot import OneHot
from embedding.onehot_char import OneHotChar
from model.seq2seq.seq2seq import Seq2Seq
from constants import *

def main():
    pd.options.display.max_colwidth = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if (torch.cuda.is_available()):
        print('USING: {}'.format(torch.cuda.get_device_name(0)))

    dataset = ThinkNookChar(from_cache=True)
    embedder = OneHotChar(dataset.chars, device)
    model = Seq2Seq(embedder, device)

    model.train(dataset, iterations=5, print_every=1)


if __name__ == '__main__':
    # Note: You must put all your training code into one function rather than in the global scope
    #       (this is good practice anyway).
    #       Subsequently you must call the set_start_method and your main function from inside this
    #       if-statement. If you don't do that, each worker will attempt to run all of your training
    #       code and everything will go very wild and very wrong.
    torch.multiprocessing.set_start_method('forkserver')
    main()
