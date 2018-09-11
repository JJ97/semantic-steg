import pandas as pd
import torch

from data.thinknook import ThinkNook
from embedding.onehot import OneHot
from model.seq2seq.seq2seq import Seq2Seq

pd.options.display.max_colwidth = 150

if __name__ == '__main__':
    # Note: You must put all your training code into one function rather than in the global scope
    #       (this is good practice anyway).
    #       Subsequently you must call the set_start_method and your main function from inside this
    #       if-statement. If you don't do that, each worker will attempt to run all of your training
    #       code and everything will go very wild and very wrong.
    torch.multiprocessing.set_start_method('forkserver')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ThinkNook(from_cache=True)
    embedder = OneHot(dataset.wordlist, device)
    model = Seq2Seq(embedder, device)

    model.train(dataset, iterations=5)