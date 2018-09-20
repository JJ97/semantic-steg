import os.path
import pickle
import html
import re
import swifter
import string
import codecs

import pandas as pd

from data.dataset import DataSet
from constants import *
from collections import defaultdict
from functools import partial

package_directory = os.path.dirname(os.path.abspath(__file__))


class ThinkNookChar(DataSet):

    def __init__(self, from_cache=True):
        self.chars = set()
        super().__init__(from_cache, keep_threshold=5, name='ThinkNookChar')

    def load(self):
        path =  os.path.join(package_directory, '../../../datasets/Sentiment Analysis Dataset.csv')
        df = pd.read_csv(path, sep=',', usecols=['SentimentText'], dtype={'SentimentText': str}, encoding='utf8')
        samples = df.SentimentText
        samples = self.parse_samples(samples)
        return samples

    def load_from_cache(self):
        path = os.path.join(package_directory, 'cache/ThinkNookChar.p')
        if os.path.exists(path):
            samples = pickle.load(open(path, "rb"))
            self.get_chars(samples)
        else:
            print('Cache not found. Loading from source')
            samples = self.load()
            pickle.dump(samples, open(path, "wb"))
        return samples

    def filter_out_bad_chars(self, tweet):
        return ''.join(c for c in tweet if c in string.printable)

    def get_chars(self, samples):
        print("...getting unique characters")
        char_counts = defaultdict(int)
        for s in samples:
            for c in s:
                char_counts[c] += 1
        self.chars = {c for (c, count) in char_counts.items() if count > self.keep_threshold}
        self.chars.update({UNK, START, END})

    def replace_uncommon_chars(self, tweet):
        return ''.join(c if c in self.chars else UNK for c in tweet)

    def parse_samples(self, samples):
        print("SANITISING")
        print("...unescaping html")
        samples = samples.apply(html.unescape)
        print("...filtering out bad characters")
        samples = samples.swifter.apply(self.filter_out_bad_chars)
        self.get_chars(samples)
        print("...replacing uncommon characters")
        samples = samples.swifter.apply(self.replace_uncommon_chars)
        print("...filtering out particularly long tweets")
        samples = samples[samples.apply(lambda x : 1 <= len(x) and len(x) <= MAX_LENGTH - 2)]
        return samples


    def parse_tweet(self, tweet):
        tweet = html.unescape(tweet)
        tweet = self.filter_out_bad_chars(tweet)
        tweet = self.replace_uncommon_chars(tweet)
        return tweet

    def get_printable_sample(self, sample):
        return sample

