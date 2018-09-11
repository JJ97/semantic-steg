import pickle
import re
import html
import swifter

from abc import ABC, abstractmethod
from constants import *
from collections import defaultdict
from functools import partial


class DataSet(ABC):

    def __init__(self, from_cache, keep_threshold, name,
                 train_test_split=0.8, validation_test_split=0.8):
        print("LOADING")
        self.wordlist = set()
        self.keep_threshold = keep_threshold
        self.samples = self.load_from_cache() if from_cache else self.load()
        self.size = self.samples.shape[0]
        self.name = name
        self.split(train_test_split, validation_test_split)
        print("LOADED {} SAMPLES FROM {}".format(self.size, self.name))

    def split(self, train_test_split, validation_test_split):
        # shuffle first
        self.samples = self.samples.sample(self.size)
        train_split_at = int(len(self.samples) * train_test_split)
        self.train_set = self.samples[:train_split_at]
        rest = self.samples[train_split_at:]
        validation_split_at = int(len(rest) * validation_test_split)
        self.validation_set = rest[:validation_split_at]
        self.test_set = rest[validation_split_at:]

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def load_from_cache(self):
        pass

    def filter_out_bad_chars(self, tweet):
        return re.sub(r'[^@# 0-9a-z]', r' ', tweet)

    def replace_mentions_and_hashtags(self, tweet):
        tweet = re.sub(r'(#|@)\S*\b', r'\1 ', tweet)
        # remove stuff like w@ter
        return re.sub(r'\S+[#@]', r' ', tweet)

    def normalize_spaces(self, tweet):
        return re.sub(r' +', r' ', tweet)

    def generate_wordlist(self, samples):
        print("...generating wordlist")
        word_counts = defaultdict(int)
        for s in samples:
            for w in s:
                word_counts[w] += 1
        self.wordlist = {w for (w, count) in word_counts.items() if count > self.keep_threshold}
        self.wordlist.update({UNK, START, END})

    def replace_uncommon_words(self, tweet):
        return [w if w in self.wordlist else UNK for w in tweet]

    def parse_samples(self, samples):
        print("SANITISING")
        print("...converting to lowercase")
        samples = samples.str.lower()
        print("...unescaping html")
        samples = samples.apply(html.unescape)
        print("...filtering out bad characters")
        samples = samples.swifter.apply(self.filter_out_bad_chars)
        print("...replacing mentions and hashtags")
        samples = samples.swifter.apply(self.replace_mentions_and_hashtags)
        print("...normalizing spaces")
        samples = samples.swifter.apply(self.normalize_spaces)
        print("...stripping")
        samples = samples.str.strip()
        print("...splitting")
        samples = samples.str.split(" ")
        self.generate_wordlist(samples)
        print("...replacing uncommon words")
        samples = samples.swifter.apply(partial(self.replace_uncommon_words))
        return samples


    def parse_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = html.unescape(tweet)
        tweet = self.filter_out_bad_chars(tweet)
        tweet = self.replace_mentions_and_hashtags(tweet)
        tweet = self.normalize_spaces(tweet)
        tweet = tweet.strip()
        tweet = tweet.split(" ")
        tweet = self.replace_uncommon_words(tweet)
        return tweet