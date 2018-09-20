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
                 train_test_split=0.975, validation_test_split=0.8):
        print("LOADING")
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

    @abstractmethod
    def parse_samples(self, samples):
        pass

    @abstractmethod
    def parse_tweet(self, tweet):
        pass

    @abstractmethod
    def get_printable_sample(self, sample):
        pass

