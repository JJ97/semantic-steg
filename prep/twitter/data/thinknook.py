import os.path
import pickle
import html
import re
import swifter

import pandas as pd

from data.dataset import DataSet
from constants import *
from collections import defaultdict
from functools import partial

package_directory = os.path.dirname(os.path.abspath(__file__))


class ThinkNook(DataSet):

    def __init__(self, from_cache=True):
        self.wordlist = set()
        super().__init__(from_cache, keep_threshold=5, name='ThinkNook')

    def load(self):
        path =  os.path.join(package_directory, '../../../datasets/Sentiment Analysis Dataset.csv')
        df = pd.read_csv(path, sep=',', usecols=['SentimentText'], dtype={'SentimentText': str})
        samples = df.SentimentText
        samples = self.parse_samples(samples)
        return samples

    def load_from_cache(self):
        path = os.path.join(package_directory, 'cache/sentiment.p')
        if os.path.exists(path):
            samples = pickle.load(open(path, "rb"))
            self.generate_wordlist(samples)
        else:
            print('Cache not found. Loading from source')
            samples = self.load()
            pickle.dump(samples, open(path, "wb"))
        return samples


    def filter_out_bad_chars(self, tweet):
        return re.sub(r'[^@# 0-9a-z]', r' ', tweet)

    def replace_mentions_and_hashtags(self, tweet):
        tweet = re.sub(r'(#|@)\S*\b', r'\1 ', tweet)
        # remove stuff like w@ter
        return re.sub(r'\S+[#@]', r' ', tweet)

    def normalize_spaces(self, tweet):
        return re.sub(r' +', r' ', tweet)

    re.sub(r'http\S* ', r'https ', t)

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
        samples = samples.str.split("")
        self.generate_wordlist(samples)
        print("...replacing uncommon words")
        samples = samples.swifter.apply(partial(self.replace_uncommon_words))
        print("...filtering out particularly long tweets")
        samples = samples[samples.apply(lambda x : 1 <= len(x) and len(x) <= MAX_LENGTH - 2)]
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

    def get_printable_sample(self, sample):
       return ' '.join(sample)
