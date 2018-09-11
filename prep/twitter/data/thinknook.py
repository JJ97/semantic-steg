import os.path
import pickle

import pandas as pd

from data.dataset import DataSet

package_directory = os.path.dirname(os.path.abspath(__file__))


class ThinkNook(DataSet):

    def __init__(self, from_cache=True):
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

