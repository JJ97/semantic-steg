import  torch
from torch.utils.data.dataset import Dataset
import twitter
import numpy as np

# DATASET
class TwitterLive():
    def __init__(self, batch_size):
        print("LOADING")
        self.api = twitter.Api(consumer_key='AK4Wrw2ooowWL9rsZG1hrXTkB',
                          consumer_secret='P3lAWIG17oBVgkXIfjmlqKPJCv20xU5pxx0oxx25wFRwy38pY3',
                          access_token_key='1013926326-19omFH0yaqDTAmwS5aY2jQPf8Z6tojnr3OHvkI3',
                          access_token_secret='0dAsZWcYXotjTQBobubt4NcK4gm3BHG21pSEeR0aWIKCR').GetStreamSample()
        self.size = np.iinfo(np.int64).max
        self.batch_size = batch_size
        print("LOADED {} SAMPLES".format(self.size))

    def __len__(self):
        return np.iinfo(np.int64).max

    def __getitem__(self, item):
        tweets = set()
        while len(tweets) < self.batch_size:
            s = next(self.api)
            if 'delete' in s or 'lang' not in s or 'text' not in s or 'truncated' not in s and 'retweeted_status' not in s:
                continue
            if s['lang'] == 'en':
                if 'extended_tweet' in s:
                    t = s['extended_tweet']['full_text']
                elif 'retweeted_status' in s:
                    t = s['retweeted_status']['text']
                else:
                    t = s['text']
                if len(t) <= 140 and '…' not in t:
                    tweets.add(t)

        tweets = list(sorted(tweets, key=len, reverse=True))
        lengths = torch.tensor([len(t) for t in tweets])
        return tweets, lengths

    # def collate(self, samples):
    #     # Sort batch by sequence length and pack
    #     inputs, lengths = zip(*samples)
    #
    #     # input_tensor = torch.stack(list(inputs))
    #     # lengths = torch.tensor(lengths, dtype=torch.uint8, requires_grad=False)
    #     #
    #     # lengths, perm_index = lengths.sort(0, descending=True)
    #     #
    #     # input_tensor = input_tensor[perm_index]
    #     # input_tensor = input_tensor.permute(1, 0).contiguous()
    #
    #     return inputs, lengths
    #
