import os.path
import pickle
import html
import re
import swifter
import torch
import random
import visdom

import pandas as pd
import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from functools import partial
from torch import optim, nn
from word2vecReader import Word2Vec
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

package_directory = os.path.dirname(os.path.abspath("__file__"))


# CONSTANTS
UNK = u"<?>"
START = u"</s>"
END = u"</>"
MAX_LENGTH = 70
TRAIN_TEST_SPLIT = 0.8
VALIDATION_TEST_SPLIT = 0.8

DEBUG = False
VISDOM = False

# torch.multiprocessing.set_start_method('forkserver')
if VISDOM:
    vis = visdom.Visdom(server='http://ncc.clients.dur.ac.uk', port=8274)

pd.options.display.max_colwidth = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (torch.cuda.is_available()):
    print('USING: {}'.format(torch.cuda.get_device_name(0)))



def dprint(s):
    if DEBUG:
        print(s)

w2v_model_path = "./word2vec_twitter_model.bin"
print("Loading the model, this can take some time...")
w2v_model = Word2Vec.load_word2vec_format(w2v_model_path, binary=True)

# DATASET
class ThinkNook(Dataset):

    def __init__(self, from_cache, word2vec, name):
        print("LOADING")
        self.word2vec = word2vec
        self.samples = self.load_from_cache() if from_cache else self.load()
        self.size = self.samples.shape[0]
        self.samples.sample(self.size)
        self.start_tensor = torch.tensor(self.word2vec[START], dtype=torch.float, device=device)
        self.end_tensor = torch.tensor(self.word2vec[END], dtype=torch.float, device=device)
        print("LOADED {} SAMPLES".format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        try:
            tweet = self.samples[item]
        except KeyError as k:
            print(k)
            tweet = []

        tweet = [START] + tweet + [END]
        return torch.tensor([self.word2vec[word] for word in tweet], dtype=torch.float, device=device)


    def load(self):
        path =  os.path.join('../datasets/Sentiment Analysis Dataset.csv')
        df = pd.read_csv(path, sep=',', usecols=['SentimentText'], dtype={'SentimentText': str})
        samples = df.SentimentText
        samples = self.parse_samples(samples)
        return samples

    def load_from_cache(self):
        path = os.path.join('../cache/sentiment.p')
        if os.path.exists(path):
            samples = pickle.load(open(path, "rb"))
        else:
            print('Cache not found. Loading from source')
            samples = self.load()
            pickle.dump(samples, open(path, "wb"))
        return samples

    def normalize_spaces(self, tweet):
        return re.sub(r' +', r' ', tweet)

    def replace_mentions(self, tweet):
        return re.sub(r'(^|\W)@\S*\b', r' _MENTION_ ', tweet)

    def replace_numbers(self, tweet):
        return re.sub(r'[0-9]+', r'_NUMBER_', tweet)

    def replace_urls(self, tweet):
        return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', r'_URL_', tweet)

    def split_punctuation(self, tweet):
        return re.sub(r'([!"$%&*+,\-./:;<=>?[\\\]^`{|}~()]+)', r' \1 ', tweet)

    def replace_unseen_words(self, tweet):
        return [word if word in self.word2vec else UNK for word in tweet]

    def parse_samples(self, samples):
        print("SANITISING")
        print("...unescaping html")
        samples = samples.apply(html.unescape)
        print("...replacing mentions")
        samples = samples.swifter.apply(self.replace_mentions)
        print("...replacing urls")
        samples = samples.swifter.apply(self.replace_urls)
        print("...replacing numbers")
        samples = samples.swifter.apply(self.replace_numbers)
        print("...splitting on punctuation")
        samples = samples.swifter.apply(self.split_punctuation)
        print("...normalizing spaces")
        samples = samples.swifter.apply(self.normalize_spaces)
        print("...stripping")
        samples = samples.str.strip()
        print("...splitting")
        samples = samples.str.split(" ")
        print("...filtering out particularly long tweets")
        samples = samples[samples.apply(lambda x : 1 <= len(x) and len(x) <= MAX_LENGTH - 2)]
        print("...replacing unseen words")
        samples = samples.swifter.apply(self.replace_unseen_words)
        return samples

    def parse_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = html.unescape(tweet)
        tweet = self.replace_mentions(tweet)
        tweet = self.replace_urls(tweet)
        tweet = self.replace_numbers(tweet)
        tweet = self.split_punctuation(tweet)
        tweet = self.normalize_spaces(tweet)
        tweet = tweet.strip()
        tweet = tweet.split(" ")
        tweet = self.replace_unseen_words(tweet)
        return tweet

    def get_printable_sample(self, sample):
       return ' '.join(sample)

    def unembed(self, decoder_outputs):
        decoder_outputs = [[v.numpy()] for v in decoder_outputs.detach().cpu().view(-1, 400)]
        return ' '.join(self.word2vec.most_similar(v, topn=1)[0][0] for v in decoder_outputs)


dataset = ThinkNook(from_cache=True, word2vec=w2v_model, name='ThinkNook')

def get_dataloaders(dataset):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    train_split_at = int(len(dataset) * TRAIN_TEST_SPLIT)
    train_indices, rest = indices[:train_split_at], indices[train_split_at:]
    validation_split_at = int(len(rest) * VALIDATION_TEST_SPLIT)
    validation_indices, test_indices = rest[:validation_split_at], rest[validation_split_at:]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=test_sampler)

    return train_loader, validation_loader, test_loader

train_loader, validation_loader, test_loader = get_dataloaders(dataset)

class Encoder(nn.Module):
    def __init__(self, embedder, hidden_size, device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=400, hidden_size=hidden_size)

        self.device = device

    def forward(self, input, hidden):
        input = input.view(-1, 1, 400)
        dprint('input {} = {}'.format(input.size(), input))
        dprint('hidden {} = {}'.format(hidden.size(), hidden))
        # takes input of shape (seq_len, batch, input_size)
        output, hidden = self.gru(input, hidden)
        # output shape of seq_len, batch, num_directions * hidden_size
        dprint('hidden {} = {}'.format(hidden.size(), hidden))
        dprint('output {} = {}'.format(output.size(), output))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Decoder(nn.Module):
    def __init__(self, embedder, hidden_size, device, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedder = embedder
        self.dropout_p = dropout_p

        self.gru = nn.GRU(input_size=400, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, 401)
        self.activation = nn.Tanh()
        self.stop_activation = nn.Sigmoid()

        self.device = device

    def forward(self, encoder_outputs, encoder_hidden):
        hidden = encoder_hidden
        input = self.embedder.start_tensor.view(1, 1, -1)

        outputs = []
        stops = []
        max_length = len(encoder_outputs)
        for i in range(max_length):
            dprint('input {} = {}'.format(input.size(), input))
            dprint('hidden {} = {}'.format(hidden.size(), hidden))
            output, hidden, stop = self.step(input, hidden)
            dprint('output {} = {}'.format(output.size(), output))
            outputs.append(output)
            stops.append(stop)
            if (stop > 0.5):
                break
        return outputs, stops

    def step(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)

        stop = self.stop_activation(output[0][0][400:])
        output = self.activation(output[0][0][:400])
        return output, hidden, stop

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Seq2Seq:
    def __init__(self, dataset, train_loader, validation_loader, test_loader, device):
        self.dataset = dataset
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.encoder = Encoder(dataset, hidden_size=256, device=device).to(device)
        self.decoder = Decoder(dataset, hidden_size=256, device=device, dropout_p=0.1).to(device)
        self.criterion = nn.MSELoss()
        self.stop_criterion = nn.NLLLoss()
        self.device = device

        print(self.encoder)
        print(self.decoder)


    def train_step(self, input_tensor, optimizer, teacher_forcing_ratio):
        encoder_hidden = self.encoder.init_hidden()

        optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self.encoder.forward(input_tensor, encoder_hidden)
        decoder_outputs, stops = self.decoder.forward(encoder_outputs, encoder_hidden)

        loss = sum(self.criterion(output, input) for output, input in zip(decoder_outputs, input_tensor[0]))

        ideal_stops = [[0] for i in range(len(input_tensor[0]) - 1)] + [[1]]
        ideal_stops = torch.tensor(ideal_stops, dtype=torch.float, device=device)
        loss += sum(self.stop_criterion(output, input) for output, input in zip(stops, ideal_stops))
        loss.backward()
        optimizer.step()

        avg_loss = float(loss.item()) / input_tensor.size(0)

        return avg_loss, decoder_outputs


    def train(self, iterations, print_every=500, validate_every=50000,
              learning_rate=0.0001, teacher_forcing_ratio=0.5):

        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                               lr=learning_rate)

        print_loss_total = 0  # Reset every print_every
        for iteration in range(iterations):

            for (i, input_tensor) in enumerate(train_loader):

                loss, decoder_output = self.train_step(input_tensor, optimizer, teacher_forcing_ratio)
                print_loss_total += loss

                if i > 0 and i % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0

                    input_text = self.dataset.unembed(input_tensor)
                    decoder_output_text = self.dataset.unembed(torch.stack(decoder_output))

                    print('{0:d} {1:d} {2:.10f}'.format(iteration, i, print_loss_avg))

                    print(input_text)
                    print(decoder_output_text)
                    print(' ', flush=True)

                    if VISDOM:
                        vis.text('{} => {}'.format(input_text, decoder_output_text))


                if i > 0 and i % validate_every == 0:
                    self.validate(self.dataset, validation_loader, print_every)

            self.validate(dataset, validation_loader)

    def validation_step(self, input_tensor, criterion):
        encoder_hidden = self.encoder.init_hidden()

        encoder_outputs, encoder_hidden = self.encoder.forward(input_tensor, encoder_hidden)
        decoder_outputs, stops = self.decoder.forward(encoder_outputs, encoder_hidden)

        loss = sum(self.criterion(output, input) for output, input in zip(decoder_outputs, input_tensor[0]))

        ideal_stops = [[0] for i in range(len(input_tensor[0]) - 1)] + [[1]]
        ideal_stops = torch.tensor(ideal_stops, dtype=torch.float, device=device)
        loss += sum(self.stop_criterion(output, input) for output, input in zip(stops, ideal_stops))

        avg_loss = float(loss.item()) / input_tensor.size(0)

        return decoder_outputs, avg_loss


    def validate(self, data, criterion, print_every):
        print("VALIDATING")

        total_validation_loss = 0

        with torch.no_grad():
            for (i, input_tensor) in enumerate(validation_loader):
                decoder_output, loss = self.validation_step(input_tensor, criterion)
                total_validation_loss += loss

                if i > 0 and i % print_every == 0:
                    print('>',self.dataset.unembed(input_tensor))
                    print('<', self.dataset.unembed(torch.stack(decoder_output)))
                    print('loss ', loss)
                    print(' ', flush=True)

        print("AVERAGE VALIDATION LOSS: {}".format(total_validation_loss / len(validation_loader)))




model = Seq2Seq(dataset, train_loader, validation_loader, test_loader, device)

model.train(iterations=5, print_every=1000)









# def get_embedding(text):
#
#     text = dataset.parse_tweet(text)
#
#     input_tensor = model.embedder.embed(text)
#
#     encoder_hidden = model.encoder.init_hidden()
#
#     encoder_outputs = torch.zeros(MAX_LENGTH, model.encoder.hidden_size, device=model.device)
#
#     for ei in range(input_tensor.size(0)):
#         encoder_output, encoder_hidden = model.encoder.forward(input_tensor[ei], encoder_hidden)
#         encoder_outputs[ei] = encoder_output[0, 0]
#
#
#     return encoder_hidden, encoder_outputs
#
# def interpolate(c, d, steps=50):
#     ch, co = get_embedding(c)
#     dh, do = get_embedding(d)
#     hiddens, outputs = interpolate_between(ch, co, dh, do, steps)
#     for h, o in zip(hiddens, outputs):
#         print(decode_interped(h, o))
#
#
# def interpolate_between(ch, co, dh, do, steps=10):
#     step_size = 1/steps
#     hiddens = [step_size * step * ch + (1 - step_size * step) * dh for step in range(steps + 1) ]
#
#     outputs = [step_size * step * co + (1 - step_size * step) * do for step in range(steps + 1) ]
#
#     # outputs = []
#     # for step in range(steps + 1):
#     #     outputs.append([step_size * step * o1 + (1 - step_size * step) * o2 for o1, o2 in zip(co, do) ])
#
#     return hiddens, outputs
#
#
# def decode_interped(encoder_hidden, encoder_outputs):
#     decoder_input = model.embedder.start_tensor
#     decoder_hidden = encoder_hidden
#
#     criterion = nn.NLLLoss()
#
#     decoder_outputs = []
#
#     # Without teacher forcing: use its own predictions as the next input
#     for di in range(30):
#         decoder_output, decoder_hidden, decoder_attention = model.decoder.forward(
#             decoder_input, decoder_hidden, encoder_outputs)
#         topv, topi = decoder_output.topk(1)
#         decoder_input = topi.squeeze().detach()  # detach from history as input
#
#         decoder_outputs.append(decoder_output)
#         if decoder_input.item() == model.embedder.end_tensor:
#             break
#
#     return model.output2tweet(decoder_outputs)
#
# hiddens, outputs = interpolate(ch, co, dh, do)
#
#
# for h,o in zip(hiddens,outputs):
#     print(decode_interped(h, o))
#
#
# def interpolate_around(c, steps=50):
#     ch, co = get_embedding(c)
#
#     for s in range(steps):
#         ich = ch + torch.tensor(np.random.normal(loc=0, scale=1, size=np.shape(ch)), dtype=torch.float).cuda()
#         ico = co + torch.tensor(np.random.normal(loc=0, scale=1, size=np.shape(co)), dtype=torch.float).cuda()
#         print(decode_interped(ich, ico))
#
# interpolate_around('hey guys whats up')
