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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

package_directory = os.path.dirname(os.path.abspath("__file__"))


# CONSTANTS
UNK = u"<?>"
START = u"</s>"
END = u"</>"
MAX_LENGTH = 70
TRAIN_TEST_SPLIT = 0.9
VALIDATION_TEST_SPLIT = 0.9
BATCH_SIZE = 32

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
        self.padding_tensor = torch.zeros_like(self.start_tensor)
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
        dprint(tweet)
        seq_length = len(tweet)
        # pad to max length
        vectors  = [self.word2vec[word] for word in tweet] + [np.zeros(400)] * (MAX_LENGTH - seq_length)
        return torch.tensor(vectors, dtype=torch.float, device=device), seq_length


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

    def unembed(self, decoder_outputs, length=MAX_LENGTH):
        # print('decoder_outputs {} = {}'.format(decoder_outputs.size(), decoder_outputs))
        decoder_outputs = [[v.numpy()] for v in decoder_outputs.detach().cpu().view(-1, 400)[:length,:]]
        return ' '.join(self.word2vec.most_similar(v, topn=1)[0][0] for v in decoder_outputs)

dataset = ThinkNook(from_cache=True, word2vec=w2v_model, name='ThinkNook')

class Encoder(nn.Module):
    def __init__(self, embedder, device, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = 256
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size=400, hidden_size=self.hidden_size, bidirectional=bidirectional)

        self.device = device

    def forward(self, input, hidden):
        dprint('hidden {} = {}'.format(hidden.size(), hidden))
        # takes input of shape (seq_len, batch, input_size)
        output, hidden = self.gru(input, hidden)
        # output shape of seq_len, batch, num_directions * hidden_size
        dprint('hidden {} = {}'.format(hidden.size(), hidden))
        # dprint('output {} = {}'.format(output.size(), output))
        return output, hidden

    def init_hidden(self):
        s = 2 if self.bidirectional else 1
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(s, BATCH_SIZE, self.hidden_size, device=self.device)

class Decoder(nn.Module):
    def __init__(self, embedder, device, bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_size = 256
        self.bidirectional = bidirectional
        self.embedder = embedder

        s = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size=400, hidden_size=self.hidden_size, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.hidden_size * s, 401)
        self.activation = nn.Tanh()
        self.stop_activation = nn.Sigmoid()

        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, teacher_forcing_p, original_input):
        hidden = encoder_hidden
        input = self.embedder.start_tensor.repeat(1, BATCH_SIZE, 1)

        outputs = []
        stops = []
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)
        padded_input, _ = pad_packed_sequence(original_input)
        dprint('padded_input {} = {}'.format(padded_input.size(), padded_input))
        max_length = len(encoder_outputs)
        for i in range(max_length):
            dprint('input {} = {}'.format(input.size(), input))
            dprint('hidden {} = {}'.format(hidden.size(), hidden))
            output, hidden, stop = self.step(input, hidden)
            dprint('output {} = {}'.format(output.size(), output))
            dprint('stop {} = {}'.format(stop.size(), stop))

            use_teacher_forcing = random.random() < teacher_forcing_p
            dprint('forced' if use_teacher_forcing else 'unforced')
            input = padded_input[i,:,:] if use_teacher_forcing else output
            input = input.view(1,-1,400)

            outputs.append(output)
            stops.append(stop)

            # Pre-empt only when ALL stop neurons are high
            if (stop > 0.5).all():
                break

        outputs = torch.stack(outputs)
        stops = torch.stack(stops)
        return outputs, stops

    def step(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)

        # Separate stop neuron from rest of the output
        stop = self.stop_activation(output[0,:,400])
        output = self.activation(output[0,:,:400])
        return output, hidden, stop

    def init_hidden(self):
        s = 2 if self.bidirectional else 1
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(s, BATCH_SIZE, self.hidden_size, device=self.device)

class Seq2Seq:
    def __init__(self, dataset, train_loader, validation_loader, test_loader, device):
        self.dataset = dataset
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.bidirectional = False

        self.encoder = Encoder(dataset, device=device,  bidirectional=self.bidirectional).to(device)
        self.decoder = Decoder(dataset, device=device,  bidirectional=self.bidirectional).to(device)
        self.criterion =  nn.MSELoss()
        self.stop_criterion = nn.MSELoss()
        self.device = device

        print(self.encoder)
        print(self.decoder)

    def print_step(self, packed_input, lengths, decoder_outputs, avg_loss, epoch, i):
        # unpack for decoding
        input_tensor, _ = pad_packed_sequence(packed_input)

        dprint('input_tensor {} = {}'.format(input_tensor.size(), input_tensor))
        dprint('lengths {} = {}'.format(lengths.size(), lengths))

        # print out a medium sized tweet
        mid = int(BATCH_SIZE / 2)
        input_to_print =  input_tensor[:lengths[mid], mid, :]
        output_to_print = decoder_outputs[:lengths[mid], mid, :]

        dprint('input_to_print {} = {}'.format(input_to_print.size(), input_to_print))
        dprint('input_to_print {} = {}'.format(output_to_print.size(), output_to_print))

        input_text = self.dataset.unembed(input_to_print)
        output_text = self.dataset.unembed(output_to_print)

        print('{0:d} {1:d} {2:.10f}'.format(epoch, i * BATCH_SIZE, avg_loss))

        print(input_text)
        print(output_text)
        print(' ', flush=True)

        if VISDOM:
            vis.text('{} => {}'.format(input_text, output_text))

    def get_loss(self, input_tensor, lengths, decoder_outputs, stops):

        dprint('decoder_outputs {} = {}'.format(decoder_outputs.size(), decoder_outputs))

        # unpack input sequence so it can be easily processed
        input_tensor, _ = pad_packed_sequence(input_tensor)
        dprint('input_tensor {} = {}'.format(input_tensor.size(), input_tensor))

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = input_tensor[:decoder_outputs.size(0), :, :]
        dprint('cropped_input {} = {}'.format(cropped_input.size(), cropped_input))

        # mask out decoder outputs in positions that don't have a corresponding original input
        # we don't want to define a loss on these outputs
        mask = torch.tensor([[[1] * 400 if torch.max(j) > 0 else [0] * 400 for j in i] for i in cropped_input],
                            dtype=torch.float, device=self.device)
        # mask = (cropped_input > self.dataset.padding_tensor).float()
        dprint('mask {} = {}'.format(mask.size(), mask))
        dprint('decoder_outputs {} = {}'.format(decoder_outputs.size(), decoder_outputs))
        decoder_outputs = decoder_outputs * mask
        dprint('decoder_outputs {} = {}'.format(decoder_outputs.size(), decoder_outputs))

        loss = self.criterion(decoder_outputs, cropped_input)

        # stop neuron should be zero everywhere other than the final position of each input sequence
        ideal_stops = torch.zeros_like(stops)
        stop_size = ideal_stops.size()
        dprint(stop_size)
        # need to check bounds in case of pre-empted decoding
        for (i, j) in enumerate(lengths):
            if (j - 1 <= stop_size[0] - 1):
                ideal_stops[j - 1, i] = 1

        dprint('stops {} = {}'.format(stops.size(), stops))
        dprint('ideal_stops {} = {}'.format(ideal_stops.size(), ideal_stops))
        loss += self.stop_criterion(stops, ideal_stops)

        return loss


    def train_step(self, input_tensor, lengths, optimizer, teacher_forcing_p):
        encoder_hidden = self.encoder.init_hidden()

        optimizer.zero_grad()

        encoder_outputs, encoder_hidden = self.encoder.forward(input_tensor, encoder_hidden)
        decoder_outputs, stops = self.decoder.forward(encoder_outputs, encoder_hidden, teacher_forcing_p, input_tensor)

        loss = self.get_loss(input_tensor, lengths, decoder_outputs, stops)

        loss.backward()
        optimizer.step()

        return loss.item(), decoder_outputs


    def train(self, epochs=1, print_every=500, validate_every=50000, learning_rate=0.0001,
              initial_teacher_forcing_p=0.8, final_teacher_forcing_p=0.1, teacher_force_decay=0.0000003):

        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate)

        print_loss_total = 0  # Reset every print_every
        for epoch in range(epochs):
            for (i, (packed_input, lengths)) in enumerate(train_loader):

                teacher_forcing_p = max(final_teacher_forcing_p, initial_teacher_forcing_p - teacher_force_decay * i)

                loss, decoder_outputs = self.train_step(packed_input, lengths, optimizer, teacher_forcing_p)
                print_loss_total += loss

                dprint((i + 1) * BATCH_SIZE)
                if i > 0 and i % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    self.print_step(packed_input, lengths, decoder_outputs, print_loss_avg, epoch, i)
                    print_loss_total = 0

                if i > 0 and i % validate_every == 0:
                    self.validate(self.dataset, validation_loader, print_every)


            self.validate(dataset, validation_loader)

    def validation_step(self, input_tensor, lengths, criterion):
        encoder_hidden = self.encoder.init_hidden()

        encoder_outputs, encoder_hidden = self.encoder.forward(input_tensor, encoder_hidden)
        decoder_outputs, stops = self.decoder.forward(encoder_outputs, encoder_hidden)

        loss = self.get_loss(input_tensor, decoder_outputs, stops)

        return loss.item(), decoder_outputs


    def validate(self, data, criterion, print_every):
        print("VALIDATING")

        total_validation_loss = 0

        with torch.no_grad():

            for (i, [input_tensor, lengths]) in enumerate(validation_loader):
                loss, decoder_outputs = self.validation_step(input_tensor, lengths, criterion)
                total_validation_loss += loss

                if i > 0 and i % print_every == 0:
                    print_loss = total_validation_loss / i
                    self.print_step(input_tensor, lengths, decoder_outputs, print_loss, 0, i)
                    j = 0

        print("AVERAGE VALIDATION LOSS: {}".format(total_validation_loss / len(validation_loader)))

def get_dataloaders(dataset):

    def collate(samples):
        # Sort batch by sequence length and pack
        inputs, lengths = zip(*samples)

        input_tensor = torch.stack(list(inputs))
        lengths = torch.tensor(lengths, device=device)

        lengths, perm_index = lengths.sort(0, descending=True)
        input_tensor = input_tensor[perm_index]
        input_tensor = input_tensor.permute(1, 0, 2).contiguous()

        dprint('input_tensor {} = {}'.format(input_tensor.size(), input_tensor))
        dprint('lengths {} = {}'.format(lengths.size(), lengths))

        packed_input = pack_padded_sequence(input_tensor, list(lengths.data))
        return packed_input, lengths

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    train_split_at = int(len(dataset) * TRAIN_TEST_SPLIT)
    train_indices, rest = indices[:train_split_at], indices[train_split_at:]
    validation_split_at = int(len(rest) * VALIDATION_TEST_SPLIT)
    validation_indices, test_indices = rest[:validation_split_at], rest[validation_split_at:]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate)
    return train_loader, validation_loader, test_loader

train_loader, validation_loader, test_loader = get_dataloaders(dataset)

model = Seq2Seq(dataset, train_loader, validation_loader, test_loader, device)

model.train(epochs=5, print_every=int(500/BATCH_SIZE))









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
