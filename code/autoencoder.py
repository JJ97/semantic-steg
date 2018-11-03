import html
import os.path
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import visdom
from torch import optim, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from word2vecReader import Word2Vec

package_directory = os.path.dirname(os.path.abspath("__file__"))

# CONSTANTS
UNK = u"<?>"
START = u"</s>"
END = u"</>"
MAX_LENGTH = 70
TRAIN_TEST_SPLIT = 0.975
VALIDATION_TEST_SPLIT = 0.8
BATCH_SIZE = 128

DEBUG = False
VISDOM = False
RESUME_FROM = ''
EVALUATE_FROM = ''


if VISDOM:
    vis = visdom.Visdom(server='localhost', port=8274)
    if vis.win_exists('training_loss'):
        vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='training_loss')
    if vis.win_exists('validation_loss'):
        vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='validation_loss')

pd.options.display.max_colwidth = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (torch.cuda.is_available()):
    print('USING: {}'.format(torch.cuda.get_device_name(0)))


def dprint(s):
    if DEBUG:
        print(s)


def save_checkpoint(encoder, decoder, optimizer, epoch, batch, best_validation_loss, filename='checkpoint.p.tar'):
    state = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'epoch': epoch,
        'batch': batch,
        'best_validation_loss': best_validation_loss,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)


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
        self.start_tensor = torch.tensor(np.append(self.word2vec[START], [0]), dtype=torch.float, device=device)
        self.end_tensor = torch.tensor(np.append(self.word2vec[END], [1]), dtype=torch.float, device=device)
        self.unk_tensor = torch.tensor(np.append(self.word2vec[UNK], [1]), dtype=torch.float, device=device)
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
        tweet = [START] + tweet
        dprint(tweet + [END])
        vectors = [np.append(self.word2vec[word], [0]) for word in tweet]
        vectors += [np.append(self.word2vec[END], [1])]

        seq_length = len(tweet) + 1
        vectors += [np.zeros(401)] * (MAX_LENGTH - seq_length)
        return torch.tensor(vectors, dtype=torch.float, device=device), seq_length

    def load(self):
        path = os.path.join('../datasets/Sentiment Analysis Dataset.csv')
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
        return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)',
                      r'_URL_', tweet)

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
        samples = samples[samples.apply(lambda x: 1 <= len(x) and len(x) <= MAX_LENGTH - 2)]
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
        decoder_outputs = [[v.numpy()] for v in decoder_outputs.detach().cpu().view(-1, 401)[:length, :400]]
        return ' '.join(self.word2vec.most_similar(v, topn=1)[0][0] for v in decoder_outputs)


dataset = ThinkNook(from_cache=True, word2vec=w2v_model, name='ThinkNook')


def clip_grad(v, min, max):
    if min and max and v.requires_grad:
        v.register_hook(lambda g: g.clamp(min, max))
    return v


class Encoder(nn.Module):
    def __init__(self, embedder, device, bidirectional=False, clip=None):
        super(Encoder, self).__init__()
        self.hidden_size = 512
        self.bidirectional = bidirectional
        self.layers = 4
        self.clip = clip
        self.gru = nn.GRU(input_size=401, hidden_size=self.hidden_size, bidirectional=bidirectional
                          , num_layers=self.layers, dropout=0.4)

        self.device = device

    def forward(self, input, hidden):
        dprint('hidden {} = {}'.format(hidden.size(), hidden))
        # takes input of shape (seq_len, batch, input_size)
        output, hidden = self.gru(input, hidden)
        hidden = clip_grad(hidden, -self.clip, self.clip)
        # output shape of seq_len, batch, num_directions * hidden_size
        dprint('hidden {} = {}'.format(hidden.size(), hidden))
        # dprint('output {} = {}'.format(output.size(), output))
        return output, hidden

    def init_hidden(self):
        s = 2 if self.bidirectional else 1
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(s * self.layers, BATCH_SIZE, self.hidden_size, device=self.device)


class Decoder(nn.Module):
    def __init__(self, embedder, device, bidirectional=False, clip=None):
        super(Decoder, self).__init__()
        self.hidden_size = 512
        self.bidirectional = bidirectional
        self.embedder = embedder
        self.layers = 4
        self.clip = clip

        s = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size=401, hidden_size=self.hidden_size, bidirectional=bidirectional
                          , num_layers=self.layers)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.hidden_size * s, 401)

        self.activation = nn.Tanh()
        self.stop_activation = nn.Sigmoid()

        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, teacher_forcing_p, original_input, word_dropout_rate):
        hidden = encoder_hidden
        input = self.embedder.start_tensor.repeat(1, BATCH_SIZE, 1)
        unk = self.embedder.unk_tensor.repeat(1, BATCH_SIZE, 1)

        outputs = [input.view(BATCH_SIZE, 401)]

        dprint('encoder_outputs {} = {}'.format(encoder_outputs.size(), encoder_outputs))
        dprint('original_input {} = {}'.format(original_input.size(), original_input))
        max_length = len(encoder_outputs)
        for i in range(max_length - 1):
            use_teacher_forcing = random.random() < teacher_forcing_p
            dprint('forced' if use_teacher_forcing else 'unforced')
            input = original_input[i, :, :] if use_teacher_forcing else input

            use_word_drouput = random.random() < word_dropout_rate
            input = unk if use_word_drouput else input

            # just to be sure
            input = input.view(1, -1, 401)

            dprint('input {} = {}'.format(input.size(), input))
            dprint('hidden {} = {}'.format(hidden.size(), hidden))
            output, hidden = self.step(input, hidden)
            output = output.view(BATCH_SIZE, 401)
            dprint('output {} = {}'.format(output.size(), output))

            use_teacher_forcing = random.random() < teacher_forcing_p
            dprint('forced' if use_teacher_forcing else 'unforced')
            input = output

            outputs.append(output)

            # Pre-empt only when ALL stop neurons are high
            if (output[:, 400] > 0.5).all():
                break

        outputs = torch.stack(outputs)
        return outputs

    def step(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = clip_grad(output, -self.clip, self.clip)
        output[:, :, :400] = self.activation(output[:, :, :400])
        output[:, :, 400] = self.stop_activation(output[:, :, 400])
        output = clip_grad(output, -self.clip, self.clip)
        return output, hidden

    def init_hidden(self):
        s = 2 if self.bidirectional else 1
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(s * self.layers, BATCH_SIZE, self.hidden_size, device=self.device)


class Seq2Seq:
    def __init__(self, dataset, train_loader, validation_loader, test_loader, device):
        self.dataset = dataset
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.reverse_input = False
        self.bidirectional = True
        self.clip = 5

        self.encoder = Encoder(dataset, device=device, bidirectional=self.bidirectional, clip=5).to(device)
        self.decoder = Decoder(dataset, device=device, bidirectional=self.bidirectional, clip=5).to(device)
        self.criterion = nn.MSELoss(reduction='sum')
        self.device = device

        print(self.encoder)
        print(self.decoder)

    def print_step(self, input_tensor, lengths, decoder_outputs, l1, epoch, i):

        dprint('input_tensor {} = {}'.format(input_tensor.size(), input_tensor))
        dprint('lengths {} = {}'.format(lengths.size(), lengths))

        # print out a medium sized tweet
        mid = int(BATCH_SIZE / 2)
        input_to_print = input_tensor[:lengths[mid], mid, :]
        output_to_print = decoder_outputs[:lengths[mid], mid, :]

        dprint('input_to_print {} = {}'.format(input_to_print.size(), input_to_print))
        dprint('input_to_print {} = {}'.format(output_to_print.size(), output_to_print))

        input_text = self.dataset.unembed(input_to_print)
        output_text = self.dataset.unembed(output_to_print)

        print('{0:d} {1:d} l1: {2:.10f}'.format(epoch, i * BATCH_SIZE, l1))

        print(input_text)
        print(output_text)
        print(' ', flush=True)

    def get_loss(self, input_tensor, lengths, decoder_outputs):

        dprint('decoder_outputs {} = {}'.format(decoder_outputs.size(), decoder_outputs))
        dprint('input_tensor {} = {}'.format(input_tensor.size(), input_tensor))

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = input_tensor[:decoder_outputs.size(0), :, :]
        dprint('cropped_input {} = {}'.format(cropped_input.size(), cropped_input))

        # mask out decoder outputs in positions that don't have a corresponding original input
        # we don't want to define a loss on these outputs
        mask = torch.tensor([[[int(torch.max(j.abs()) > 0)] * 401 for j in i] for i in cropped_input],
                            dtype=torch.float, device=self.device)
        # mask = (cropped_input > self.dataset.padding_tensor).float()
        dprint('mask {} = {}'.format(mask.size(), mask))
        dprint('decoder_outputs {} = {}'.format(decoder_outputs.size(), decoder_outputs))
        decoder_outputs = decoder_outputs * mask
        dprint('decoder_outputs {} = {}'.format(decoder_outputs.size(), decoder_outputs))

        unmasked_count = torch.nonzero(mask).size(0)
        dprint('unmasked_count {}'.format(unmasked_count))
        l1 = self.criterion(decoder_outputs, cropped_input) / unmasked_count
        dprint('l1 {}'.format(l1))

        return l1

    def train_step(self, input_tensor, lengths, optimizer, teacher_forcing_p, word_dropout_rate):
        encoder_hidden = self.encoder.init_hidden()

        optimizer.zero_grad()

        if self.reverse_input:
            input_tensor = input_tensor.flip(0)

        packed_input = pack_padded_sequence(input_tensor, list(lengths.data))

        encoder_outputs, encoder_hidden = self.encoder.forward(packed_input, encoder_hidden)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)
        decoder_outputs = self.decoder.forward(encoder_outputs, encoder_hidden, teacher_forcing_p, input_tensor,
                                               word_dropout_rate)

        l1 = self.get_loss(input_tensor, lengths, decoder_outputs)

        loss = l1
        loss.backward()

        optimizer.step()

        return l1.item(), decoder_outputs

    def train(self, optimizer, epochs=1, print_every=500, validate_every=50000,
              initial_teacher_forcing_p=0.8, final_teacher_forcing_p=0.1, teacher_force_decay=0.0000003,
              word_dropout_rate=0.5, best_validation_loss=np.inf, epoch=0, batch=0):

        print('USING: {}'.format(self.device))

        validations_since_best = 0
        for epoch in range(epoch, epochs):
            print_l1_total = 0  # Reset every print_every
            for i in range(batch, len(train_loader)):
                input_tensor, lengths = train_loader[i]

                if input_tensor.size(1) != BATCH_SIZE:
                    break

                samples_processed = (epoch * BATCH_SIZE * len(train_loader)) + ((i + 1) * BATCH_SIZE)

                teacher_forcing_p = max(final_teacher_forcing_p,
                                        initial_teacher_forcing_p - teacher_force_decay * samples_processed)

                l1, decoder_outputs = self.train_step(input_tensor, lengths, optimizer, teacher_forcing_p,
                                                      word_dropout_rate)
                print_l1_total += l1

                dprint(samples_processed)
                if i > 0 and i % print_every == 0:
                    print_l1_avg = print_l1_total / print_every
                    self.print_step(input_tensor, lengths, decoder_outputs, print_l1_avg, epoch, i)
                    print_l1_total, print_l2_total = 0, 0

                    # noinspection PyArgumentList
                    vis.line(X=np.array([int(samples_processed)]),
                             Y=np.array([[float(print_l1_avg)]]),
                             win='training_loss',
                             opts=dict(title="train", xlabel='samples processed', ylabel='loss', legend=['l1']),
                             update='append')

                if i > 0 and i % validate_every == 0 or i == len(train_loader) - 1:
                    val = self.validate(validation_loader, print_every)

                    # noinspection PyArgumentList
                    vis.line(X=np.array([int(samples_processed)]),
                             Y=np.array([[float(val / len(validation_loader))]]),
                             win='validation_loss',
                             opts=dict(title="val", xlabel='samples processed', ylabel='loss', legend=['l1']),
                             update='append')

                    if val < best_validation_loss:
                        best_validation_loss = val
                        validations_since_best = 0
                        save_checkpoint(self.encoder, self.decoder, optimizer, epoch, i, best_validation_loss)
                    else:
                        validations_since_best += 1

                    print("{} SINCE LAST BEST VALIDATION".format(validations_since_best))

                    if validations_since_best >= 3:
                        return

            batch = 0

    def validation_step(self, input_tensor, lengths):
        encoder_hidden = self.encoder.init_hidden()

        if self.reverse_input:
            input_tensor = input_tensor.flip(0)

        packed_input = pack_padded_sequence(input_tensor, list(lengths.data))

        encoder_outputs, encoder_hidden = self.encoder.forward(packed_input, encoder_hidden)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)
        decoder_outputs = self.decoder.forward(encoder_outputs, encoder_hidden, 0, input_tensor, 0)

        l1 = self.get_loss(input_tensor, lengths, decoder_outputs)

        return l1.item(), decoder_outputs

    def validate(self, validation_loader, print_every):
        print("VALIDATING")

        total_l1_loss = 0
        with torch.no_grad():
            for (i, [input_tensor, lengths]) in enumerate(validation_loader):
                if input_tensor.size(1) != BATCH_SIZE:
                    break

                l1, decoder_outputs = self.validation_step(input_tensor, lengths)
                total_l1_loss += l1

                if i > 0 and i % print_every == 0:
                    print_l1_loss = total_l1_loss / i
                    self.print_step(input_tensor, lengths, decoder_outputs, print_l1_loss, 0, i)

        print("AVERAGE VALIDATION LOSS: {}".format((total_l1_loss) / len(validation_loader)))
        return total_l1_loss


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

    return input_tensor, lengths


def get_dataloaders(dataset):
    indices = list(range(len(dataset)))

    rng = np.random.RandomState(0xDA7A5E7)
    rng.shuffle(indices)

    train_split_at = int(len(dataset) * TRAIN_TEST_SPLIT)
    train_indices, rest = indices[:train_split_at], indices[train_split_at:]
    validation_split_at = int(len(rest) * VALIDATION_TEST_SPLIT)
    validation_indices, test_indices = rest[:validation_split_at], rest[validation_split_at:]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                               collate_fn=collate)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler,
                                                    collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate)
    print('{} TRAIN  {} VALIDATE  {} TEST'.format(len(train_loader), len(validation_loader), len(test_loader)))
    return train_loader, validation_loader, test_loader


train_loader, validation_loader, test_loader = get_dataloaders(dataset)

model = Seq2Seq(dataset, train_loader, validation_loader, test_loader, device)

optimizer = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.0001)

if RESUME_FROM:
    x = torch.load(RESUME_FROM, map_location='cuda')
    model.encoder.load_state_dict(x['encoder_state_dict'])
    model.decoder.load_state_dict(x['decoder_state_dict'])
    optimizer.load_state_dict(x['optimizer'])
    model.encoder.train()
    model.decoder.train()

    epoch = x['epoch']
    batch = x['batch']
    val = x['best_validation_loss']

    model.train(optimizer, epochs=50000, print_every=int(10000 / BATCH_SIZE), validate_every=int(200000 / BATCH_SIZE),
                epoch=epoch, batch=batch, best_validation_loss=val)

elif EVALUATE_FROM:
    x = torch.load(EVALUATE_FROM, map_location='cpu')
    model.encoder.load_state_dict(x['encoder_state_dict'])
    model.decoder.load_state_dict(x['decoder_state_dict'])
    optimizer.load_state_dict(x['optimizer'])
    model.encoder.eval()
    model.decoder.eval()

else:
    model.train(optimizer, epochs=50000, print_every=int(10000 / BATCH_SIZE), validate_every=int(200000 / BATCH_SIZE))


def interpolate_between(i, j, steps=10):
    i, i_len = collate([dataset[i]] * BATCH_SIZE)
    j, j_len = collate([dataset[j]] * BATCH_SIZE)
    print(i_len, j_len)
    print(dataset.unembed(i, i_len.data[0]))
    print(dataset.unembed(j, j_len.data[0]))

    i_h, j_h = model.encoder.init_hidden(), model.encoder.init_hidden()

    packed_i = pack_padded_sequence(i, list(i_len))
    packed_j = pack_padded_sequence(j, list(j_len))

    i_eo, i_h = model.encoder.forward(packed_i, i_h)
    j_eo, j_h = model.encoder.forward(packed_j, j_h)

    step_size = 1 / steps
    hiddens = [step_size * step * i_h + (1 - step_size * step) * j_h for step in range(steps + 1)]

    i_eo, _ = pad_packed_sequence(i_eo) if i_len > j_len else pad_packed_sequence(j_eo)

    for h in hiddens:
        hidden2tweet(i_eo, h)
        # hidden2tweet(i_eo, i_h)


def interpolate_around(i, scale=0.1, steps=10):
    i, i_len = collate([dataset[i]] * BATCH_SIZE)
    print(i_len)
    print(dataset.unembed(i, i_len.data[0]))

    i_h = model.encoder.init_hidden()

    packed_i = pack_padded_sequence(i, list(i_len))

    i_eo, i_h = model.encoder.forward(packed_i, i_h)

    i_eo, _ = pad_packed_sequence(i_eo)

    hidden2tweet(torch.zeros(70, 1, 400), i_h)

    for s in range(steps):
        h = i_h + torch.tensor(np.random.normal(loc=0, scale=scale, size=np.shape(i_h)), dtype=torch.float)
        hidden2tweet(torch.zeros(70, 1, 400), h)


def hidden2tweet(eo, eh):
    outputs = model.decoder.forward(eo, eh, 0, eo, 0)
    print(dataset.unembed(outputs))


def hidden_hists():
    qs = []
    for (i, [input_tensor, lengths]) in enumerate(validation_loader):
        encoder_hidden = model.encoder.init_hidden()
        packed_input = pack_padded_sequence(input_tensor, list(lengths.data))
        _, encoder_hidden = model.encoder.forward(packed_input, encoder_hidden)

        first_unit = encoder_hidden[4, 0, 5]
        qs.append(float(first_unit))

        if i > 0 and i % 500 == 0:
            torch.tensor(qs).view(-1)
            vis.histogram(qs)
            break
