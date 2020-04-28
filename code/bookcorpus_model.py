
import os.path
import random

import numpy as np
import torch
import math
import hashlib
import nltk

import torch.nn.functional as F
import visdom
import plotly.graph_objs as ago

from torch import optim, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.normal import Normal
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, RandomSampler
from bookcorpus import BookCorpus

from gensen import GenSen, GenSenSingle

package_directory = os.path.dirname(os.path.abspath("__file__"))

# CONSTANTS
UNK = u"<unk>"
START = u"<s>"
END = u"</s>"
MAX_LENGTH = 70
TRAIN_TEST_SPLIT = 0.99
VALIDATION_TEST_SPLIT = 0.8
BATCH_SIZE = 2
EMBEDDING_SIZE = 512

NAME = 'x'

DEBUG = False
VISDOM = True
SAVE_CHECKPOINT = True
RESUME_FROM = ''
EVALUATE_FROM = 'bc_newgru_amsgrad_noise_006_clip1_x'

device = "cuda" if torch.cuda.is_available() else "cpu"


def std_mean_plot(iters, means, stds, plottitle=None, xlabel='x', ylabel='y', curvenames=None):
    # plotly's default colours
    colors = ((31, 119, 180, 0.5),  # muted blue
              (255, 127, 14, 0.5),  # safety orange
              (44, 160, 44, 0.5),  # cooked asparagus green
              (214, 39, 40, 0.5),  # brick red
              (148, 103, 189, 0.5),  # muted purple
              (140, 86, 75, 0.5),  # chestnut brown
              (227, 119, 194, 0.5),  # raspberry yogurt pink
              (127, 127, 127, 0.5),  # middle gray
              (188, 189, 34, 0.5),  # curry yellow-green
              (23, 190, 207, 0.5))  # blue-teal
    plotdata = []
    means = np.array(means)
    stds = np.array(stds)
    lowdata = means-stds
    highdata = means+stds
    # for i in range(len(means)):
    trace = ago.Scatter(
        x=iters,
        y=lowdata,
        fill='none',
        mode='lines',
        opacity=0.0,
        line={'color': 'rgba{}'.format(colors[0%10][:-1] + (0.0,))},
        showlegend=False)
    plotdata.append(trace)
    trace = ago.Scatter(
        x=iters,
        y=highdata,
        mode='lines',
        fill='tonexty',
        opacity=0.0,
        fillcolor='rgba{}'.format(colors[0%10]),
        line={'color': 'rgba{}'.format(colors[0 % 10][:-1] + (0.0,))},
        showlegend=False)
    plotdata.append(trace)
    trace = ago.Scatter(
        x=iters,
        y=means,
        fill=None,
        mode='lines',
        line={'color' : 'rgb{}'.format(colors[0%10][:-1])},
        name=plottitle)
    plotdata.append(trace)
    plot_layout = ago.Layout(title=plottitle,
                            xaxis={'title':xlabel},
                            yaxis={'title':ylabel})
    return ago.Figure(data=plotdata, layout=plot_layout)




def dprint(s):
    if DEBUG:
        print(s)


def save_checkpoint(decoder, optimizer_gen,
                    samples_processed, best_validation_loss, filename=NAME):

    if SAVE_CHECKPOINT:
        state = {
            'samples_processed': samples_processed,
            'best_validation_loss': best_validation_loss,
        }
        torch.save(state, '{}.p.tar'.format(filename))
        torch.save(decoder.state_dict(), '{}_decoder.p.tar'.format(filename))
        torch.save(optimizer_gen.state_dict(), '{}_optimizer_gen.p.tar'.format(filename))


def clip_grad(v, min, max):
    if min and max and v.requires_grad:
        v.register_hook(lambda g: g.clamp(min, max))
    return v

def freeze(m):
    for param in m.parameters():
        param.requires_grad = False

def unfreeze(m):
    for param in m.parameters():
        param.requires_grad = True


class ConditionalGRU(nn.Module):
    """A Gated Recurrent Unit (GRU) cell with peepholes."""

    def __init__(self, input_dim, hidden_dim, dropout=0.):
        """Initialize params."""
        super(ConditionalGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_weights = nn.Linear(self.input_dim, 3 * self.hidden_dim)
        self.hidden_weights = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.peep_weights = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)

        self.reset = nn.Sigmoid()
        self.input = nn.Sigmoid()
        self.new = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        """Set params."""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden, ctx):
        r"""Propogate input through the layer.

        inputs:
        input   - batch size x target sequence length  x embedding dimension
        hidden  - batch size x hidden dimension
        ctx     - batch size x source sequence length  x hidden dimension
        returns: output, hidden
        output  - batch size x target sequence length  x hidden dimension
        hidden  - (batch size x hidden dimension, \
            batch size x hidden dimension)
        """
        def recurrence(input, hidden, ctx):
            """Recurrence helper."""
            input_gate = self.input_weights(input)
            hidden_gate = self.hidden_weights(hidden)
            peep_gate = self.peep_weights(ctx)
            i_r, i_i, i_n = input_gate.chunk(3, 1)
            h_r, h_i, h_n = hidden_gate.chunk(3, 1)
            p_r, p_i, p_n = peep_gate.chunk(3, 1)
            resetgate = self.reset(i_r + h_r + p_r)
            inputgate = self.input(i_i + h_i + p_i)
            newgate = self.new(i_n + resetgate * h_n + p_n)
            hy = newgate + inputgate * (hidden - newgate)

            return hy

        input = input.transpose(0, 1)
        ctx = ctx.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, ctx[i])
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        return output, hidden



class NewGRU(nn.Module):
    """A Gated Recurrent Unit (GRU) cell with peepholes."""

    def __init__(self, input_dim, hidden_dim, dropout=0.):
        """Initialize params."""
        super(NewGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_weights_r = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_weights_r = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.peep_weights_r = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.input_weights_i = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_weights_i = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.peep_weights_i = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.input_weights_n = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_weights_n = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.peep_weights_n = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.reset = nn.Sigmoid()
        self.input = nn.Sigmoid()
        self.new = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        """Set params."""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden, ctx):
        r"""Propogate input through the layer.

        inputs:
        input   - batch size x target sequence length  x embedding dimension
        hidden  - batch size x hidden dimension
        ctx     - batch size x source sequence length  x hidden dimension
        returns: output, hidden
        output  - batch size x target sequence length  x hidden dimension
        hidden  - (batch size x hidden dimension, \
            batch size x hidden dimension)
        """
        def recurrence(input, hidden, ctx):
            """Recurrence helper."""
            i_r = self.input_weights_r(input)
            i_i = self.input_weights_i(input)
            i_n = self.input_weights_n(input)

            h_r = self.hidden_weights_r(hidden)
            h_i = self.hidden_weights_i(hidden)

            p_r = self.peep_weights_r(ctx)
            p_i = self.peep_weights_i(ctx)
            p_n = self.peep_weights_n(ctx)



            resetgate = self.reset(i_r + h_r + p_r)
            inputgate = self.input(i_i + h_i + p_i)
            newgate = self.new(i_n + self.hidden_weights_n(resetgate * hidden) + p_n)
            hy = (1 - inputgate) * hidden + inputgate * newgate
            return hy

        input = input.transpose(0, 1)
        ctx = ctx.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, ctx[i])
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        return output, hidden



class Decoder(nn.Module):
    def __init__(self, hidden_size, latent_size, vocab_size, layers, device, clip=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.layers = 1
        self.clip = clip

        self.stop_threshold = torch.tensor([0.5], dtype=torch.float, device=device)

        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.layers)
        self.gru = NewGRU(input_dim=EMBEDDING_SIZE, hidden_dim=hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.hidden_size, EMBEDDING_SIZE )
        self.unembed = nn.Linear(EMBEDDING_SIZE, vocab_size)

        self.activation = nn.Tanh()
        self.stop_activation = nn.Sigmoid()

        self.device = device

    def forward(self, encoder_outputs, latent, word_dropout_rate):

        # hidden = self.latent2hidden(latent)

        hidden = latent.view(BATCH_SIZE, self.hidden_size)

        original_hidden = torch.unsqueeze(latent, 1).clone()

        input = START_TENSOR.repeat(BATCH_SIZE, 1, 1).to(device)
        unk = UNK_TENSOR.repeat(BATCH_SIZE, 1, 1).to(device)

        outputs = [self.unembed(input.view(BATCH_SIZE, EMBEDDING_SIZE))]

        # stops = [torch.zeros((BATCH_SIZE, 1)).to(device)]

        max_length = len(encoder_outputs[0])
        for i in range(max_length - 1):

            # use_word_drouput = random.random() < word_dropout_rate
            # input = unk if use_word_drouput else input

            # just to be sure
            input = input.view(BATCH_SIZE, 1, EMBEDDING_SIZE)

            output, hidden = self.step(input, hidden, original_hidden)

            input = output
            output = self.unembed(output)

            outputs.append(output)

            # stops.append(stop)
            # Pre-empt only when ALL stop neurons are high
            # if (stop > self.stop_threshold[0]).all():
            #     break

        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0, 1)

        return outputs
        #
        # stops = torch.stack(stops)
        # stops = torch.transpose(stops, 0, 1)
        # return outputs, stops

    def step(self, input, hidden, original_hidden):
        output, hidden = self.gru(input, hidden, original_hidden)
        # output = self.dropout(output)
        output = self.out(output)
        if output.requires_grad:
            output.register_hook(lambda x: x.clamp(min=-1, max=1))
        if hidden.requires_grad:
            hidden.register_hook(lambda x: x.clamp(min=-1, max=1))
        # output = clip_grad(output, -self.clip, self.clip)
        # output = self.activation(output)
        # output[:, :, :EMBEDDING_SIZE] = self.activation(output[:, :, :EMBEDDING_SIZE])
        # output[:, :, EMBEDDING_SIZE] = self.stop_activation(output[:, :, EMBEDDING_SIZE])
        # output = clip_grad(output, -self.clip, self.clip)

        # stops = output[:, :, -1].view(BATCH_SIZE, 1)
        output = output.view(BATCH_SIZE, EMBEDDING_SIZE)

        # return output, stops, hidden
        return output, hidden


class Seq2SeqGAN:
    def __init__(self, train_loader, validation_loader, test_loader, device):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.clip = 5

        self.latent_size = 2048
        self.decoder_hidden_size = 2048

        self.decoder_layers = 2

        self.noise = Normal(torch.tensor([0.0], requires_grad=False), torch.tensor([0.6], requires_grad=False))

        self.encoder = GenSenSingle(
            model_folder='./data/models',
            filename_prefix='nli_large_bothskip',
            pretrained_emb='./data/embedding/glove.840B.300d.h5',
            cuda=torch.cuda.is_available()
        )
        vocab_size = len(self.encoder.encoder.src_embedding.weight)
        self.encoder.encoder.to(device)
        self.decoder = Decoder(self.decoder_hidden_size, self.latent_size, vocab_size,
                               self.decoder_layers, device=device
                               , clip=5).to(device)

        weight_mask = torch.ones(vocab_size).to(device)
        weight_mask[self.encoder.word2id['<pad>']] = 0
        self.criterion = nn.CrossEntropyLoss(weight=weight_mask)
        self.criterion.to(device)
        self.bce = nn.BCELoss()
        self.device = device

        self.embedding_norms = torch.norm(self.encoder.encoder.src_embedding.weight, 1)

        print(self.decoder)

    def print_step(self, input_tensor, lengths, decoder_outputs, losses, epoch, i):

        # print out a medium sized tweet
        mid = int(BATCH_SIZE / 2)

        input_to_print = input_tensor[mid, : lengths[mid]].view(-1)
        output_to_print = decoder_outputs[mid, :lengths[mid], :]

        input_text = ' '.join([self.encoder.id2word[int(i)] for i in input_to_print])
        output_text = self.unembed(output_to_print)

        print('{0:d} {1:d} l1: {2:.10f}'.format(epoch, i * BATCH_SIZE, losses))

        print(input_text)
        print(output_text)
        print(' ', flush=True)

    def get_loss(self, cropped_input, lengths, decoder_outputs):

        l1 = self.criterion(decoder_outputs.contiguous().view(-1, decoder_outputs.size(2)),
                            cropped_input.contiguous().view(-1))

        #
        # ideal_stops = torch.zeros_like(stops)
        # for i, l in enumerate(lengths):
        #     if l <= ideal_stops.size(1):
        #         ideal_stops[i, l-1:] = 1
        # l2 = self.bce(stops, ideal_stops)

        # return l1, l2
        return l1

    def train_step(self, input_tensor, lengths, optimizer_gen, word_dropout_rate):
        optimizer_gen.zero_grad()

        encoder_outputs, encoder_hidden, embedded_input, lengths = self.encoder.get_representation_and_embedded_input(
            input_tensor, pool='last', return_numpy=False, tokenize=True
        )

        encoder_outputs = encoder_outputs.detach()
        encoder_hidden = encoder_hidden.detach()
        embedded_input = embedded_input.detach()
        lengths = lengths.to(device)
        lengths = lengths.detach()

        #
        # noise = self.noise.sample(encoder_hidden.size()).view_as(encoder_hidden).to(self.device)
        # encoder_hidden += noise

        # decoder_outputs, stops = self.decoder.forward(encoder_outputs, encoder_hidden, word_dropout_rate)
        decoder_outputs = self.decoder.forward(encoder_outputs, encoder_hidden, word_dropout_rate)

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = embedded_input[:, :decoder_outputs.size(1)]

        # l1, l2 = self.get_loss(cropped_input, lengths, decoder_outputs, stops)
        l1 = self.get_loss(cropped_input, lengths, decoder_outputs)

        loss_gen = l1
        loss_gen.backward()

        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
        optimizer_gen.step()

        losses = np.array([l1.item()])
        return losses, decoder_outputs.data, embedded_input.data, lengths.data

    def validation_step(self, input_tensor, lengths):
        encoder_outputs, encoder_hidden, embedded_input, lengths = self.encoder.get_representation_and_embedded_input(
            input_tensor, pool='last', return_numpy=False, tokenize=False
        )

        encoder_outputs = encoder_outputs.detach()
        encoder_hidden = encoder_hidden.detach()
        embedded_input = embedded_input.detach()
        lengths = lengths.to(device)
        lengths = lengths.detach()

        decoder_outputs = self.decoder.forward(encoder_outputs, encoder_hidden, 0)

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = embedded_input[:, :decoder_outputs.size(1)]
        l1 = self.get_loss(cropped_input, lengths, decoder_outputs)

        losses = np.array([l1.item()])
        return losses, decoder_outputs.data, embedded_input.data, lengths.data

    def train(self, optimizer_gen, epochs=1, print_every=500, validate_every=50000,
              word_dropout_rate=0.0, best_validation_loss=np.inf, start_at=0):

        print('USING: {}'.format(self.device))

        validations_since_best = 0

        means = [[],[]]
        stds = [[],[]]
        ce_vals = [[],[]]
        iters = []
        for epoch in range(epochs):


            for i, (input_tensor, lengths) in enumerate(train_loader):

                lengths = lengths.to(device)

                if len(input_tensor) != BATCH_SIZE:
                    break

                samples_processed = (epoch * BATCH_SIZE * len(train_loader)) + ((i + 1) * BATCH_SIZE) + start_at

                losses, decoder_outputs, embedded_input, lengths = self.train_step(input_tensor, lengths,
                                                                        optimizer_gen, word_dropout_rate)

                ce_vals[0].append(losses[0])

                if i > 0 and i % print_every == 0:
                    means[0].append(np.mean(np.array(ce_vals[0])))
                    stds[0].append(np.std(np.array(ce_vals[0])))
                    ce_vals[0] = []
                    iters.append(samples_processed)

                    self.print_step(embedded_input, lengths, decoder_outputs,
                                    means[0][-1],  epoch, i)

                    # for y, l in zip(print_total,
                    #                 ['reconstruction', 'stops']):
                    l = 'reconstruction'
                    mean_plot = std_mean_plot(iters, means[0], stds[0], plottitle=l, xlabel='samples processed', ylabel='MSE',
                                              curvenames=None)
                    vis.plotlyplot(mean_plot, win=l)

                if i > 0 and i % validate_every == 0:
                    val_l1s = self.validate(validation_loader, print_every, samples_processed)

                    val_l1s = np.array(val_l1s)
                    means[1].append(val_l1s.mean())
                    stds[1].append(val_l1s.std())
                    mean_plot = std_mean_plot(iters, means[1], stds[1], plottitle='validation loss',
                                              xlabel='samples processed', ylabel='MSE',
                                              curvenames=None)
                    vis.plotlyplot(mean_plot, win='validation loss')

                    if means[1][-1] < best_validation_loss:
                        best_validation_loss = means[1][-1]
                        validations_since_best = 0

                        save_checkpoint(self.decoder,
                                        optimizer_gen, samples_processed,
                                        best_validation_loss)

                    else:
                        validations_since_best += 1

                    print("{} SINCE LAST BEST VALIDATION".format(validations_since_best))

                    if validations_since_best >= 100:
                        return

                del input_tensor
                del decoder_outputs
                del embedded_input


    def validate(self, validation_loader, print_every, samples_processed):
        print("VALIDATING")

        print_total = np.array([0.0] )

        l1s = []

        with torch.no_grad():

            for (i, [input_tensor, lengths]) in enumerate(validation_loader):
                if len(input_tensor) != BATCH_SIZE:
                    break

                lengths = lengths.to(device)

                losses, decoder_outputs, embedded_input, lengths = self.validation_step(input_tensor, lengths)

                print_total += losses
                l1s.append(losses[0])

                if i > 0 and i % print_every == 0:

                    self.print_step(embedded_input, lengths, decoder_outputs,
                                    print_total[0] / i, 0, i)

                del input_tensor
                del decoder_outputs
                del embedded_input


        print("AVERAGE VALIDATION LOSS: {}".format(float(print_total[0]) / len(validation_loader)))
        return l1s

    # def unembed(self, decoder_outputs, length=MAX_LENGTH):
    #
    #     indices = [int(torch.argmax(
    #         torch.mm(self.encoder.encoder.src_embedding.weight,
    #                                      torch.unsqueeze(d, 1)[:EMBEDDING_SIZE])
    #         / self.embedding_norms
    #     )) for d in decoder_outputs]
    #     return ' '.join([self.encoder.id2word[i] for i in indices])

    def unembed(self, decoder_outputs, length=MAX_LENGTH):
        indices = [int(torch.argmax(d)) for d in decoder_outputs]
        # indices = [int(torch.argmin(
        #     torch.norm(self.encoder.encoder.src_embedding.weight - d[:EMBEDDING_SIZE], dim=1)
        # )) for d in decoder_outputs]
        return ' '.join([self.encoder.id2word[i] for i in indices])

    def beam_search_decoder(self, data, k=5):
        sm = torch.nn.LogSoftmax(dim=1)
        data = sm(data)
        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        for row in data:
            all_candidates = list()
            row_scores, row_indices = torch.topk(row, k=k)
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for rs, ri in zip(row_scores, row_indices):
                    candidate = [seq + [int(ri)], score * float(rs)]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:k]
        return  [' '.join([self.encoder.id2word[i] for i in seq]) for seq, score in sequences]

class BucketSampler(Sampler):

    def __init__(self, indices, lengths, batch_size, bucket_size):
        super(BucketSampler, self).__init__(indices)
        self.indices = indices
        self.lengths = lengths
        print(len(indices), len(lengths))
        self.batch_size = batch_size
        self.bucket_size = bucket_size

    def __iter__(self):
        buckety_sort = lambda i : self.lengths[i] + random.random() * self.bucket_size

        n = len(self.indices)
        num_chunks = n // self.batch_size

        self.indices = [self.indices[i] for i in torch.randperm(len(self.indices))]

        self.indices = [self.indices[i:i + num_chunks] for i in range(0, n, num_chunks)]

        extra = self.indices[self.batch_size] if len(self.indices) == self.batch_size + 1 else []

        self.indices = self.indices[:self.batch_size]

        self.indices = [list(sorted(c, key=buckety_sort)) for c in self.indices]


        self.indices = [self.indices[chunk][batch]
                        for batch in torch.randperm(len(self.indices[0]))
                        for chunk in range(len(self.indices))
                        if batch < len(self.indices[chunk])] + extra

        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_dataloaders(dataset):
    # indices = list(range(len(dataset)))
    #
    # rng = np.random.RandomState(0xDA7A5E7)
    # rng.shuffle(indices)
    #
    # train_split_at = int(len(dataset) * TRAIN_TEST_SPLIT)
    # train_indices, rest = indices[:train_split_at], indices[train_split_at:]
    # validation_split_at = int(len(rest) * VALIDATION_TEST_SPLIT)
    # validation_indices, test_indices = rest[:validation_split_at], rest[validation_split_at:]

    # train_lengths = {i : dataset[i][1] for i in train_indices}
    # validation_lengths =  {i : dataset[i][1] for i in validation_indices}
    # test_lengths = {i :dataset[i][1] for i in test_indices}

    BUCKET_SIZE = 5
    # train_sampler = BucketSampler(train_indices, train_lengths, BATCH_SIZE, BUCKET_SIZE)
    # validation_sampler = BucketSampler(validation_indices, validation_lengths, BATCH_SIZE, BUCKET_SIZE)
    # test_sampler = BucketSampler(test_indices, test_lengths, BATCH_SIZE, BUCKET_SIZE)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=dataset.collate, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True,
                                                    collate_fn=dataset_val.collate)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True,
                                              collate_fn=dataset_test.collate)
    print('{} TRAIN  {} VALIDATE  {} TEST'.format(len(train_loader), len(validation_loader), len(test_loader)))
    return train_loader, validation_loader, test_loader

def go():
    if RESUME_FROM:
        model.decoder.load_state_dict(torch.load('{}_decoder.p.tar'.format(RESUME_FROM), map_location=device))
        model.decoder.train()

        x = torch.load('{}.p.tar'.format(RESUME_FROM), map_location='cpu')
        samples_processed = x['samples_processed'] if 'samples_processed' in x else 0
        val = x['best_validation_loss']

        optimizer_gen.load_state_dict(torch.load('{}_optimizer_gen.p.tar'.format(RESUME_FROM), map_location=device))

        print("STARTING FROM {} SAMPLES".format(samples_processed))
        model.train(optimizer_gen, epochs=50000,
                    print_every=int(50000 / BATCH_SIZE), validate_every=int(750000 * 6 / BATCH_SIZE),
                    start_at=samples_processed, best_validation_loss=val)
    elif EVALUATE_FROM:
        model.decoder.load_state_dict(torch.load('{}_decoder.p.tar'.format(EVALUATE_FROM), map_location=device))

    else:
        model.train(optimizer_gen, epochs=50000,
                    print_every=int(50000 / BATCH_SIZE), validate_every=int(750000 * 6 / BATCH_SIZE))


def interpolate_between(i, j, steps=10):
    i, i_len = dataset.collate([dataset[i]] * BATCH_SIZE)
    j, j_len = dataset.collate([dataset[j]] * BATCH_SIZE)
    print(i_len, j_len)
    print(model.unembed(i, i_len.data[0]))
    print(model.unembed(j, j_len.data[0]))

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
    with torch.no_grad():
        noise_func = Normal(torch.tensor([0.0], requires_grad=False), torch.tensor([scale], requires_grad=False))
        i, i_len = dataset.collate([dataset[i]] * BATCH_SIZE)
        i_len = i_len.to(device)

        encoder_outputs, encoder_hidden, embedded_input, lengths = model.encoder.get_representation_and_embedded_input(
            i, pool='last', return_numpy=False, tokenize=True
        )

        encoder_outputs = encoder_outputs.detach()
        encoder_hidden = encoder_hidden.detach()
        embedded_input = embedded_input.detach()
        lengths = lengths.to(device)
        lengths = lengths.detach()

        print(i[0])
        outputs, _ = model.decoder.forward(encoder_outputs, encoder_hidden, 0)
        print(model.unembed(outputs[0, :, :]))
        for s in range(steps):
            noise = noise_func.sample(encoder_hidden.size()).view_as(encoder_hidden).to(model.device)
            h = encoder_hidden + noise
            outputs, _ = model.decoder.forward(encoder_outputs, h, 0)
            print(model.unembed(outputs[0, :, :]))

def interpolate_around_own(text, scale=0.1, steps=10, alpha=0.01):
    inp = (text, len(text.split()))
    noise_func = Normal(torch.tensor([0.0], requires_grad=False), torch.tensor([scale], requires_grad=False))
    i, i_len = dataset.collate([inp] * BATCH_SIZE)
    i_len = i_len.to(device)

    encoder_outputs, encoder_hidden, embedded_input, lengths = model.encoder.get_representation_and_embedded_input(
        i, pool='last', return_numpy=False, tokenize=True
    )

    encoder_outputs = encoder_outputs.detach()
    encoder_hidden = encoder_hidden.detach()
    embedded_input = embedded_input.detach()
    lengths = lengths.to(device)
    lengths = lengths.detach()

    print(i[0])

    uniques = set()
    outputs, _ = model.decoder.forward(encoder_outputs, encoder_hidden, 0)
    print(model.unembed(outputs[0, :, :]))
    for s in range(steps):
        noise = noise_func.sample(encoder_hidden.size()).view_as(encoder_hidden).to(model.device)
        h = encoder_hidden.clone() + noise
        print('NEXT')
        for ii in range(10):
            h = torch.autograd.Variable(h, requires_grad=True)
            outputs, stops = model.decoder.forward(encoder_outputs, h, 0)
            cropped_input = embedded_input[:, :outputs.size(1)]

            l1, _ = model.get_loss(cropped_input,lengths, outputs, stops)

            l1.backward()
            # print(h.grad)

            if l1 < 2:
                print(float(l1))
                print(ii, float(l1), model.unembed(outputs[0, :, :]))

                uniques.add(model.unembed(outputs[0, :, :]))
            h = h - alpha * h.grad
            # print(h)
        print('')
    print(uniques)

def check_bleu(output, targ):
    hypothesis = output.split()
    reference = targ.split()
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
    return BLEUscore

def multipath_forward_interp(text, scale=0.1, samples=10, steps=100, alpha=0.2, cutoff=0.1, payload=None, n=4):
    with torch.no_grad():
        inp = (text, len(text.split()))

        original_alpha = alpha

        i, i_len = dataset.collate([inp] * BATCH_SIZE)
        i_len = i_len.to(device)

        encoder_outputs, encoder_hidden, embedded_input, lengths = model.encoder.get_representation_and_embedded_input(
            i, pool='last', return_numpy=False, tokenize=True
        )

        encoder_outputs = encoder_outputs.detach()
        encoder_hidden = encoder_hidden.detach()
        embedded_input = embedded_input.detach()
        lengths = lengths.to(device)
        lengths = lengths.detach()

        # print(i[0])

        uniques = set()
        outputs = model.decoder.forward(encoder_outputs, encoder_hidden, 0)
        # print(model.beam_search_decoder(outputs[0, :, :]))
        for s in range(samples):
            h = encoder_hidden.clone()

            from_inp = dataset[random.randint(0, len(dataset))]

            i_from, i_len_from = dataset.collate([from_inp] * BATCH_SIZE)
            encoder_outputs_from, encoder_hidden_from, embedded_input_from, lengths_from = model.encoder.get_representation_and_embedded_input(
                i_from, pool='last', return_numpy=False, tokenize=True
            )

            encoder_outputs_from = encoder_outputs_from.detach()
            encoder_hidden_from = encoder_hidden_from.detach()
            embedded_input_from = embedded_input_from.detach()
            lengths_from = lengths_from.to(device)
            lengths_from = lengths_from.detach()


            alpha = original_alpha
            for ii in range(steps):
                encoder_hidden_from = torch.autograd.Variable(encoder_hidden_from, requires_grad=True)
                outputs = model.decoder.forward([[0] * 70], encoder_hidden_from, 0)

                out_size = min(int(outputs.size(1)), int(embedded_input.size(1)))
                cropped_input = embedded_input[:, :out_size]
                cropped_output = outputs[:, :out_size]

                l1 = model.get_loss(cropped_input,lengths, cropped_output)

                l1.backward()
                # print(h.grad)

                # print(ii, float(l1), alpha)
                if (l1) < cutoff:
                    out_strings =  model.beam_search_decoder(outputs[0, :, :])
                    out_strings = [s[:s.find('</s>')] for s in out_strings]
                    # print(ii, float(l1), out_strings[0])
                    # alpha = original_alpha
                    for s in out_strings:
                        output = s
                        if len(text.split()) <= 2 or check_bleu(output, text) > 0.1:
                            # print(output.strip())
                            uniques.add(output.strip())
                            if get_payload(output.strip(), n=n) == payload:
                                return output.strip()
                else:
                    alpha = min(1, alpha*1.1)
                encoder_hidden_from = encoder_hidden_from - alpha * encoder_hidden_from.grad
                # print(h)
            print('')
        print(uniques)
        print(len(uniques))
# multipath_forward_interp('hello chris how are you?')


def multipath_backward_interp(text, scale=0.1, samples=10, steps=100, alpha=0.1, cutoff=0.1, payload=None, n=4):
    inp = (text, len(text.split()))

    original_alpha = alpha

    i, i_len = dataset.collate([inp] * BATCH_SIZE)
    i_len = i_len.to(device)

    encoder_outputs, encoder_hidden, embedded_input, lengths = model.encoder.get_representation_and_embedded_input(
        i, pool='last', return_numpy=False, tokenize=True
    )

    encoder_outputs = encoder_outputs.detach()
    encoder_hidden = encoder_hidden.detach()
    embedded_input = embedded_input.detach()
    lengths = lengths.to(device)
    lengths = lengths.detach()

    # print(i[0])

    uniques = set()
    outputs = model.decoder.forward(encoder_outputs, encoder_hidden, 0)
    # print(model.beam_search_decoder(outputs[0, :, :]))

    jj = 0
    l1 = 1
    while l1 > cutoff :
        encoder_hidden = torch.autograd.Variable(encoder_hidden, requires_grad=True)
        outputs = model.decoder.forward([[0] * 70], encoder_hidden, 0)

        out_size = min(int(outputs.size(1)), int(embedded_input.size(1)))
        cropped_input = embedded_input[:, :out_size]
        cropped_output = outputs[:, :out_size]

        l1 = model.get_loss(cropped_input, lengths, cropped_output)

        l1.backward()
        # print(h.grad)

        print(jj, float(l1), alpha)
        out_strings =  model.beam_search_decoder(outputs[0, :, :])
        out_strings = [' '.join(s[:s.find('</s>')].split()[1:]) for s in out_strings]
        print(jj, float(l1), out_strings[0])
        # alpha = original_alpha
        for s in out_strings:
            output = s
            if len(text.split()) <= 2 or check_bleu(output, text) > 0.1:
                # print(output.strip())
                uniques.add(output.strip())
                if get_payload(output.strip(), n=n) == payload:
                    return output.strip()

        encoder_hidden = encoder_hidden - alpha  * encoder_hidden.grad / float(l1)
        jj += 1



    for s in range(samples):
        h = encoder_hidden.clone()

        from_inp = dataset[random.randint(0, len(dataset))]

        i_from, i_len_from = dataset.collate([from_inp] * BATCH_SIZE)
        encoder_outputs_from, encoder_hidden_from, embedded_input_from, lengths_from = model.encoder.get_representation_and_embedded_input(
            i_from, pool='last', return_numpy=False, tokenize=True
        )

        encoder_outputs_from = encoder_outputs_from.detach()
        encoder_hidden_from = encoder_hidden_from.detach()
        embedded_input_from = embedded_input_from.detach()
        lengths_from = lengths_from.to(device)
        lengths_from = lengths_from.detach()

        alpha = original_alpha
        for ii in range(steps):
            h = torch.autograd.Variable(h, requires_grad=True)
            outputs = model.decoder.forward([[0] * 70], h, 0)

            out_size = min(int(outputs.size(1)), int(embedded_input_from.size(1)))
            cropped_input = embedded_input_from[:, :out_size]
            cropped_output = outputs[:, :out_size]

            l1 = model.get_loss(cropped_input, lengths, cropped_output)

            l1.backward()
            # print(h.grad)

            out_size2 = min(int(outputs.size(1)), int(embedded_input.size(1)))
            ci_2 = embedded_input[:, :out_size2]
            co_2 = outputs[:, :out_size2]
            l2 = model.get_loss(ci_2, lengths, co_2)

            print(ii, float(l1), float(l2), alpha)
            if (l2) < cutoff:
                out_strings =  model.beam_search_decoder(outputs[0, :, :])
                out_strings = [' '.join(s[:s.find('</s>')].split()[1:]) for s in out_strings]
                print(ii, float(l1), out_strings[0])
                # alpha = original_alpha
                for s in out_strings:
                    output = s
                    if len(text.split()) <= 2 or check_bleu(output, text) > 0.1:
                        # print(output.strip())
                        uniques.add(output.strip())
                        if get_payload(output.strip(), n=n) == payload:
                            return output.strip()
            else:
                break
            h = h - alpha * h.grad
            # print(h)
        print('')
    print(uniques)
    print(len(uniques))
# multipath_backward_interp('the quick brown fox jumps over the lazy dog')




def wandering_interp(text, p=0.5, samples=10, steps=100, alpha=0.01, cutoff=0.1, payload=None, n=4):
    inp = (text, len(text.split()))

    original_alpha = alpha

    i, i_len = dataset.collate([inp] * BATCH_SIZE)
    i_len = i_len.to(device)

    encoder_outputs, encoder_hidden, embedded_input, lengths = model.encoder.get_representation_and_embedded_input(
        i, pool='last', return_numpy=False, tokenize=True
    )

    encoder_outputs = encoder_outputs.detach()
    encoder_hidden = encoder_hidden.detach()
    embedded_input = embedded_input.detach()
    lengths = lengths.to(device)
    lengths = lengths.detach()

    # print(i[0])

    uniques = set()
    outputs = model.decoder.forward(encoder_outputs, encoder_hidden, 0)
    # print(model.beam_search_decoder(outputs[0, :, :]))

    original_input = embedded_input.clone()

    l1 = 1
    jj = 0
    while l1 > cutoff:
        encoder_hidden = torch.autograd.Variable(encoder_hidden, requires_grad=True)
        outputs = model.decoder.forward([[0] * 70], encoder_hidden, 0)

        out_size = min(int(outputs.size(1)), int(embedded_input.size(1)))
        cropped_input = embedded_input[:, :out_size]
        cropped_output = outputs[:, :out_size]

        l1 = model.get_loss(cropped_input, lengths, cropped_output)

        l1.backward()
        # print(h.grad)

        # print(jj, float(l1), alpha)
        out_strings =  model.beam_search_decoder(outputs[0, :, :])
        out_strings = [s[:s.find('</s>')] for s in out_strings]
        # print(jj, float(l1), out_strings[0])
        # alpha = original_alpha
        for s in out_strings:
            output = s
            if len(text.split()) <= 2 or check_bleu(output, text) > 0.1:
                # print(output.strip())
                uniques.add(output.strip())
                if get_payload(output.strip(), n=n) == payload:
                    return output.strip()
        encoder_hidden = encoder_hidden - alpha * encoder_hidden.grad / float(l1)
        jj += 1

    h = encoder_hidden.clone()
    for s in range(samples * steps):

        if random.random() < p:
            to = original_input
        else:

            from_inp = dataset[random.randint(0, len(dataset))]

            i_from, i_len_from = dataset.collate([from_inp] * BATCH_SIZE)
            encoder_outputs_from, encoder_hidden_from, embedded_input_from, lengths_from = model.encoder.get_representation_and_embedded_input(
                i_from, pool='last', return_numpy=False, tokenize=True
            )

            embedded_input_from = embedded_input_from.detach()

            to = embedded_input_from

        h = torch.autograd.Variable(h, requires_grad=True)
        outputs = model.decoder.forward([[0] * 70], h, 0)

        out_size = min(int(outputs.size(1)), int(to.size(1)))
        cropped_input = to[:, :out_size]
        cropped_output = outputs[:, :out_size]

        l1 = model.get_loss(cropped_input, lengths, cropped_output)

        l1.backward()
        # print(h.grad)

        # print(s, float(l1), alpha)
        if (l1) < cutoff:
            out_strings =  model.beam_search_decoder(outputs[0, :, :])
            out_strings = [s[:s.find('</s>')] for s in out_strings]
            print(s, float(l1), out_strings[0])
            # alpha = original_alpha
            for s in out_strings:
                output = s
                if len(text.split()) <= 2 or check_bleu(output, text) > 0.1:
                    # print(output.strip())
                    uniques.add(output.strip())
                    if get_payload(output.strip(), n=n) == payload:
                        return output.strip()
                uniques.add(s)

        h = h - alpha * h.grad / float(l1)

            # print(h)
        print('')
    print(uniques)
    print(len(uniques))
# wandering_interp('it was a beautiful sunny day')


def get_random_paraphrase_vector(paras):
    p = random.sample(paras, 1)[0]
    p1 = p[0].strip()
    p2 = p[1].strip()
    p1_i, _ = dataset.collate([(p1, len(p1.split()))] * BATCH_SIZE)
    p2_i, _ = dataset.collate([(p2, len(p2.split()))] * BATCH_SIZE)
    _, p1_h, _, _ = model.encoder.get_representation_and_embedded_input(
        p1_i, pool='last', return_numpy=False, tokenize=True
    )
    _, p2_h, _, _ = model.encoder.get_representation_and_embedded_input(
        p2_i, pool='last', return_numpy=False, tokenize=True
    )
    return p2_h - p1_h


def analogical_interp(text, paras, samples=10, steps=100, alpha=0.01, cutoff=0.1, payload=None, n=4):
    inp = (text, len(text.split()))

    original_alpha = alpha

    i, i_len = dataset.collate([inp] * BATCH_SIZE)
    i_len = i_len.to(device)

    encoder_outputs, encoder_hidden, embedded_input, lengths = model.encoder.get_representation_and_embedded_input(
        i, pool='last', return_numpy=False, tokenize=True
    )

    encoder_outputs = encoder_outputs.detach()
    encoder_hidden = encoder_hidden.detach()
    embedded_input = embedded_input.detach()
    lengths = lengths.to(device)
    lengths = lengths.detach()


    uniques = set()
    outputs = model.decoder.forward(encoder_outputs, encoder_hidden, 0)


    h = encoder_hidden.clone()
    for s in range(samples * steps):

        para_vec = get_random_paraphrase_vector(paras)

        h_para = h + para_vec.expand_as(h)
        outputs = model.decoder.forward([[0] * 70], h_para, 0)

        out_strings =  model.beam_search_decoder(outputs[0, :, :])
        out_strings = [' '.join(s[:s.find('</s>')].split()[1:]) for s in out_strings]
        # alpha = original_alpha

        # uniques.add(out_strings[0])

        output = out_strings[0].strip()
        if len(text.split()) <= 2 or check_bleu(output, text) > 0.1:
            print(output.strip(), get_payload(output.strip(), n=n))
            uniques.add(output.strip())
            if get_payload(output.strip(), n=n) == payload:
                return output.strip()


def get_paraphrases():
    paras = open('paraphrases.txt', 'r').readlines()
    paras = [p.split('\t') for p in paras]
    return paras

def get_payload(t, n=4):
    payload = hashlib.sha256(str(t).encode()).hexdigest()[:2]
    payload = bin(int(payload,16))[2:].zfill(8)[:n]
    return int(payload, 2)


def get_data(out_file='analog', max_length=1000):
    for n in range(1,5):
        with open('{}_{}.txt'.format(out_file, n), 'w', buffering=1) as out:
            i = 0
            for t in open('para_test.txt', 'r'):
                t = t.split('\t')[0]
                payload = random.randint(0, 2**n-1)
                para = analogical_interp(t, paras, samples=10, steps=100, alpha=0.01, cutoff=0.1,
                                         n=n, payload=payload)
                out.write('{}\t{}\t{}\n'.format(i, para, get_payload(para, n)))
                print('{}\t{}\t{}'.format(t, para, get_payload(para, n)), flush=True)
                i += 1
                if i >= max_length:
                    break


def get_bleu(paras, gt):
    bleu_sum = 0
    n = 0
    for p, g in zip(paras,gt):
        hypothesis = p.split('\t')[1].split()
        reference = [gg.strip().split() for gg in g.split('\t')]
        bleu = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights=(0.5, 0.5))
        n += 1
        bleu_sum += bleu
        print(n, bleu, bleu_sum/n )


def embed(t, p=None):
    return analogical_interp(t, paras, samples=10, steps=100, alpha=0.01, cutoff=0.1,
                                         n=4, payload=p)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')

    if VISDOM:
        vis = visdom.Visdom(server='http://ncc1.clients.dur.ac.uk', port=8274, env=NAME)
        for w in ['reconstruction', 'validation', 'stops', 'bow']:
            if not vis.win_exists(w):
                vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win=w)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (torch.cuda.is_available()):
        print('USING: {}'.format(torch.cuda.get_device_name(0)))


    dataset = BookCorpus('train.txt')
    dataset_val = BookCorpus('val.txt')
    dataset_test = BookCorpus('test.txt')

    if EVALUATE_FROM:
        paras = get_paraphrases()

    train_loader, validation_loader, test_loader = get_dataloaders(dataset)

    model = Seq2SeqGAN(train_loader, validation_loader, test_loader, device)
    START_TENSOR = torch.tensor(model.encoder.encoder.src_embedding.weight[model.encoder.word2id[START]],
                             dtype=torch.float32, device=device, requires_grad=False)
    END_TENSOR = torch.tensor(model.encoder.encoder.src_embedding.weight[model.encoder.word2id[END]],
                             dtype=torch.float32, device=device, requires_grad=False)
    UNK_TENSOR = torch.tensor(model.encoder.encoder.src_embedding.weight[model.encoder.word2id[UNK]],
                             dtype=torch.float32, device=device, requires_grad=False)

    optimizer_gen = optim.Adam(list(model.decoder.parameters()), lr=0.001, amsgrad=True)

    go()

    t = "It was a beautiful sunny day"
    embed(t, 4)

