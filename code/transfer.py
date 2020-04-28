
import os.path
import random

import numpy as np
import torch
import math

import torch.nn.functional as F
import visdom
from torch import optim, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.normal import Normal
from torch.utils.data.sampler import Sampler
from thinknook_text import ThinkNookText

from gensen import GenSen, GenSenSingle

package_directory = os.path.dirname(os.path.abspath("__file__"))

# CONSTANTS
UNK = u"<unk>"
START = u"<s>"
END = u"</s>"
MAX_LENGTH = 70
TRAIN_TEST_SPLIT = 0.975
VALIDATION_TEST_SPLIT = 0.8
BATCH_SIZE = 48
EMBEDDING_SIZE = 512

NAME = 'transfer_amsgrad_newdata3_lr_000005_x'

DEBUG = False
VISDOM = True
SAVE_CHECKPOINT = True
RESUME_FROM = ''
EVALUATE_FROM = ''


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



class Decoder(nn.Module):
    def __init__(self, hidden_size, latent_size, vocab_size, layers, device, clip=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.layers = 1
        self.clip = clip

        self.stop_threshold = torch.tensor([0.5], dtype=torch.float, device=device)

        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.layers)
        self.gru = ConditionalGRU(input_dim=EMBEDDING_SIZE, hidden_dim=hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.hidden_size, EMBEDDING_SIZE + 1)
        self.unembed = nn.Linear(EMBEDDING_SIZE, vocab_size)

        self.activation = nn.Tanh()
        self.stop_activation = nn.Sigmoid()

        self.device = device

    def forward(self, encoder_outputs, latent, word_dropout_rate):

        hidden = self.latent2hidden(latent)

        hidden = hidden.view(BATCH_SIZE, self.hidden_size)

        original_hidden = torch.unsqueeze(latent, 1).clone()

        input = START_TENSOR.repeat(BATCH_SIZE, 1, 1).to(device)
        unk = UNK_TENSOR.repeat(BATCH_SIZE, 1, 1).to(device)

        outputs = [self.unembed(input.view(BATCH_SIZE, EMBEDDING_SIZE))]
        stops = [torch.zeros((BATCH_SIZE, 1)).to(device)]

        max_length = len(encoder_outputs[0])
        for i in range(max_length - 1):

            use_word_drouput = random.random() < word_dropout_rate
            input = unk if use_word_drouput else input

            # just to be sure
            input = input.view(BATCH_SIZE, 1, EMBEDDING_SIZE)

            output, stop, hidden = self.step(input, hidden, original_hidden)

            input = output
            output = self.unembed(output)

            outputs.append(output)
            stops.append(stop)
            # Pre-empt only when ALL stop neurons are high
            if (stop > self.stop_threshold[0]).all():
                break

        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0, 1)

        stops = torch.stack(stops)
        stops = torch.transpose(stops, 0, 1)
        return outputs, stops

    def step(self, input, hidden, original_hidden):
        output, hidden = self.gru(input, hidden, original_hidden)
        # output = self.dropout(output)
        output = self.out(output)
        output = clip_grad(output, -self.clip, self.clip)
        output[:, :, :EMBEDDING_SIZE] = self.activation(output[:, :, :EMBEDDING_SIZE])
        output[:, :, EMBEDDING_SIZE] = self.stop_activation(output[:, :, EMBEDDING_SIZE])
        output = clip_grad(output, -self.clip, self.clip)

        stops = output[:, :, -1].view(BATCH_SIZE, 1)
        output = output[:, :, :-1].view(BATCH_SIZE, EMBEDDING_SIZE)

        return output, stops, hidden


class Seq2SeqGAN:
    def __init__(self, train_loader, validation_loader, test_loader, device):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.clip = 5

        self.latent_size = 2048
        self.decoder_hidden_size = 2048

        self.decoder_layers = 2

        self.noise = Normal(torch.tensor([0.0], requires_grad=False), torch.tensor([0.12], requires_grad=False))

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

        weight_mask = torch.ones(vocab_size).cuda()
        weight_mask[self.encoder.word2id['<pad>']] = 0
        self.criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
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

        print('{0:d} {1:d} l1: {2:.10f} l2: {3:.10f}'.format(epoch, i * BATCH_SIZE,
                                                                          losses[0], losses[1]))

        print(input_text)
        print(output_text)
        print(' ', flush=True)

    def get_loss(self, cropped_input, lengths, decoder_outputs, stops):

        l1 = self.criterion(decoder_outputs.contiguous().view(-1, decoder_outputs.size(2)),
                            cropped_input.contiguous().view(-1))


        ideal_stops = torch.zeros_like(stops)
        for i, l in enumerate(lengths):
            if l <= ideal_stops.size(1):
                ideal_stops[i, l-1] = 1
        l2 = self.bce(stops, ideal_stops)

        return l1, l2

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


        noise = self.noise.sample(encoder_hidden.size()).view_as(encoder_hidden).to(self.device)
        encoder_hidden += noise

        decoder_outputs, stops = self.decoder.forward(encoder_outputs, encoder_hidden, word_dropout_rate)

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = embedded_input[:, :decoder_outputs.size(1)]

        l1, l2 = self.get_loss(cropped_input, lengths, decoder_outputs, stops)

        loss_gen = l1 + l2
        loss_gen.backward()
        optimizer_gen.step()

        losses = np.array([l1.item(), l2.item()])
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

        decoder_outputs, stops = self.decoder.forward(encoder_outputs, encoder_hidden, 0)

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = embedded_input[:, :decoder_outputs.size(1)]
        l1, l2 = self.get_loss(cropped_input, lengths, decoder_outputs, stops)

        losses = np.array([l1.item(), l2.item()])
        return losses, decoder_outputs.data, embedded_input.data, lengths.data

    def train(self, optimizer_gen, epochs=1, print_every=500, validate_every=50000,
              word_dropout_rate=0.0, best_validation_loss=np.inf, start_at=0):

        print('USING: {}'.format(self.device))

        validations_since_best = 0
        for epoch in range(epochs):

            print_total = np.array([0.0] * 2)

            for i, (input_tensor, lengths) in enumerate(train_loader):

                lengths = lengths.to(device)

                if len(input_tensor) != BATCH_SIZE:
                    break

                samples_processed = (epoch * BATCH_SIZE * len(train_loader)) + ((i + 1) * BATCH_SIZE) + start_at

                losses, decoder_outputs, embedded_input, lengths = self.train_step(input_tensor, lengths,
                                                                        optimizer_gen, word_dropout_rate)

                print_total += losses

                if i > 0 and i % print_every == 0:
                    print_total /= print_every
                    self.print_step(embedded_input, lengths, decoder_outputs,
                                    print_total,  epoch, i)

                    for y, l in zip(print_total,
                                    ['reconstruction', 'stops']):
                        vis.line(X=np.array([int(samples_processed)]),
                                 Y=np.array([[float(y)]]),
                                 win=l,
                                 opts=dict(title=l, xlabel='samples processed', ylabel='loss', legend=['train']),
                                 update='append')
                    print_total *= 0

                if i > 0 and i % validate_every == 0:
                    val = self.validate(validation_loader, print_every, samples_processed)

                    vis.line(X=np.array([int(samples_processed)]),
                             Y=np.array([[float(val / len(validation_loader))]]),
                             win='validation',
                             opts=dict(title="validation", xlabel='samples processed', ylabel='loss', legend=['val']),
                             update='append')

                    if val < best_validation_loss:
                        best_validation_loss = val
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

        print_total = np.array([0.0] * 2)

        with torch.no_grad():

            for (i, [input_tensor, lengths]) in enumerate(validation_loader):
                if len(input_tensor) != BATCH_SIZE:
                    break

                lengths = lengths.to(device)

                losses, decoder_outputs, embedded_input, lengths = self.validation_step(input_tensor, lengths)

                print_total += losses

                if i > 0 and i % print_every == 0:

                    self.print_step(embedded_input, lengths, decoder_outputs,
                                    print_total / i, 0, i)

                del input_tensor
                del decoder_outputs
                del embedded_input


        print("AVERAGE VALIDATION LOSS: {}".format(float(print_total[0]) / len(validation_loader)))
        return float(print_total[0])

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
    indices = list(range(len(dataset)))

    rng = np.random.RandomState(0xDA7A5E7)
    rng.shuffle(indices)

    train_split_at = int(len(dataset) * TRAIN_TEST_SPLIT)
    train_indices, rest = indices[:train_split_at], indices[train_split_at:]
    validation_split_at = int(len(rest) * VALIDATION_TEST_SPLIT)
    validation_indices, test_indices = rest[:validation_split_at], rest[validation_split_at:]

    train_lengths = {i : dataset[i][1] for i in train_indices}
    validation_lengths =  {i : dataset[i][1] for i in validation_indices}
    test_lengths = {i :dataset[i][1] for i in test_indices}

    BUCKET_SIZE = 5
    train_sampler = BucketSampler(train_indices, train_lengths, BATCH_SIZE, BUCKET_SIZE)
    validation_sampler = BucketSampler(validation_indices, validation_lengths, BATCH_SIZE, BUCKET_SIZE)
    test_sampler = BucketSampler(test_indices, test_lengths, BATCH_SIZE, BUCKET_SIZE)


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                               collate_fn=dataset.collate, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler,
                                                    collate_fn=dataset.collate)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler,
                                              collate_fn=dataset.collate)
    print('{} TRAIN  {} VALIDATE  {} TEST'.format(len(train_loader), len(validation_loader), len(test_loader)))
    return train_loader, validation_loader, test_loader

def go():
    if RESUME_FROM:
        model.decoder.load_state_dict(torch.load('{}_decoder.p.tar'.format(RESUME_FROM), map_location='cuda'))
        model.decoder.train()

        x = torch.load('{}.p.tar'.format(RESUME_FROM), map_location='cpu')
        samples_processed = x['samples_processed'] if 'samples_processed' in x else 0
        val = x['best_validation_loss']

        optimizer_gen.load_state_dict(torch.load('{}_optimizer_gen.p.tar'.format(RESUME_FROM), map_location='cuda'))

        print("STARTING FROM {} SAMPLES".format(samples_processed))
        model.train(optimizer_gen, epochs=50000,
                    print_every=int(50000 / BATCH_SIZE), validate_every=int(750000 / BATCH_SIZE),
                    start_at=samples_processed, best_validation_loss=val)
    elif EVALUATE_FROM:
        model.decoder.load_state_dict(torch.load('{}_decoder.p.tar'.format(EVALUATE_FROM), map_location='cuda'))

    else:
        model.train(optimizer_gen, epochs=50000,
                    print_every=int(50000 / BATCH_SIZE), validate_every=int(250000 / BATCH_SIZE))


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


    dataset = ThinkNookText()

    train_loader, validation_loader, test_loader = get_dataloaders(dataset)

    model = Seq2SeqGAN(train_loader, validation_loader, test_loader, device)
    START_TENSOR = torch.tensor(model.encoder.encoder.src_embedding.weight[model.encoder.word2id[START]],
                             dtype=torch.float32, device=device, requires_grad=False)
    END_TENSOR = torch.tensor(model.encoder.encoder.src_embedding.weight[model.encoder.word2id[END]],
                             dtype=torch.float32, device=device, requires_grad=False)
    UNK_TENSOR = torch.tensor(model.encoder.encoder.src_embedding.weight[model.encoder.word2id[UNK]],
                             dtype=torch.float32, device=device, requires_grad=False)

    optimizer_gen = optim.Adam(list(model.decoder.parameters()), lr=0.00005, betas=(0.5, 0.9), amsgrad=True)

    go()

