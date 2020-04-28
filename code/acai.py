import html
import os.path
import pickle
import random
import re
import time
import h5py

import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
import visdom
from torch import optim, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.normal import Normal
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from thinknook import ThinkNook

package_directory = os.path.dirname(os.path.abspath("__file__"))

# CONSTANTS
UNK = u"<?>"
START = u"</s>"
END = u"</>"
MAX_LENGTH = 70
TRAIN_TEST_SPLIT = 0.975
VALIDATION_TEST_SPLIT = 0.8
BATCH_SIZE = 64

NAME = 'dubs2_noreparam_batch64_00015in10mil'

DEBUG = False
VISDOM = True
SAVE_CHECKPOINT = True
RESUME_FROM = ''
EVALUATE_FROM = ''


def dprint(s):
    if DEBUG:
        print(s)


def save_checkpoint(encoder, decoder, disc_prior, disc_output, optimizer_gen, optimizer_disc, samples_processed,
                    best_validation_loss, filename=NAME):

    if SAVE_CHECKPOINT:
        state = {
            'samples_processed': samples_processed,
            'best_validation_loss': best_validation_loss,
        }
        torch.save(state, '{}.p.tar'.format(filename))
        torch.save(encoder.state_dict(), '{}_encoder.p.tar'.format(filename))
        torch.save(decoder.state_dict(), '{}_decoder.p.tar'.format(filename))
        torch.save(disc_prior.state_dict(), '{}_disc_prior.p.tar'.format(filename))
        torch.save(disc_output.state_dict(), '{}_disc_output.p.tar'.format(filename))
        torch.save(optimizer_gen.state_dict(), '{}_optimizer_gen.p.tar'.format(filename))
        torch.save(optimizer_disc.state_dict(), '{}_optimizer_disc.p.tar'.format(filename))


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


class Embedder(nn.Module):
    def __init__(self, weight_matrix, device):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weight_matrix)
        self.device = device

    def forward(self, input):
        embedded_input = self.embedding(input)
        return  embedded_input


class Encoder(nn.Module):
    def __init__(self, hidden_size, latent_size, layers, device, bidirectional=False, clip=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.layers = layers
        self.clip = clip

        self.hidden_factor = (2 if self.bidirectional else 1) * self.layers

        self.gru = nn.GRU(input_size=401, hidden_size=self.hidden_size, bidirectional=bidirectional
                          ,num_layers=self.layers, dropout=0.4)
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        # self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        self.device = device

    def forward(self, input, hidden, lengths):
        # dprint('hidden {} = {}'.format(hidden.size(), hidden))
        # takes input of shape (seq_len, batch, input_size)
        input = pack_padded_sequence(input, list(lengths.data))
        output, hidden = self.gru(input, hidden)
        hidden = clip_grad(hidden, -self.clip, self.clip)

        if self.bidirectional or self.layers > 1:
            # flatten hidden state
            s = (2 if self.bidirectional else 1)
            hidden = hidden.view(BATCH_SIZE, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # output shape of seq_len, batch, num_directions * hidden_size
        # dprint('hidden {} = {}'.format(hidden.size(), hidden))

        latent = self.reparameterize(hidden)
        # dprint('output {} = {}'.format(output.size(), output))
        return output, latent

    def reparameterize(self, hidden):
        mean = self.hidden2mean(hidden)
        return mean
        # logv = self.hidden2logv(hidden)
        # std = torch.exp(0.5 * logv)
        #
        # z = torch.randn([BATCH_SIZE, self.latent_size], device=self.device)
        # z = z * std + mean
        # return z

    def init_hidden(self):
        s = 2 if self.bidirectional else 1
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(s * self.layers, BATCH_SIZE, self.hidden_size, device=self.device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, latent_size, layers, device, bidirectional=False, clip=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.layers = layers
        self.clip = clip

        s = (2 if self.bidirectional else 1)
        self.hidden_factor = s * self.layers
        self.stop_threshold = torch.tensor([0.5], dtype=torch.float, device=device)

        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.gru = nn.GRU(input_size=401, hidden_size=self.hidden_size, bidirectional=bidirectional
                          , num_layers=self.layers)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.hidden_size * s, 401)

        self.activation = nn.Tanh()
        self.stop_activation = nn.Sigmoid()

        self.device = device

    def forward(self, encoder_outputs, latent, teacher_forcing_p, original_input, word_dropout_rate):

        hidden = self.latent2hidden(latent)
        original_input = original_input.to(self.device)


        if self.bidirectional or self.layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, BATCH_SIZE, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        input = START_TENSOR.repeat(1, BATCH_SIZE, 1).to(device)
        unk = UNK_TENSOR.repeat(1, BATCH_SIZE, 1).to(device)

        # dprint('input {} = {}'.format(input.size(), input))

        outputs = [input.view(BATCH_SIZE, 401)]

        # dprint('encoder_outputs {} = {}'.format(encoder_outputs.size(), encoder_outputs))
        # dprint('original_input {} = {}'.format(original_input.size(), original_input))
        max_length = len(encoder_outputs)
        for i in range(max_length - 1):

            use_teacher_forcing = random.random() < teacher_forcing_p
            # dprint('forced' if use_teacher_forcing else 'unforced')
            input = original_input[i, :, :] if use_teacher_forcing else input

            use_word_drouput = random.random() < word_dropout_rate
            input = unk if use_word_drouput else input

            # just to be sure
            input = input.view(1, -1, 401)


            # dprint('input {} = {}'.format(input.size(), input))
            # dprint('hidden {} = {}'.format(hidden.size(), hidden))
            output, hidden = self.step(input, hidden)

            output = output.view(BATCH_SIZE, 401)
            # dprint('output {} = {}'.format(output.size(), output))

            # dprint('forced' if use_teacher_forcing else 'unforced')
            input = output

            outputs.append(output)
            # Pre-empt only when ALL stop neurons are high
            if (output[:, 400] > self.stop_threshold[0]).all():
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
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.hidden_factor, BATCH_SIZE, self.hidden_size, device=self.device)


# inspired by https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/
class PriorDiscriminator(nn.Module):
    def __init__(self, latent_size, hidden_size, device):
        super(PriorDiscriminator, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.lin1 = nn.Linear(self.latent_size, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin3 = nn.Linear(self.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.2)

        self.activation = nn.Sigmoid()

        self.device = device

    def forward(self, x):
        x = self.dropout(self.lin1(x))
        x = F.relu(x)
        x = self.dropout(self.lin2(x))
        x = self.lin3(x)
        x = self.activation(x)
        return x

#
class OutputDiscriminator(nn.Module):
    def __init__(self, hidden_size, layers, device, bidirectional=False, clip=None):
        super(OutputDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.layers = layers
        self.clip = clip

        self.hidden_factor = (2 if self.bidirectional else 1) * self.layers

        self.gru = nn.GRU(input_size=401, hidden_size=self.hidden_size, bidirectional=bidirectional
                          ,num_layers=self.layers, dropout=0.4)
        self.hidden2out = nn.Linear(hidden_size * self.hidden_factor, 1)
        self.activation = nn.Sigmoid()

        self.device = device

    def forward(self, input, hidden):
        # dprint('hidden {} = {}'.format(hidden.size(), hidden))
        # takes input of shape (seq_len, batch, input_size)
        # input = pack_padded_sequence(input, list(lengths.data))
        _, hidden = self.gru(input, hidden)
        hidden = clip_grad(hidden, -self.clip, self.clip)

        if self.bidirectional or self.layers > 1:
            # flatten hidden state
            s = (2 if self.bidirectional else 1)
            hidden = hidden.view(BATCH_SIZE, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # output shape of seq_len, batch, num_directions * hidden_size
        # dprint('hidden {} = {}'.format(hidden.size(), hidden))

        output = self.hidden2out(hidden)
        output = self.activation(output)
        # dprint('output {} = {}'.format(output.size(), output))
        return output

    def init_hidden(self):
        s = 2 if self.bidirectional else 1
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(s * self.layers, BATCH_SIZE, self.hidden_size, device=self.device)


class Seq2SeqGAN:
    def __init__(self, weight_matrix, index2word, train_loader, validation_loader, test_loader, device):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.weight_matrix = weight_matrix
        self.i2w = index2word

        self.reverse_input = False
        self.bidirectional = True
        self.clip = 5

        self.encoder_hidden_size = 512
        self.decoder_hidden_size = 512
        self.output_discriminator_hidden_size=512
        self.latent_size = 1024

        self.encoder_layers = 4
        self.decoder_layers = 4

        self.embedder = Embedder(weight_matrix, device).to(device)
        self.encoder = Encoder(self.encoder_hidden_size, self.latent_size, self.encoder_layers, device=device,
                               bidirectional=self.bidirectional, clip=5).to(device)
        self.decoder = Decoder(self.decoder_hidden_size, self.latent_size, self.decoder_layers, device=device,
                               bidirectional=self.bidirectional, clip=5).to(device)
        self.prior_discriminator = PriorDiscriminator(self.latent_size, self.latent_size, device=device).to(device)
        self.output_discriminator = OutputDiscriminator(self.output_discriminator_hidden_size, layers=1,
                                                        device=device, bidirectional=True, clip=5).to(device)

        self.criterion = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss()
        self.device = device

        print(self.embedder)
        print(self.encoder)
        print(self.decoder)
        print(self.prior_discriminator)
        print(self.output_discriminator)

    def print_step(self, input_tensor, lengths, decoder_outputs, l1, l2, l3, epoch, i):

        # dprint('input_tensor {} = {}'.format(input_tensor.size(), input_tensor))
        # dprint('lengths {} = {}'.format(lengths.size(), lengths))

        # print out a medium sized tweet
        mid = int(BATCH_SIZE / 2)
        input_to_print = input_tensor[:lengths[mid], mid, :]
        output_to_print = decoder_outputs[:lengths[mid], mid, :]
        #
        # dprint('input_to_print {} = {}'.format(input_to_print.size(), input_to_print))
        # dprint('input_to_print {} = {}'.format(output_to_print.size(), output_to_print))

        input_text = self.unembed(input_to_print)
        output_text = self.unembed(output_to_print)

        print('{0:d} {1:d} l1: {2:.10f}, l2: {3:.10f}, l3: {4:.10f}'.format(epoch, i * BATCH_SIZE, l1, l2, l3))

        print(input_text)
        print(output_text)
        print(' ', flush=True)

    def get_loss(self, cropped_input, hidden, lengths, decoder_outputs):
        # mask out decoder outputs in positions that don't have a corresponding original input
        # we don't want to define a loss on these outputs
        mask = torch.tensor([[[int(torch.max(j.abs()) > 0)] * 401 for j in i] for i in cropped_input],
                            dtype=torch.float, device=self.device)
        decoder_outputs = decoder_outputs * mask

        unmasked_count = torch.nonzero(mask).size(0)
        l1 = self.criterion(decoder_outputs, cropped_input) / unmasked_count

        # # GAN bits
        # d_prior_fake = self.prior_discriminator.forward(hidden)
        # # # generator loss
        # l2 = self.bce(d_prior_fake.mean(), torch.ones(1)[0].to(device))
        #
        # d_output_hidden = self.output_discriminator.init_hidden()
        # d_output_fake = self.output_discriminator.forward(decoder_outputs, d_output_hidden)
        # # generator loss
        # l3 = self.bce(d_output_fake.mean(), torch.ones(1)[0].to(device))
        #
        # return l1, l2, l3, hidden
        return l1
        # return l1, l1, l1, hidden

    def calc_gradient_penalty(D, real_data, fake_data, lmbda=10):
        alpha = torch.rand_like(real_data).to(device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates.requires_grad = True
        disc_interpolates = self.prior_discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
        return gradient_penalty


    def get_discriminator_loss(self, hidden, input_seq, output_seq):
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        real_gaussian = m.sample(hidden.size()).to(self.device)
        real_gaussian = real_gaussian.view_as(hidden)

        d_prior_real, d_prior_fake = self.prior_discriminator.forward(real_gaussian),\
                                     self.prior_discriminator.forward(hidden.detach())

        l_prior_real = self.bce(d_prior_real.mean(), torch.ones(1)[0].to(device))
        l_prior_fake = self.bce(d_prior_fake.mean(), torch.zeros(1)[0].to(device))
        grad_penalty_prior = self.calc_gradient_penalty(real_gaussian.data, hidden.data,
                                                               d_prior_real, d_prior_fake, lmbda=1)

        l4 = (l_prior_real + l_prior_fake + grad_penalty_prior)

        d_output_hidden = self.output_discriminator.init_hidden()
        d_output_real = self.output_discriminator.forward(input_seq.detach(), d_output_hidden)

        d_output_hidden = self.output_discriminator.init_hidden()
        d_output_fake = self.output_discriminator.forward(output_seq.detach(), d_output_hidden)

        l_output_real = self.bce(d_output_real.mean(),  torch.ones(1)[0].to(device))
        l_output_fake = self.bce(d_output_fake.mean(), torch.zeros(1)[0].to(device))
        grad_penalty_output = self.calc_output_gradient_penalty(input_seq.data, output_seq.data,
                                                                d_output_real, d_output_fake, lmbda=1)

        l5 = (l_output_real + l_output_fake + grad_penalty_output)
        return l4, l5
        # return l4, l4

    def train_step(self, input_tensor, lengths, optimizer_gen, optimizer_disc, teacher_forcing_p, word_dropout_rate, lambda_g,
                   disc_iters=10, phase='both'):
        encoder_hidden = self.encoder.init_hidden()

        optimizer_gen.zero_grad()

        if self.reverse_input:
            input_tensor = input_tensor.flip(0)

        unfreeze(self.encoder)
        unfreeze(self.decoder)
        freeze(self.prior_discriminator)
        freeze(self.output_discriminator)

        alpha = 0.5 - (torch.rand((BATCH_SIZE, self.latent_size)) - 0.5).abs()


        embedded_input = self.embedder.forward(input_tensor)
        encoder_outputs, encoder_hidden = self.encoder.forward(embedded_input, encoder_hidden, lengths)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)

        encoder_hidden_mix = alpha * encoder_hidden + (1 - alpha) * encoder_hidden[::-1, :]

        decoder_outputs = self.decoder.forward(encoder_outputs, encoder_hidden, teacher_forcing_p, embedded_input,
                                               word_dropout_rate)
        decoder_mix_outputs = self.decoder.forward(encoder_outputs, encoder_hidden_mix, teacher_forcing_p, embedded_input,
                                               word_dropout_rate)

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = embedded_input[:decoder_outputs.size(0), :, :]
        l1 = self.get_loss(cropped_input, encoder_hidden, lengths, decoder_outputs)

        loss_disc = tf.reduce_mean(
            tf.square(disc(decode_mix) - alpha[:, 0, 0, 0]))

        l2 = self.discriminator(d)


        freeze(self.encoder)
        freeze(self.decoder)
        unfreeze(self.output_discriminator)
        unfreeze(self.prior_discriminator)


        for i in range(disc_iters):
            optimizer_disc.zero_grad()
            l4, l5 = self.get_discriminator_loss(hidden, cropped_input, decoder_outputs)
            loss_disc = l4 * lambda_g + l5 * lambda_g
            # loss_disc = l4 * lambda_g
            loss_disc.backward()
            optimizer_disc.step()

        # print('rest')
        return l1.item(), l2.item(), l3.item(), l4.item(), l5.item(), decoder_outputs.data, embedded_input.data
        # return l1.item(), l2.item(), l3.item(), l4.item(), l5.item(), decoder_outputs.detach(), embedded_input.detach()

    def train(self, optimizer_gen, optimizer_disc, epochs=1, print_every=500, validate_every=50000,
              initial_teacher_forcing_p=0.0, final_teacher_forcing_p=0.0, teacher_force_decay=0.0000003,
              word_dropout_rate=0.5, best_validation_loss=np.inf, start_at=0, final_lambda_g = 0.00015,
              lambda_g_step= 0.00015/10000000):

        print('USING: {}'.format(self.device))

        validations_since_best = 0
        for epoch in range(epochs):

            print_l1_total, print_l2_total, print_l3_total, print_l4_total, print_l5_total = 0, 0, 0, 0, 0

            for i, (input_tensor, lengths) in enumerate(train_loader):
                freeze(self.embedder)

                input_tensor = input_tensor.to(device)
                lengths = lengths.to(device)

                if input_tensor.size(1) != BATCH_SIZE:
                    break

                samples_processed = (epoch * BATCH_SIZE * len(train_loader)) + ((i + 1) * BATCH_SIZE) + start_at

                teacher_forcing_p = max(final_teacher_forcing_p,
                                        initial_teacher_forcing_p - teacher_force_decay * samples_processed)
                lambda_g = min(final_lambda_g, lambda_g_step * samples_processed)
                # lambda_g = 0.0035 if random.random() < 0.9 else 0

                l1, l2, l3, l4, l5, decoder_outputs, embedded_input = self.train_step(input_tensor, lengths,
                                                                              optimizer_gen, optimizer_disc,
                                                                              teacher_forcing_p, word_dropout_rate,
                                                                              lambda_g)
                print_l1_total += l1
                print_l2_total += l2
                print_l3_total += l3
                print_l4_total += l4
                print_l5_total += l5

                # dprint(samples_processed)
                if i > 0 and i % print_every == 0:
                    print_l1_avg = print_l1_total / print_every
                    print_l2_avg = print_l2_total / print_every
                    print_l3_avg = print_l3_total / print_every
                    print_l4_avg = print_l4_total / print_every
                    print_l5_avg = print_l5_total / print_every

                    self.print_step(embedded_input, lengths, decoder_outputs,
                                    print_l1_avg, print_l2_avg, print_l3_avg,  epoch, i)
                    print_l1_total, print_l2_total, print_l3_total, print_l4_total, print_l5_total = 0, 0, 0, 0, 0

                    for y, l in zip([print_l1_avg, print_l2_avg, print_l3_avg, print_l4_avg, print_l5_avg],
                                    ['reconstruction', 'prior generator', 'output generator', 'prior discriminator',
                                     'output discriminator']):
                        # noinspection PyArgumentList
                        vis.line(X=np.array([int(samples_processed)]),
                                 Y=np.array([[float(y)]]),
                                 win=l,
                                 opts=dict(title=l, xlabel='samples processed', ylabel='loss', legend=['train']),
                                 update='append')

                if i > 0 and i % validate_every == 0:
                    val = self.validate(validation_loader, print_every, samples_processed)

                    # noinspection PyArgumentList
                    vis.line(X=np.array([int(samples_processed)]),
                             Y=np.array([[float(val / len(validation_loader))]]),
                             win='validation_loss',
                             opts=dict(title="validation loss", xlabel='samples processed', ylabel='loss', legend=['val']),
                             update='append')

                    if val < best_validation_loss:
                        best_validation_loss = val
                        validations_since_best = 0

                        save_checkpoint(self.encoder, self.decoder, self.prior_discriminator,
                                        self.output_discriminator,
                                        optimizer_gen, optimizer_disc, samples_processed,
                                        best_validation_loss)
                        # save_checkpoint(self.encoder, self.decoder, self.prior_discriminator,
                        #                 self.prior_discriminator,
                        #                 optimizer_gen, optimizer_disc, samples_processed,
                        #                 best_validation_loss)
                    else:
                        validations_since_best += 1

                    print("{} SINCE LAST BEST VALIDATION".format(validations_since_best))

                    if validations_since_best >= 100:
                        return

                del input_tensor
                del decoder_outputs
                del embedded_input


    def validation_step(self, input_tensor, lengths):
        encoder_hidden = self.encoder.init_hidden()

        if self.reverse_input:
            input_tensor = input_tensor.flip(0)

        embedded_input = self.embedder.forward(input_tensor)
        encoder_outputs, encoder_hidden = self.encoder.forward(embedded_input, encoder_hidden, lengths)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)
        decoder_outputs = self.decoder.forward(encoder_outputs, encoder_hidden, 0, input_tensor, 0)

        # resize input to match decoder output (due to pre-empting decoder)
        cropped_input = embedded_input[:decoder_outputs.size(0), :, :]
        l1, l2, l3, hidden = self.get_loss(cropped_input, encoder_hidden, lengths, decoder_outputs)
        # l3 = self.get_discriminator_loss(hidden)

        del l3
        del encoder_outputs
        del cropped_input
        return l1.item(), l2.item(),  decoder_outputs.data,\
               encoder_hidden.data, embedded_input.data



    def validate(self, validation_loader, print_every, samples_processed):
        print("VALIDATING")

        total_l1_loss, total_l2_loss, total_l3_loss = 0, 0, 0

        with torch.no_grad():
            hiddens = []

            for (i, [input_tensor, lengths]) in enumerate(validation_loader):
                if input_tensor.size(1) != BATCH_SIZE:
                    break

                input_tensor = input_tensor.to(device)
                lengths = lengths.to(device)

                l1, l2, decoder_outputs, hidden, embedded_input = self.validation_step(input_tensor, lengths)
                hiddens.append(hidden)
                total_l1_loss += l1
                total_l2_loss += l2

                if i > 0 and i % print_every == 0:
                    print_l1_loss = total_l1_loss / i
                    print_l2_loss = total_l2_loss / i
                    print_l3_loss = total_l3_loss / i
                    self.print_step(embedded_input, lengths, decoder_outputs,
                                    print_l1_loss, print_l2_loss, print_l3_loss, 0, i)

                del input_tensor
                del decoder_outputs
                del hidden
                del embedded_input

            hidden_hists(random.randint(0,7),0,random.randint(0,8),
                         hiddens=hiddens, label=samples_processed//BATCH_SIZE)


        print("AVERAGE VALIDATION LOSS: {}".format((total_l1_loss) / len(validation_loader)))
        return total_l1_loss

    def unembed(self, decoder_outputs, length=MAX_LENGTH):

        indices = [torch.argmax(torch.mm(self.weight_matrix, torch.unsqueeze(d, 1).to('cpu'))) for d in decoder_outputs]
        return ' '.join([self.i2w[i] for i in indices])

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

    # train_sampler = SubsetRandomSampler(train_indices)
    # validation_sampler = SubsetRandomSampler(validation_indices)
    # test_sampler = SubsetRandomSampler(test_indices)

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
        model.encoder.load_state_dict(torch.load('{}_encoder.p.tar'.format(RESUME_FROM), map_location='cuda'))
        model.decoder.load_state_dict(torch.load('{}_decoder.p.tar'.format(RESUME_FROM), map_location='cuda'))
        model.prior_discriminator.load_state_dict(torch.load('{}_disc_prior.p.tar'.format(RESUME_FROM), map_location='cuda'))
        model.output_discriminator.load_state_dict(torch.load('{}_disc_output.p.tar'.format(RESUME_FROM), map_location='cuda'))
        model.encoder.train()
        model.decoder.train()
        model.prior_discriminator.train()
        model.output_discriminator.train()

        x = torch.load('{}.p.tar'.format(RESUME_FROM), map_location='cpu')
        samples_processed = x['samples_processed'] if 'samples_processed' in x else 0
        val = x['best_validation_loss']

        del x
        optimizer_gen.load_state_dict(torch.load('{}_optimizer_gen.p.tar'.format(RESUME_FROM), map_location='cuda'))
        optimizer_disc.load_state_dict(torch.load('{}_optimizer_disc.p.tar'.format(RESUME_FROM), map_location='cuda'))

        model.train(optimizer_gen, optimizer_disc, epochs=50000,
                    print_every=int(250000 / BATCH_SIZE), validate_every=int(1500000 / BATCH_SIZE),
                    start_at=samples_processed, best_validation_loss=val)

    elif EVALUATE_FROM:
        model.encoder.load_state_dict(torch.load('{}_encoder.p.tar'.format(EVALUATE_FROM), map_location='cuda'))
        model.decoder.load_state_dict(torch.load('{}_decoder.p.tar'.format(EVALUATE_FROM), map_location='cuda'))
        model.prior_discriminator.load_state_dict(torch.load('{}_disc_prior.p.tar'.format(EVALUATE_FROM), map_location='cuda'))
        model.output_discriminator.load_state_dict(torch.load('{}_disc_output.p.tar'.format(EVALUATE_FROM), map_location='cuda'))

        optimizer_gen.load_state_dict(torch.load('{}_optimizer_gen.p.tar'.format(EVALUATE_FROM), map_location='cuda'))
        optimizer_disc.load_state_dict(torch.load('{}_optimizer_disc.p.tar'.format(EVALUATE_FROM), map_location='cuda'))

    else:
        model.train(optimizer_gen, optimizer_disc, epochs=50000,
                    print_every=int(250000 / BATCH_SIZE), validate_every=int(1500000 / BATCH_SIZE))

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
        i, i_len = dataset.collate([dataset[i]] * BATCH_SIZE)
        i = i.to(device)
        i_len = i_len.to(device)

        em = model.embedder.forward(i)
        print(model.unembed(em[:i_len[0], 0, :]))

        encoder_hidden = model.encoder.init_hidden()
        embedded_input = model.embedder.forward(i)
        encoder_outputs, encoder_hidden = model.encoder.forward(embedded_input, encoder_hidden, i_len)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)

        outputs = model.decoder.forward(encoder_outputs, encoder_hidden, 0, encoder_outputs, 0)
        print(model.unembed(outputs[:, 0, :]))
        for s in range(steps):
            h = encoder_hidden + torch.tensor(np.random.normal(loc=0, scale=scale, size=np.shape(encoder_hidden)),
                                              dtype=torch.float, device=device)
            outputs = model.decoder.forward(encoder_outputs, h, 0, encoder_outputs, 0)
            print(model.unembed(outputs[:, 0, :]))

def hidden2tweet(eo, eh):
    outputs = model.decoder.forward(eo, eh, 0, eo, 0)
    print(model.unembed(outputs))


def hidden_hists(j ,k, l, hiddens=None, label="hists"):
    qs = []

    if hiddens:
        for h in hiddens:
            for x in range(BATCH_SIZE):
                first_unit = h[x, l]
                qs.append(float(first_unit))
    else:
        for (i, [input_tensor, lengths]) in enumerate(validation_loader):

            if input_tensor.size(1) != BATCH_SIZE:
                break

            input_tensor = input_tensor.to(device)

            encoder_hidden = model.encoder.init_hidden()
            embedded_input = model.embedder.forward(input_tensor)
            _, encoder_hidden = model.encoder.forward(embedded_input, encoder_hidden, lengths)

            for x in range(BATCH_SIZE):
                first_unit = encoder_hidden[x, l]
                qs.append(float(first_unit))

    torch.tensor(qs).view(-1)
    vis.histogram(qs, opts=dict(title=label))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')

    if VISDOM:
        vis = visdom.Visdom(server='http://ncc1.clients.dur.ac.uk', port=8274, env=NAME)
        for w in ['reconstruction', 'prior generator', 'output generator', 'prior discriminator',
                                     'output discriminator', 'validation loss']:
            if not vis.win_exists(w):
                vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win=w)

    pd.options.display.max_colwidth = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (torch.cuda.is_available()):
        print('USING: {}'.format(torch.cuda.get_device_name(0)))

    i2w =  torch.load('i2w.p.tar')
    weight_matrix = torch.load('weight_matrix.p.tar')


    START_TENSOR = torch.tensor(weight_matrix[i2w.index(START)], dtype=torch.float32, device=device, requires_grad=False)
    END_TENSOR = torch.tensor(weight_matrix[i2w.index(END)], dtype=torch.float32, device=device, requires_grad=False)
    UNK_TENSOR = torch.tensor(weight_matrix[i2w.index(UNK)], dtype=torch.float32, device=device, requires_grad=False)
    PADDING_TENSOR = torch.zeros_like(START_TENSOR)


    dataset = ThinkNook(from_cache=True, name='ThinkNook')

    train_loader, validation_loader, test_loader = get_dataloaders(dataset)

    model = Seq2SeqGAN(weight_matrix, i2w, train_loader, validation_loader, test_loader, device)

    optimizer_gen = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.0001)
    # optimizer_disc = optim.Adam(list(model.prior_discriminator.parameters()), lr=0.00001)
    optimizer_disc = optim.Adam(list(model.prior_discriminator.parameters())
                                + list(model.output_discriminator.parameters()), lr=0.00001)
    go()

    #
# for i, t in enumerate(dataset.samples):
#     vec = np.zeros((len(t) + 2,), dtype=np.uint32)
#     vec[0] = w2i[START]
#     for j, w in enumerate(t):
#         vec[j+1] = w2i[w] if w in w2i else w2i[UNK]
#     vec[j+2] = w2i[END]
#     file['tweets2'][i] = vec
#     print(i)


