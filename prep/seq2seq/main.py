from io import open
import unicodedata
import string
import re
import random
import pickle

import torch
print(torch.__version__)
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from consts import *
from lang import Language
from input import read_file, read_cache
from model import Encoder, Decoder
from training import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if USE_CACHE:
    input_lang, output_lang, pairs = (read_cache(c) for c in CACHE)
else:
    input_lang, output_lang, pairs = read_file(FILE, INCLUDE_PHRASES, suffix='i')

random.shuffle(pairs)
train_set = pairs[:int(0.8 * len(pairs))]
test_set = pairs[int(0.8 * len(pairs)):]

validation_set = train_set[int(0.8 * len(train_set)):]
train_set = train_set[:int(0.8 * len(train_set))]

print("{} train   {} validation    {} test".format(len(train_set), len(validation_set), len(test_set)), flush=True)


def pair2tensors(pair):
    input_tensor = input_lang.sentence2tensor(pair[0], device)
    target_tensor = output_lang.sentence2tensor(pair[1], device)
    return (input_tensor, target_tensor)


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def decode_output(decoder_outputs):
    decoded_words = []
    for decoder_output in decoder_outputs:
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[topi.item()])

    return ' '.join(decoded_words)

def train_iterations(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    random.shuffle(train_set)
    random.shuffle(validation_set)

    criterion = nn.NLLLoss()

    iter = 0
    while iter < n_iters:
        for training_pair in train_set:
            training_tensors = pair2tensors(training_pair)
            input_tensor = training_tensors[0]
            target_tensor = training_tensors[1]

            loss, decoder_output = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, device)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('{0:d} {1:.5f}'.format(iter, loss))

                print(training_pair[0])
                print(training_pair[1])
                print(decode_output(decoder_output), flush=True)
                print()

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if iter == n_iters:
                break

            iter += 1

        print("VALIDATING")

        for pair in validation_set:
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

        random.shuffle(train_set)


    show_plot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = input_lang.sentence2tensor(sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate_randomly(encoder, decoder, n=10):
    print("TESTING")
    for pair in test_set:
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



if __name__ == '__main__':
    # Note: You must put all your training code into one function rather than in the global scope
    #       (this is good practice anyway).
    #       Subsequently you must call the set_start_method and your main function from inside this
    #       if-statement. If you don't do that, each worker will attempt to run all of your training
    #       code and everything will go very wild and very wrong.
    torch.multiprocessing.set_start_method('forkserver')
    hidden_size = 512
    encoder1 = Encoder(input_lang.n_words, hidden_size, device).to(device)
    attn_decoder1 = Decoder(hidden_size, output_lang.n_words, device, dropout_p=0.1).to(device)

    train_iterations(encoder1, attn_decoder1, 100000, print_every=50)
    evaluate(encoder1, attn_decoder1)

