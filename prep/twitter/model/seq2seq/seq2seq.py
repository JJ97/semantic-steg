import torch
import random
import gc

from .encoder import Encoder
from .decoder import Decoder
from embedding.onehot import OneHot
from torch import optim, nn
from constants import *

class Seq2Seq:

    def __init__(self, embedder, device):
        self.embedder = embedder
        self.encoder = Encoder(embedder.input_size, hidden_size=512, device=device).to(device)
        self.decoder = Decoder(embedder.input_size, hidden_size=512, device=device,
                               dropout_p=0.1).to(device)
        self.device = device


    def get_decoder_output(self, input_tensor, encoder_outputs, encoder_hidden, criterion, use_teacher_forcing):
        decoder_input = self.embedder.start_tensor
        decoder_hidden = encoder_hidden

        loss = 0
        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(input_tensor.size(0)):
                decoder_output, decoder_hidden, decoder_attention = self.decoder.forward(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, input_tensor[di])
                decoder_input = input_tensor[di]  # Teacher forcing
                decoder_outputs.append(decoder_output)
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(input_tensor.size(0)):
                decoder_output, decoder_hidden, decoder_attention = self.decoder.forward(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, input_tensor[di])
                decoder_outputs.append(decoder_output)
                if decoder_input.item() == self.embedder.end_tensor:
                    break

        return  decoder_outputs, loss


    def train_iteration(self, input_tensor, encoder_optimizer, decoder_optimizer,
                        criterion, teacher_forcing_ratio):
        encoder_hidden = self.encoder.init_hidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=self.device)

        for ei in range(input_tensor.size(0)):
            encoder_output, encoder_hidden = self.encoder.forward(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        use_teacher_forcing = random.random() < teacher_forcing_ratio

        decoder_outputs, loss = self.get_decoder_output(input_tensor, encoder_outputs, encoder_hidden,
                                                        criterion, use_teacher_forcing)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        avg_loss = float(loss.item()) / input_tensor.size(0)
        out = self.output2tweet(decoder_outputs)

        del input_tensor
        del loss
        del decoder_outputs
        del encoder_outputs


        return avg_loss, out

    def validation_iteration(self, input_tensor, criterion):
        encoder_hidden = self.encoder.init_hidden()

        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=self.device)

        for ei in range(input_tensor.size(0)):
            encoder_output, encoder_hidden = self.encoder.forward(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_outputs, loss = self.get_decoder_output(input_tensor, encoder_outputs, encoder_hidden,
                                                        criterion, use_teacher_forcing=False)
        
        out = self.output2tweet(decoder_outputs)
        l = float(loss.item())

        del input_tensor
        del loss
        del decoder_outputs
        del encoder_outputs

        return out, l
        
    def output2tweet(self, decoder_outputs):
        decoded_words = []
        for decoder_output in decoder_outputs:
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == self.embedder.end_tensor:
                decoded_words.append(END)
                break
            else:
                decoded_words.append(self.embedder.unembed(topi.item()))
        return ' '.join(decoded_words)


    def validate(self, data, criterion):
        print("VALIDATING")

        total_validation_loss = 0

        with torch.no_grad():
            for sample in data.validation_set:
                input_tensor = self.embedder.embed(sample)
                print('>', ' '.join(sample))
                decoder_outputs, loss = self.validation_iteration(input_tensor, criterion)
                total_validation_loss += loss
                output_sentence = decoder_outputs
                print('<', output_sentence)
                print('loss ', loss)
                print(' ', flush=True)

        print("AVERAGE VALIDATION LOSS: {}".format(total_validation_loss / data.validation_set.size()))




    def train(self, data, iterations, print_every=500, validate_every=25000,
              learning_rate=0.0025, teacher_forcing_ratio=0.5):
        criterion = nn.NLLLoss()

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        print_loss_total = 0  # Reset every print_every
        for iteration in range(iterations):

            data.train_set = data.train_set.sample(frac=1)
            data.validation_set = data.validation_set.sample(frac=1)

            for (i, sample) in enumerate(data.train_set):
                input_tensor = self.embedder.embed(sample)

                if i == 100000:
                    teacher_forcing_ratio = 0.25
                    encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=0.0015)

                if i == 200000:
                    encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=0.0005)

                loss, decoder_output = self.train_iteration(input_tensor, encoder_optimizer, decoder_optimizer,
                        criterion, teacher_forcing_ratio)
                print_loss_total += loss

                if i > 0 and i % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('{0:d} {1:d} {2:.10f}'.format(iteration, i, print_loss_avg))

                    print(' '.join(sample))
                    print(decoder_output)
                    print(' ', flush=True)
                    gc.collect()

                if i > 0 and i % validate_every == 0:
                    self.validate(data, criterion)
                    gc.collect()

                del input_tensor


            self.validate(data, criterion)

