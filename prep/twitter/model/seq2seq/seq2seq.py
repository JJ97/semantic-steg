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
        self.encoder = Encoder(embedder.input_size, hidden_size=256, device=device).to(device)
        self.decoder = Decoder(embedder.input_size, hidden_size=256, device=device,
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


    def train_iteration(self, input_tensor, optimizer, criterion, teacher_forcing_ratio):
        encoder_hidden = self.encoder.init_hidden()

        optimizer.zero_grad()

        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=self.device)

        for ei in range(input_tensor.size(0)):
            encoder_output, encoder_hidden = self.encoder.forward(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        use_teacher_forcing = random.random() < teacher_forcing_ratio

        decoder_outputs, loss = self.get_decoder_output(input_tensor, encoder_outputs, encoder_hidden,
                                                        criterion, use_teacher_forcing)

        loss.backward()

        optimizer.step()

        avg_loss = float(loss.item()) / input_tensor.size(0)
        out = self.output2tweet(decoder_outputs)

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
        avg_loss = float(loss.item()) / input_tensor.size(0)

        del input_tensor
        del loss
        del decoder_outputs
        del encoder_outputs

        return out, avg_loss
        
    def output2tweet(self, decoder_outputs):
        decoded_words = []
        for decoder_output in decoder_outputs:
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(topi.item())
            if topi.item() == self.embedder.end_tensor:
                break
        return self.embedder.unembed(decoded_words)


    def validate(self, data, criterion, print_every):
        print("VALIDATING")

        total_validation_loss = 0

        with torch.no_grad():
            for (i, sample) in enumerate(data.validation_set):
                input_tensor = self.embedder.embed(sample)
                output_sentence, loss = self.validation_iteration(input_tensor, criterion)
                total_validation_loss += loss

                if i > 0 and i % print_every == 0:
                    print('>', data.get_printable_sample(sample))
                    print('<', output_sentence)
                    print('loss ', loss)
                    print(' ', flush=True)

        print("AVERAGE VALIDATION LOSS: {}".format(total_validation_loss / data.validation_set.size))



    def train(self, data, iterations, print_every=500, validate_every=50000,
              learning_rate=0.0001, teacher_forcing_ratio=0.5):
        criterion = nn.NLLLoss()

        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                       lr=learning_rate)

        print_loss_total = 0  # Reset every print_every
        for iteration in range(iterations):

            data.train_set = data.train_set.sample(frac=1)
            data.validation_set = data.validation_set.sample(frac=1)

            for (i, sample) in enumerate(data.train_set):
                input_tensor = self.embedder.embed(sample)

                loss, decoder_output = self.train_iteration(input_tensor, optimizer, criterion, teacher_forcing_ratio)
                print_loss_total += loss

                if i > 0 and i % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('{0:d} {1:d} {2:.10f}'.format(iteration, i, print_loss_avg))

                    print(data.get_printable_sample(sample))
                    print(decoder_output)
                    print(' ', flush=True)

                if i > 0 and i % validate_every == 0:
                    self.validate(data, criterion, print_every)



            self.validate(data, criterion)

