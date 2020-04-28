import torch
import torch.nn.functional as F
import nltk
from torch.nn import CrossEntropyLoss
from torch import nn
from ppdb import ppdb as pp
import random
import hashlib

from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from torch.distributions.normal import Normal

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# ppdb = pp.load_ppdb('./ppdb-2.0-tldr', equiv_min=0.4)

VOCAB_SIZE = 50257
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


class GPT2Model1(GPT2Model):


    def __init__(self, config):
        super(GPT2Model1, self).__init__(config)


    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None, std=0.01, reset=True):
        self.noise = Normal(torch.tensor([0.0], requires_grad=False), torch.tensor([std], requires_grad=False))
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)

        # inputs_embeds.requires_grad = True
        position_embeds = self.wpe(position_ids)
        # position_embeds += self.noise.sample(position_embeds.size()).view_as(position_embeds).to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        if reset:
            hidden_states += self.noise.sample(hidden_states.size()).view_as(hidden_states).to(device)
        # past += self.noise.sample(past.size()).view_as(past).to(device)
        presents = []
        for block, layer_past in zip(self.h, past):
            # if layer_past is not None:
            #     layer_past += self.noise.sample(layer_past.size()).view_as(layer_past).to(device)
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2LMHeadModel1(GPT2LMHeadModel):


    def __init__(self, config):
        super(GPT2LMHeadModel1, self).__init__(config)
        self.transformer = GPT2Model1(config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None, std=0.01, reset=True):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past, std=std, reset=reset)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents

model = GPT2LMHeadModel1.from_pretrained('gpt2').to(device)
model.eval()

# for param in model.parameters():
#      param.requires_grad = True

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def predict(t):
    t = tokenizer.encode(t)
    input_tensor = torch.tensor([t]).to(device)

    out_token = ''
    out_str = ''

    l = 0
    while '.' not in out_token:
        # Convert inputs to PyTorch tensors
        predictions, past = model(input_tensor)

        # get the predicted last token
        idx = torch.argmax(predictions[0, -1, :]).item()
        out_token = tokenizer.decode([idx])
        out_str += out_token

        input_tensor = torch.cat((input_tensor, torch.tensor([[idx]]).to(device)), dim=1)
        print(out_token)

        l += 1

        if l == 40:
            print(past)
            return out_str
    print(past)
    return out_str

def backto(t, alpha=0.1):
    xe = torch.nn.CrossEntropyLoss().to(device)

    t = tokenizer.encode(t)
    input_tensor = torch.tensor([t]).to(device)

    original_input = input_tensor.clone()

    start = tokenizer.encoder['<|endoftext|>']
    context = torch.full((1, 1), start, device=device, dtype=torch.long, requires_grad=True)


    prev = context
    output = context
    past = None
    preds = []
    outstring = ''
    for i in range(len(t)):
        preds, past = model(output)

        logits = preds[:, -1, :] / 1
        logits = top_k_logits(logits, k=1)
        log_probs = F.softmax(logits, dim=-1)
        if False:
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)
        output = torch.cat((output, prev), dim=1)
        token = tokenizer.decode([int(prev)])
        outstring += token

    print(outstring)
# preds = torch.cat(tuple(preds), dim=1)
    l1 = xe(preds.contiguous().view(-1, preds.size(2)),
       original_input.contiguous().view(-1))
    l1 = torch.autograd.Variable(l1, requires_grad=True)

    l1.backward()

    return past
    # for p in past:
    #     if p.grad:
    #         print(p.grad)
    #     else:
    #         print("no grad")

    # new_past = []
    # for p in hids:
    #     if p.grad:
    #         print('g')
    #         new_past.append(p - alpha * p.grad)
    #     else:
    #         print('n')
    #         new_past.append(p )
    #
    # out_str = ''
    # for i in range(len(t)):
    #     predictions, past = model(input_tensor, past=new_past)
    #     idx = torch.argmax(predictions[0, -1, :]).item()
    #     out_token = tokenizer.decode([idx])
    #     out_str += out_token

    # return out_str
# backto('High on a hill was a lonely')


def check_bleu(output, targ):
    hypothesis = output.split()
    reference = targ.split()
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
    return BLEUscore

#the maximum is bigram, so assign the weight into 2 half.

def sample_sequence(ctx, targ, model, start_token=None, batch_size=1, context=None,
                    temperature=1.0, top_k=10, sample=True, reps=3, std=0.01):


    target = tokenizer.encode(ctx + targ)
    length = len(target)
    target = torch.tensor([target]).to(device)

    noise = Normal(torch.tensor([0.0], requires_grad=False), torch.tensor([1.0], requires_grad=False))

    uniques = set()


    with torch.no_grad():
        if start_token is None:
            assert context is not None, 'Specify exactly one of start_token and context!'
            context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        else:
            assert context is None, 'Specify exactly one of start_token and context!'
            context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

        prev = target.clone()
        output = ''
        past = None
        reset = True
        for i in range(500*10):
            logits, past = model(prev, past=past, std=std, reset=reset)
            reset = False

            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)

            log_probs = F.softmax(logits, dim=-1)
            # log_probs += noise.sample(log_probs.size()).view_as(log_probs)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)

            token = tokenizer.decode([int(prev)])
            if token != '<|endoftext|>':
                output += token
            if '\n' in token or len(output.split()) > 2 * len(targ.split()) or token=='<|endoftext|>':
                if ' '.join(output.split()) == targ:
                    temperature += 0.1
                    print(temperature)
                bleu = check_bleu(output, targ)
                # print('<s> ' + output.strip() + '</s> ' + str(bleu))
                if bleu > 0.2:
                    uniques.add(output.strip())
                    print("YAY" + output)
                output = ''
                prev = target.clone()
                past =None
                reset = True






    # xe = torch.nn.CrossEntropyLoss().to(device)
    # preds = torch.cat(tuple(preds), dim=0)
    #
    # l1 = xe(preds.contiguous().view(-1, preds.size(1)),
    #    target.contiguous().view(-1))
    # l1 = torch.autograd.Variable(l1, requires_grad=True)
    #
    # l1.backward()
    #
    # for p in past:
    #     if p.grad:
    #         print("HAS GRAD")
    #     else:
    #         print("NO GRAD :(")

    # print(out)
    for u in uniques:
        print(u)
    return uniques

def naive_paraphrase(s, ppdb, n=15, p=0.1):
    sents = [s]
    for i in range(n):
        new_sent = []
        for w in s.split():
            paras = ppdb.get_rhs(w)
            new_word = random.sample(paras, 1)[0][0] if paras and random.random() < p else w
            new_sent.append(new_word)
        sents.append(' '.join(new_sent))
    return  sents


def get_payload(t, n=4):
    payload = hashlib.sha256(str(t).encode()).hexdigest()[:2]
    payload = bin(int(payload,16))[2:].zfill(8)[:n]
    return int(payload, 2)

def get_paraphrases():
    paras = open('paraphrases.txt', 'r').readlines()
    paras = [p.split('\t') for p in paras]
    return paras

def get_random_context(n=4):
    context = ''
    for pp in random.sample(paras, n):
        new_ctx = context
        new_ctx += '---\n'
        new_ctx += pp[0].strip() + '\n'
        new_ctx += pp[1].strip() + '\n'
        if len(new_ctx) > 800:
            break
        else:
            context = new_ctx
    context += '---\n'
    return context


def get_data(model, out_file='gpt2', max_length=10000):
    for n in range(1,5):
        with open('{}_{}.txt'.format(out_file, n), 'w', buffering=1) as out:
            i = 0
            for t in open('para_test.txt', 'r'):
                t = t.split('\t')[0]
                payload = random.randint(0, 2**n-1)
                para = sample(t, model, start_token=tokenizer.encoder['<|endoftext|>'],
                              reps=1, std=0.0001,
                              n=n, payload=payload)
                out.write('{}\t{}\t{}\n'.format(i, para, get_payload(para, n)))
                print('{}\t{}\t{}'.format(t, para, get_payload(para, n)), flush=True)
                i += 1
                if i >= max_length:
                    break



def sample(targ, model, start_token=None, batch_size=1, context=None,
                    temperature=1.0, top_k=10, sample=True, reps=3, std=0.01, payload=None, n=4):

    noise = Normal(torch.tensor([0.0], requires_grad=False), torch.tensor([1.0], requires_grad=False))

    uniques = set()

    with torch.no_grad():
        for jj in range(1000):

            ctx = get_random_context(4)
            # print(ctx)
            target = tokenizer.encode(ctx + targ + '\n')
            length = len(target)
            target = torch.tensor([target]).to(device)

            if start_token is None:
                assert context is not None, 'Specify exactly one of start_token and context!'
                context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            else:
                # assert context is None, 'Specify exactly one of start_token and context!'
                context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

            prev = target.clone()
            output = ''
            past = None
            reset = True
            for i in range(len(targ.split()) * 2):
                logits, past = model(prev, past=past, std=std, reset=reset)
                reset = False

                logits = logits[:, -1, :] / temperature
                logits = top_k_logits(logits, k=top_k)
                logits[0][6329] = -float("inf")
                logits[0][12] = -float("inf")
                logits[0][438] = -float("inf")

                # log_probs = F.softmax(logits, dim=-1)
                # log_probs += noise.sample(log_probs.size()).view_as(log_probs)
                log_probs = F.softmax(logits, dim=-1)
                if sample:
                    prev = torch.multinomial(log_probs, num_samples=1)
                else:
                    _, prev = torch.topk(log_probs, k=1, dim=-1)

                token = tokenizer.decode([int(prev)])
                if token != '<|endoftext|>':
                    output += token
                if len(output.split()) > 2 * len(targ.split()):
                    output = ''
                    prev = target.clone()
                    past = None
                    reset = True

                if '\n' in token or token=='<|endoftext|>':
                    bleu = check_bleu(output, targ)
                    # print(output.strip())
                    # print('<s> ' + output.strip() + '</s> ' + str(bleu))
                    if ' '.join(output.split()) == targ:
                        temperature += 0.1
                    if len(targ.split()) <= 2 or bleu > 0.2:
                        print(output.strip(), get_payload(output.strip(), n=n))
                        uniques.add(output.strip())
                        if get_payload(output.strip(), n=n) == payload:
                            return output.strip()
                        break
                    # print(output.strip())
                    output = ''
                    prev = target.clone()
                    past =None
                    reset = True



def embed(t, p=None):
    return sample(t, model, start_token=tokenizer.encoder['<|endoftext|>'],
           reps=1, std=0.0001,
           n=4, payload=p)

# s = naive_paraphrase(s, ppdb, n=5)

# s = '\n'.join(s) + '\n'

# p = backto('hello my name is john')
paras = get_paraphrases()
with torch.no_grad():
    # s = sample_sequence(ctx, targ, model, start_token=tokenizer.encoder['<|endoftext|>'], reps=1, std=0.0001)
    # get_data(model)
    t = "It was a beautiful sunny day"
    embed(t,4)
