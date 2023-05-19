import os
import numpy as np
import itertools
import random
import copy
import argparse
import time
import yaml

import torch
import math
import torch
from torch import nn, Tensor
import gensim.downloader

import utils
from models import RNN


###
# Arg parse
###
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help="path to config")
parser.add_argument('--data', type=str, default='data/gesture',
                    help='location of the data corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

# For experiments
parser.add_argument('--task', type=str, default='train', choices=['train', 'compute'])
parser.add_argument('--load', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--output_ppl', type=str, default='', help='path to save the perplexity (ppl) scores of test data')

###
# Init args
###
def init_args(args):
    config = args.config
    train_params = config['train_params']
    batch_size = train_params['batch_size']
    bptt = train_params['bptt']
    eol_id = train_params['eol_id']
    lr = train_params['lr']
    clip = train_params['clip']
    vars(args)['batch_size'] = batch_size
    vars(args)['bptt'] = bptt
    vars(args)['eol_id'] = eol_id
    vars(args)['lr'] = lr
    vars(args)['clip'] = clip

###
# Data
###
def data_process(raw_iter, pad_id=None):
    def _tokenize(input_str):
        return [int(t) for t in input_str.split()]
    # We need to think about the best way to insert dialog boundary to the data
    # Idea: add a special token "$" at the boundary, and change the shape of src_mask
    # so that the tokens after "$" will not be predicted.
    if pad_id:
        data = [torch.tensor(_tokenize(item) + [pad_id]) for item in raw_iter]
    else:
        data = [torch.tensor(_tokenize(item)) for item in raw_iter]

    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: torch.Tensor, bsz: int, device, batch_first=False) -> torch.Tensor:
    """
    :param data: the tensor returned by data_process, shape [N]
    :param bsz: batch size
    :return: Tensor of shape [bsz, N // bsz] if batch_first, otherwise, [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    if batch_first:
        data = data.view(bsz, seq_len).contiguous()
    else:
        data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(data: torch.Tensor, i: int, bptt: int, batch_first=False, flatten_target=True):
    """
    Generates a pair of source-target sequences for the model.
    t subdivides the batchified data into chunks of length bptt
    :param data: Tensor, shape [bsz, L] or [L, bsz] (indicated by batch_first), which is the output of batchify()
    :param i:
    :param bptt:
    :param batch_first: bool
    :return:
    """
    seq_len = min(bptt, len(data) - 1 - i)
    if batch_first:
        source = data[:, i: i+seq_len]
        target = data[:, i+1: i+1+seq_len]
    else:
        source = data[i: i+seq_len]
        target = data[i+1: i+1+seq_len]
    if flatten_target:
        target = target.reshape(-1)
    return source, target

def load_data(args):
    train_file = os.path.join(args.data, 'train.txt')
    test_file = os.path.join(args.data, 'test.txt')
    train_iter = utils.MyTextIterableDataset(train_file)
    test_iter = utils.MyTextIterableDataset(test_file)
    train_data = data_process(train_iter, pad_id=args.eol_id)
    test_data = data_process(test_iter, pad_id=args.eol_id)
    train_loader = batchify(train_data, bsz=args.batch_size, device=args.device)
    test_loader = batchify(test_data, bsz=args.batch_size, device=args.device)
    return train_loader, test_loader

###
# Model
###
def init_model(args):
    config = args.config
    model_params = config['model_params']
    model_type = model_params['type']
    emsize = model_params['emsize']
    ntokens = model_params['ntokens']
    nhid = model_params['nhid']
    nlayers = model_params['nlayers']
    dropout = model_params['dropout']
    pretrained_embeds = model_params['pretrained_embeds']
    tokenizer = model_params['tokenizer']
    vars(args)['ntokens'] = ntokens
    if pretrained_embeds:
        embed = load_pretrained_embed(emsize, tokenizer)
        vars(args)['pretrained_embeds'] = embed
    else:
        vars(args)['pretrained_embeds'] = None

    model = RNN(model_type, ntokens + 1, emsize, nhid, nlayers, dropout, args).to(args.device) # the added token is for end-of-line (EOL)
    return model

def repackage_hidden(h, device):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach().to(device)
    else:
        return tuple(repackage_hidden(v, device) for v in h)

def load_pretrained_embed(emsize, tokenizer):
    # tt_embed = torchtext.vocab.GloVe(name="6B", dim=emsize)
    wv = gensim.downloader.load('word2vec-google-news-300')
    vocab = tokenizer.get_vocab()
    embed = np.random.randn(len(vocab), emsize)
    for word in vocab:
        if word in wv:
            w_id = vocab[word]
            embed[w_id, :] = wv[word]
    return torch.from_numpy(embed)


###
# Train and evaluate
###
def train(model, train_loader, criterion, optimizer, scheduler, epoch, args) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 50
    start_time = time.time()

    bptt = args.bptt
    ntokens = args.ntokens + 1 # the added token is for end-of-line (EOL)

    hidden = model.init_hidden(args.batch_size)
    num_batches = len(train_loader) // bptt
    for batch, i in enumerate(range(0, train_loader.size(0) - 1, bptt)):
        data, targets = get_batch(train_loader, i, bptt=bptt)
        hidden = repackage_hidden(hidden, args.device)
        try:
            output, hidden = model(data, hidden)
        except Exception:
            print('batch:', batch)
            raise

        loss = criterion(output.view(-1, ntokens), targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

@torch.no_grad()
def evaluate(model, eval_loader, criterion, args):
    model.eval()  # turn on evaluation mode
    bptt = args.bptt
    ntokens = args.ntokens + 1 # the added token is for end-of-line (EOL)
    total_loss = 0.

    hidden = model.init_hidden(args.batch_size)
    for i in range(0, eval_loader.size(0) - 1, bptt):
        data, targets = get_batch(eval_loader, i, bptt=bptt)
        seq_len = data.size(0)
        hidden = repackage_hidden(hidden, args.device)
        try:
            output, hidden = model(data, hidden)
        except Exception:
            print('data:', data)
            raise
        output_flat = output.view(-1, ntokens)
        total_loss += seq_len * criterion(output_flat, targets).item()

    return total_loss / (len(eval_loader) - 1)


###
# Compute perplexity scores, using a trained model
###
@torch.no_grad()
def ppl_by_line(model, data_loader, criterion, args, output_file=None):
    model.eval()
    bptt = args.bptt
    ppls = []
    hidden = model.init_hidden(args.batch_size)
    for i in range(0, data_loader.size(0) - 1, bptt):
        data, targets = get_batch(data_loader, i, bptt=bptt, flatten_target=False) # Do not flatten target, so that the targets tensor is in shape [N, C]
        hidden = repackage_hidden(hidden, device)
        try:
            output, hidden = model(data, hidden)
        except Exception:
            print('data:', data)
            raise
        # output is of shape [L,N,C], where N is the batch size. N=1 is expected, but if not use a for loop to iterate over N
        N = output.size(1)
        if N == 1:
            output = output.squeeze() # [L,1,C] -> [L,C]
            targets = targets.squeeze() # [L,1] -> [L]
            loss = criterion(output, targets)
            ppl = math.exp(loss.item())
            ppls.append(ppl)
            if args.eol_id in data:
                ppls.append('\n')
            if not args.output_ppl:
                print(ppl, end=' ')
        else:
            for i in range(N):
                o = output[:,i,:]
                t = targets[:,i]
                loss = criterion(o, t)
                ppl = math.exp(loss.item())
                ppls.append(ppl)
                if args.eol_id in data[:,i]:
                    ppls.append('\n')
                if not args.output_ppl:
                    print(str(ppl), end=' ')

    if output_file:
        with open(output_file, 'w') as f:
            line_count = 0
            position = 0
            for i, item in enumerate(ppls):
                if item == '\n':
                    line_count += 1
                    position = 0
                else:
                    f.write(f'{line_count},{item},{position}\n')
                    position += 1
    # print(ppls)

def compute(args):
    init_args(args)
    vars(args)['batch_size'] = 1 # For compute task, set batch_size=1, so that the begin->end order is guaranteed
    train_loader, test_loader = load_data(args)
    criterion = nn.CrossEntropyLoss()
    with open(args.load, 'rb') as f:
        model = torch.load(f, map_location=args.device)
        print('model loaded')
    if args.output_ppl:
        ppl_by_line(model, train_loader, criterion, args, args.output_ppl + '_train.txt')
        ppl_by_line(model, test_loader, criterion, args, args.output_ppl + '_test.txt')
    else:
        ppl_by_line(model, train_loader, criterion, args)
        ppl_by_line(model, test_loader, criterion, args)


###
# Main
###
def main(args):
    init_args(args)
    model = init_model(args)
    train_loader, test_loader = load_data(args)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_model = None

    config = args.config
    train_params = config['train_params']
    epochs = train_params['epochs']

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model, train_loader, criterion, optimizer, scheduler, epoch, args)
        val_loss = evaluate(model, test_loader, criterion, args)
        val_ppl = math.exp(val_loss)

        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        scheduler.step()

    # Save model
    data_name = os.path.basename(os.path.normpath(args.data)) # normpath removes the "/" in path if any
    print(f'Save the model to: best_{args.config["name"]}_{data_name}.pt')
    torch.save(best_model, f'best_{args.config["name"]}_{data_name}.pt')


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    vars(args)['config'] = config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vars(args)['device'] = device

    if args.task == 'train':
        main(args)
    elif args.task == 'compute':
        compute(args)