# from main import repackage_hidden
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tokenizers import Tokenizer
import gensim.downloader
from utils import Corpus
from models import MixedRNN

import argparse
import yaml
import os
import time
import math
import copy
import numpy as np
from typing import List


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help="path to config")
parser.add_argument('--data', type=str, default='data/mix',
                    help='location of the data corpus')
parser.add_argument('--tokenizer_file', type=str, default='',
                    help='pretrained tokenizer')
parser.add_argument('--tied', action='store_true', default=False)

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--log_file', type=str, default='', help='output file to write log information')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--mix_type', type=str, choices=['sum', 'concat', 'bilinear'], default='sum',
                    help='type')
parser.add_argument('--pred_task', type=str, choices=['word', 'gesture'], default='',
                    help='prediction task')

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
    lr = train_params['lr']
    clip = train_params['clip']

    vars(args)['batch_size'] = batch_size
    vars(args)['lr'] = lr
    vars(args)['clip'] = clip

###
# Load data (tokenizer initialization is included)
###
def load_data(args):
    corpus = Corpus(args.data)
    train_data, val_data, test_data = corpus.get_data()

    if args.tokenizer_file:
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
    else:
        tokenizer_path = os.path.join(args.data, 'word_level_tokenizer.json')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f'Tokenizer file {tokenizer_path} not found. Please supply one.')
        tokenizer = Tokenizer.from_file(tokenizer_path)

    tokenizer.enable_padding(pad_id=1, pad_token="[PAD]")
    vars(args)['tokenizer'] = tokenizer
    vars(args)['w_pad_id'] = 1
    w_ntokens = tokenizer.get_vocab_size() # [PAD] is already included in vocab
    vars(args)['w_ntokens'] = w_ntokens

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b, tokenizer=tokenizer: collate_batch_pair(b, args)) # shuffle=False will cause error
    test_loader = DataLoader(test_data, batch_size=40, shuffle=False,
                             collate_fn=lambda b, tokenizer=args.tokenizer: collate_batch_pair(b, args))
    val_loader = None
    if val_data:
        val_loader = DataLoader(val_data, batch_size=40, shuffle=False,
            collate_fn=lambda b, tokenizer=tokenizer: collate_batch_pair(b, args))

    return train_loader, val_loader, test_loader

def _tokenize_words(batch: List[str], tokenizer):
    batch = [item.strip() for item in batch]
    encoded_results = tokenizer.encode_batch(batch)
    ids_list, attn_mask_list, lengths_list = [], [], []
    for res in encoded_results:
        ids_list.append(res.ids)
        attn_mask_list.append(res.attention_mask)
        lengths_list.append(res.attention_mask.count(1)) # mask is of 1s and 0s, the count of 1s is the actual length

    ids = torch.tensor(ids_list, dtype=torch.int64).permute(1,0) # transpose batch to dim=1
    attention_mask = torch.tensor(attn_mask_list, dtype=torch.int64).permute(1,0) # Same as ids
    lengths = torch.tensor(lengths_list, dtype=torch.int64)

    return ids, lengths, attention_mask

def _tokenize_gestures(batch: List[str], pad):
    batch = [item.strip() for item in batch]
    ids_list: List[torch.Tensor] = []
    lengths = []
    for b in batch:
        ids = list(map(int, b.split()))
        lengths.append(len(ids))
        ids = torch.tensor(ids, dtype=torch.int64)
        ids_list.append(ids)
    # Pad ids_list
    ids_padded = pad_sequence(ids_list, padding_value=pad) # default output for pad_sequence is batch_size at dim=1, i.e., batch_first=False
    lengths = torch.tensor(lengths, dtype=torch.int64)

    return ids_padded, lengths

def collate_batch_pair(batch, args):
    """
    PyTorch tutorial for writing custom code for seq2seq NLP tasks:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    """
    tokenizer = args.tokenizer
    g_pad_id = args.g_pad_id # 82
    seq1, seq2 = zip(*[item.strip().split('\t') for item in batch])
    seq1_ids, seq1_lengths, _ = _tokenize_words(seq1, tokenizer)
    seq2_ids, seq2_lengths = _tokenize_gestures(seq2, pad = g_pad_id)

    return seq1_ids, seq1_lengths, seq2_ids, seq2_lengths


###
# Model
###
def init_model(args):
    config = args.config
    model_params = config['model_params']
    model_type = model_params['type']
    w_emsize = model_params['w_emsize']
    nhid = model_params['nhid']
    nlayers = model_params['nlayers']
    dropout = model_params['dropout']
    vars(args)['model_type'] = model_type
    vars(args)['w_emsize'] = w_emsize
    vars(args)['nhid'] = nhid
    vars(args)['nlayers'] = nlayers
    vars(args)['dropout'] = dropout

    # get vocab_size from tokenizer, including the padding token
    tokenizer = args.tokenizer # print(tokenizer.id_to_token(1)) # 1 maps [PAD], so using pad_id=1 is okay

    pretrained_embeds = model_params['pretrained_embeds']
    if pretrained_embeds:
        embed = load_pretrained_embed(w_emsize, tokenizer)
        vars(args)['pretrained_embeds'] = embed
    else:
        vars(args)['pretrained_embeds'] = None

    # Gesture params
    g_ntokens = model_params['g_ntokens']
    g_emsize = model_params['g_emsize']
    mix_emsize = model_params['mix_emsize']
    pred_task = model_params['pred_task']
    vars(args)['g_ntokens'] = g_ntokens
    vars(args)['g_pad_id'] = g_ntokens
    vars(args)['g_emsize'] = g_emsize
    vars(args)['mix_emsize'] = mix_emsize
    vars(args)['pred_task'] = pred_task

    # Model
    model = MixedRNN(args).to(args.device)
    return model


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


def get_targets(ids, pad_id = 1, flat_target = True):
    """
    params:
        flat_target: True for training, False for computing perplexity per sentence
    returns:
        targets
    """
    # target is the same as the input_ids
    pads = torch.full((ids.shape[0], 1), pad_id)
    ids_shifted = torch.cat((ids, pads), dim=1) # Use torch.cat to add a column of all pad_id to the right of ids
    targets = ids_shifted[:, 1:]
    if flat_target:
        targets = targets.reshape(-1) # targets () is a flat tensor

    return targets

def repackage_hidden(h, device):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach().to(device)
    else:
        return tuple(repackage_hidden(v, device) for v in h)


###
# Training 
###
def train(model, train_loader, criterion, optimizer, scheduler, epoch, args):
    model.train()
    total_loss = 0.
    start_time = time.time()

    hidden = model.init_hidden(args.batch_size)
    for step, batch in enumerate(train_loader):
        words, words_len, gestures, gestures_len = batch
        targets = get_targets(words, pad_id=args.w_pad_id)
        words = words.to(args.device)
        gestures = torch.ones(gestures.shape).long()
        gestures = gestures.to(args.device)
        targets = targets.to(args.device)

        optimizer.zero_grad()
        actual_bsz = words.shape[1]
        if actual_bsz != args.batch_size: # For the last batch's actual size is not necessarily args.batch_size
            hidden = model.init_hidden(actual_bsz)
            # NOTE: precisely, we should not initialize a brand new hidden state,
            # but instead, should take a proportion of the hidden at previous step
        hidden = repackage_hidden(hidden, args.device)
        # Forward
        try:
            output, hidden = model(words, gestures, words_len, hidden)
        except Exception:
            # Print debug info
            print('words.shape:', words.shape)
            print('gestures.shape: ', gestures.shape)
            print('words_len.shape:', words_len.shape)
            if isinstance(hidden, tuple):
                print(hidden[0].shape)
                print(hidden[1].shape)
            raise
        output_flat = output.view(-1, args.w_ntokens) # In order to match the size of `targets`, which is also a flat tensor
        # Backward
        try:
            loss = criterion(output_flat, targets)
            loss.backward()
        except RuntimeError:
            # Print debug info
            print(output.shape)
            print('target shape: ', targets.shape)
            raise
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        optimizer.step()

        total_loss += loss.item()
        if step % args.log_interval == 0 and step > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {step:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


###
# Evaluate 
###
@torch.no_grad()
def evaluate(model, eval_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(args.batch_size)

    for batch in eval_loader:
        words, words_len, gestures, gestures_len = batch
        targets = get_targets(words, pad_id=args.w_pad_id)
        words = words.to(args.device)
        gestures = torch.ones(gestures.shape).long()
        gestures = gestures.to(args.device)
        targets = targets.to(args.device)

        actual_bsz = words.shape[1]
        if actual_bsz != args.batch_size: # For the last iter in an epoch, the actual size of data is not necessarily args.batch_size
            hidden = model.init_hidden(actual_bsz)
        hidden = repackage_hidden(hidden, args.device)
        output, hidden = model(words, gestures, words_len, hidden)

        loss = criterion(output.view(-1, args.w_ntokens), targets)
        total_loss += loss.item()

    total_loss /= len(eval_loader)
    return total_loss


###
# A standalone function for computing and outputing perplexity scores
###
@torch.no_grad()
def ppl_by_sentence(model, data_loader, criterion, args, output_file=None):
    model.eval()
    fwriter = None
    if args.output_ppl:
        fwriter = open(args.output_ppl, 'w')

    hidden = model.init_hidden(args.batch_size)
    for batch in data_loader:
        data, data_lengths, targets = get_targets(batch, flat_target=False) # should not flat the output
        data = data.to(device)
        targets = targets.to(device)

        if data.shape[0] != args.batch_size: # For the last iter in an epoch, the actual size of data is not necessarily args.batch_size
            hidden = model.init_hidden(data.shape[0])
        hidden = repackage_hidden(hidden, device)
        output, hidden = model(data, data_lengths, hidden)
    
        if args.output_ppl:
            for i in range(output.shape[0]):
                o = output[i]
                t = targets[i]
                ppl = criterion(o, t) # 
                fwriter.write(str(ppl.item()))
                fwriter.write('\n')
    if fwriter:
        fwriter.close()


def main(args):
    init_args(args)
    train_loader, val_loader, test_loader = load_data(args)
    if val_loader is None:
        val_loader = test_loader
    model = init_model(args) # model comes after data, because model depends on the tokenizer loaded together with data

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=args.w_pad_id) # ignore the [pad] token
    best_val_loss = float('inf')
    best_model = None

    config = args.config
    train_params = config['train_params']
    epochs = train_params['epochs']
    log_writer = None
    if args.log_file:
        print(f'writing log to {args.log_file}')
        log_writer = open(args.log_file, 'w')
        log_writer.write('epoch,loss\n')

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
        if log_writer:
            log_writer.write(f'{epoch}, {val_loss:.2f}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        scheduler.step()

    # Save model
    if log_writer:
        log_writer.close()
    data_name = os.path.basename(os.path.normpath(args.data)) # normpath removes the "/" in path if any
    print(f'Save the model to: best_{args.config["name"]}_{data_name}.pt')
    torch.save(best_model, f'best_{args.config["name"]}_{data_name}.pt')


def compute(args):
    init_args(args)
    vars(args)['batch_size'] = 1 # For compute task, set batch_size=1, so that the begin->end order is guaranteed
    train_loader, test_loader = load_data(args)
    criterion = nn.CrossEntropyLoss(ignore_index=args.w_pad_id)
    with open(args.load, 'rb') as f:
        model = torch.load(f, map_location=args.device)
        model.rnn.flatten_parameters()
    print('model loaded')

    ppl_by_sentence(model, test_loader, criterion)
    # evaluate(model, test_loader, criterion)


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
