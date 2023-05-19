import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import gensim.downloader
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from utils import Corpus
from models import MixedTransformerModel, generate_square_subsequent_mask

import argparse
import os
import time
import yaml
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

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--log_file', type=str, default='', help='output file to write log information')
parser.add_argument('--save', type=str, default='',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--mix_type', type=str, choices=['sum', 'concat', 'bilinear'], default='sum',
                    help='type')
parser.add_argument('--pred_task', type=str, choices=['word', 'gesture'], default='',
                    help='prediction task')

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
    lr = train_params['lr']
    clip = train_params['clip']
    vars(args)['batch_size'] = batch_size
    vars(args)['lr'] = lr
    vars(args)['clip'] = clip
    print('init args success!')

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
    w_ntokens = tokenizer.get_vocab_size()
    vars(args)['w_ntokens'] = w_ntokens

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b, args=args: collate_batch_pair(b, args)) # shuffle=False will cause error
    test_loader = DataLoader(test_data, batch_size=40, shuffle=False,
                             collate_fn=lambda b, args=args: collate_batch_pair(b, args))
    val_loader = None
    if val_data:
        val_loader = DataLoader(val_data, batch_size=40, shuffle=False,
                                collate_fn=lambda b, args=args: collate_batch_pair(b, args))
    print('load data success!')
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
    attention_mask = torch.tensor(attn_mask_list, dtype=torch.int64) # Do not use .permute(1,0)
    # because the attention_mask will be used as src_key_padding_mask in torch.nn.transformer,
    # whose dim should be (N, S), i.e., batch_size is the first dim
    # Convert to key_pad_mask by flipping: 0 -> True, 1 -> False
    pad_mask = attention_mask == 0
    lengths = torch.tensor(lengths_list, dtype=torch.int64)

    return ids, lengths, pad_mask

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
    seq1_ids, seq1_lengths, pad_mask = _tokenize_words(seq1, tokenizer)
    seq2_ids, seq2_lengths = _tokenize_gestures(seq2, pad = g_pad_id)

    return seq1_ids, seq1_lengths, pad_mask, seq2_ids

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

###
# Model
###
def init_model(args):
    config = args.config
    model_params = config['model_params']
    w_emsize = model_params['w_emsize']
    d_model = model_params['d_model']
    d_hid = model_params['d_hid']
    nhead = model_params['nhead']
    nlayers = model_params['nlayers']
    dropout = model_params['dropout']
    vars(args)['w_emsize'] = w_emsize
    vars(args)['d_model'] = d_model
    vars(args)['d_hid'] = d_hid
    vars(args)['nhead'] = nhead
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

    model = MixedTransformerModel(args).to(args.device)
    print('init model success!')
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


###
# Training
###
def train(model, train_loader, criterion, optimizer, scheduler, epoch, args):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        words, _, words_pad_mask, gestures = batch
        targets = get_targets(words, pad_id=args.w_pad_id)
        words = words.to(args.device)
        words_pad_mask = words_pad_mask.to(args.device)
        targets = targets.to(args.device)
        seq_len = words.size(0)
        words_attn_mask = generate_square_subsequent_mask(seq_len).to(args.device)
        gestures = gestures.to(args.device)

        optimizer.zero_grad()
        try:
            output = model(words, words_attn_mask, words_pad_mask, gestures)
        except RuntimeError:
            raise

        output_flat = output.view(-1, args.w_ntokens) # In order to match the size of `targets`, which is also a flat tensor
        try:
            loss = criterion(output_flat, targets)
            loss.backward()
        except RuntimeError:
            raise
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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
    # NOTE: about how to add src_key_padding_mask
    # https://zhuanlan.zhihu.com/p/353365423
    # https://huggingface.co/docs/transformers/preprocessing#pad
    # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer\

    for batch in eval_loader:
        words, _, words_pad_mask, gestures = batch
        targets = get_targets(words, pad_id=args.w_pad_id)
        words = words.to(args.device)
        words_pad_mask = words_pad_mask.to(args.device)
        targets = targets.to(args.device)
        seq_len = words.size(0)
        words_attn_mask = generate_square_subsequent_mask(seq_len).to(args.device)
        gestures = gestures.to(args.device)

        output = model(words, words_attn_mask, words_pad_mask, gestures)
        loss = criterion(output.view(-1, args.w_ntokens), targets)
        total_loss += loss.item()

    total_loss /= len(eval_loader)
    return total_loss


###
# A standalone function for computing and outputing perplexity scores
###
@torch.no_grad()
def ppl_by_sentence(model, data_loader):
    model.eval()
    fwriter = None
    if args.output_ppl:
        fwriter = open(args.output_ppl, 'w')

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch in data_loader:
        pass
    if fwriter:
        fwriter.close()


def main(args):
    init_args(args)
    train_loader, val_loader, test_loader = load_data(args)
    if val_loader is None:
        val_loader = test_loader
    model = init_model(args) # model comes after data, because model depends on the tokenizer loaded together with data

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
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


def compute():
    init_args(args)
    vars(args)['batch_size'] = 1 # For compute task, set batch_size=1, so that the begin->end order is guaranteed
    train_loader, test_loader = load_data(args)
    criterion = nn.CrossEntropyLoss(ignore_index=args.w_pad_id)
    with open(args.load, 'rb') as f:
        model = torch.load(f, map_location=args.device)
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

    print(f'Task: {args.task}')
    if args.task == 'train':
        main(args)
    elif args.task == 'compute':
        compute(args)