import tokenizers
import os
import sys
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/word',
                    help='location of the data corpus')
parser.add_argument('--output', type=str, default='word_level_tokenizer.json')



def train_wordlevel_tokenizer(input_path, output_file, prefix = '', special_tokens = None):
    """
    https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#training-the-tokenizer
    We can set the training arguments like vocab_size or min_frequency (here left at their default values of 30,000 and 0) but the most important part is to give the special_tokens we plan to use later on (they are not used at all during training) so that they get inserted in the vocabulary.

    The order in which you write the special tokens list matters: here "[UNK]" will get the ID 0, "[CLS]" will get the ID 1 and so forth.
    """
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) #TODO: not sure if this unk_token matters
    tokenizer.pre_tokenizer = Whitespace()

    if not special_tokens:
        special_tokens = []
    default_special_tokens = ['[UNK]', '[PAD]']
    for tok in default_special_tokens:
        if tok not in special_tokens:
            special_tokens.append(tok)
    trainer = WordLevelTrainer(special_tokens=special_tokens)

    # Check if the corpus data exist
    if os.path.isdir(input_path):
        candidate_files = [os.path.join(input_path, f'{prefix}train.txt') for split in ['train', 'test', 'valid']]
        files = []
        for file in candidate_files:
            if not os.path.exists(file):
                print(f'{file} does not exist!')
            else:
                files.append(file)
    elif os.path.isfile(input_path):
        if not os.path.exists(input_path):
            print(f'{input_path} does not exist!')
            print('no training is done.')
            return
        else:
            files = [input_path]

    tokenizer.train(files, trainer)
    tokenizer.save(output_file)


def test_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file('word-level-tokenizer-wiki2.json')

    # Single sentence
    output = tokenizer.encode("Hello, y'all!")
    print(output.tokens)

    # Batch of sentences
    tokenizer.enable_padding(pad_id=1, pad_token="[PAD]")
    outputs = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
    print(outputs[0].tokens)
    print(outputs[1].tokens)


def check_special_tokens(input_path):
    special_tokens = Counter()
    with open(input_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            for tok in tokens:
                if tok.startswith('<') and tok.endswith('>'):
                    special_tokens[tok] += 1
    print('==============')
    print(f'{len(special_tokens)} Special tokens detected:')
    for k, v in special_tokens.most_common():
        print(f'{k}: {v}')
    print('==============')
    for k, _ in special_tokens.most_common():
        sys.stdout.write(f'\'{k}\', ')
        sys.stdout.flush()


def main(args):
    # train_wordlevel_tokenizer('word-level-tokenizer-wiki2.json')
    # train_wordlevel_tokenizer(input_path='.data/WikiText2/wikitext-2/', output_file='word-level-tokenizer-wiki2_pad.json', special_tokens=['[UNK]', '[PAD]']) 
    # NOTE: the token in original corpus data is <unk>. If we use [UNK] as special_token, the resulting .json dictionary file still contains <unk> rather than [UNK]

    # check_special_tokens(input_path='data/ytb1_label.txt')
    # train_wordlevel_tokenizer(input_path='data/ytb1_label.txt', output_file='ytb1_label_pad.json', special_tokens=['<a>', '<b>', '<c>', '<d>', '<e>', '<f>', '<g>', '<h>'])

    # check_special_tokens(input_path='data/cleaned_data_final.txt')
    # train_wordlevel_tokenizer(input_path='data/position/cleaned_data_final.txt',
    #                           output_file='data/position/cleaned_data_final_pad.json',
    #                           special_tokens=['<70>', '<69>', '<43>', '<42>', '<40>', '<78>', '<39>',\
    #                                           '<52>', '<51>', '<67>', '<79>', '<66>', '<48>', '<60>',\
    #                                           '<61>', '<72>', '<129>'])
    input_path = args.data
    output_file = args.output
    train_wordlevel_tokenizer(input_path, output_file)

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    main(args)