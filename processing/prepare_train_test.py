import glob
import os
import random
import argparse
import itertools
from typing import Tuple, List

import tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=3, help='the seed used for shuffling data')
parser.add_argument('--gestures_by_id_dir', type=str, help='the directory containing pure gestures data by video ids')
parser.add_argument('--words_by_id_dir', type=str, help='the directory containing pure words data by video ids')
parser.add_argument('--mixed_by_id_dir', type=str, help='the directory containing raw mixed (words and gestures) data by video ids')
parser.add_argument('--additional-task', type=str, choices=['compress-gestures', 'train-tokenizer', 'process-single-quotes'])


# Main task: Split to train and test files
def prepare_train_test(data_dir, output_dir, seed: int, test_ratio:float = 0.2,  shuffle_over='files'):
    data_files = [fname for fname in glob.glob(os.path.join(data_dir, '*.txt')) if ('train' not in fname and 'test' not in fname)]
    if shuffle_over == 'files':
        random.seed(seed)
        random.shuffle(data_files)
        all_filelines = list(map(_file_to_oneline, data_files))
        test_count = int(len(all_filelines) * test_ratio)
        test_data = all_filelines[:test_count]
        train_data = all_filelines[test_count:]
    elif shuffle_over == 'lines':
        all_lines = list(map(_file_to_multilines, data_files))
        all_lines = itertools.chain.from_iterable(all_lines) # List[List] => List[str]
        all_lines = list(all_lines)
        random.seed(seed)
        random.shuffle(all_lines)
        test_count = int(len(all_lines) * test_ratio)
        test_data = all_lines[:test_count]
        train_data = all_lines[test_count:]
    else:
        raise ValueError(f'Wrong parameter value for `shuffle_over`: {shuffle_over}, which has to be "files" or "lines"')
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for line in train_data:
            f.write(line + '\n')
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for line in test_data:
            f.write(line + '\n')

def _file_to_oneline(input_file) -> str:
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    line_str = ' '.join(lines)
    return line_str

def _file_to_multilines(input_file) -> List[str]:
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


# Additional task 1: compress-gestures
def compress_gestures(input_dir, output_dir):
    """
    input_dir: the directory containing the uncompressed gestures; can be the `output_dir` used by prepare_gesture()
        Assuming there are train.txt and test.txt existing in this directory
    output_dir: the directory to save the compressed gestures
    """
    input_file_train = os.path.join(input_dir, 'train.txt')
    input_file_test = os.path.join(input_dir, 'test.txt')
    assert os.path.exists(input_file_train)
    assert os.path.exists(input_file_test)

    output_file_train = os.path.join(output_dir, 'train.txt')
    output_file_test = os.path.join(output_dir, 'test.txt')
    compressed_lines_train = _compress_single_file(input_file_train)
    compressed_lines_test = _compress_single_file(input_file_test)
    with open(output_file_train, 'w') as f:
        for line in compressed_lines_train:
            f.write(line + '\n')
    with open(output_file_test, 'w') as f:
        for line in compressed_lines_test:
            f.write(line + '\n')

def _compress_single_file(input_file) -> List[str]:
    compressed_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            compressed_items = []
            for idx, item in enumerate(items):
                if idx == 0:
                    compressed_items.append(item)
                elif item == compressed_items[-1]:
                    continue
                else:
                    compressed_items.append(item)
            compressed_lines.append(' '.join(compressed_items))
    return compressed_lines


# Additional task 2: train-tokenizer
def train_wordlevel_tokenizer(input_path, output_file, special_tokens = None):
    """
    https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#training-the-tokenizer
    We can set the training arguments like vocab_size or min_frequency (here left at their default values of 30,000 and 0) but the most important part is to give the special_tokens we plan to use later on (they are not used at all during training) so that they get inserted in the vocabulary.

    The order in which you write the special tokens list matters: here "[UNK]" will get the ID 0, "[CLS]" will get the ID 1 and so forth.
    """
    tokenizer = tokenizers.Tokenizer(WordLevel(unk_token='[UNK]'))
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
        candidate_files = glob.glob(os.path.join(input_path, '*.txt'))
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


# Additional task 3: process-single-quotes
def handle_single_quotes(input_file, tokenizer: tokenizers.Tokenizer):
    processed = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            words_str, gestures_str = line.split('\t')
            flag, new_words, new_gestures = _handle_single_quotes_worker(words_str, gestures_str, tokenizer)
            if flag:
                processed.append(' '.join(new_words) + '\t' + ' '.join(new_gestures))
            else:
                processed.append(line)
    return processed

def _handle_single_quotes_worker(words_str: str, gestures_str: str, tokenizer: tokenizers.Tokenizer):
    words = words_str.split()
    gestures = gestures_str.split()
    try:
        assert len(words) == len(gestures)
    except AssertionError:
        print(words_str, len(words))
        print(gestures_str, len(gestures))
        if len(gestures) < len(words):
            gestures = [gestures[0]]*(len(words) - len(gestures)) + gestures # fix the shorter gesture seq
        else:
            raise
    words_tokenized = tokenizer.encode(words_str).tokens
    if len(words_tokenized) == len(gestures):
        return False, None, None
    else:
        new_words, new_gestures = [], []
        j = 0 # current index in words
        for i, w in enumerate(words_tokenized):
            new_words.append(w)
            new_gestures.append(gestures[j])
            if w == words[j]:
                j += 1
            elif i+1 == len(words_tokenized) or j+1 == len(words):
                break
            elif words_tokenized[i+1] == words[j+1]:
                j += 1
        try:
            assert len(new_words) == len(new_gestures)
        except AssertionError:
            print(f'words_str: {words_str}')
            print(f'gestures_str: {gestures_str}')
            raise
        return True, new_words, new_gestures


def main(args):
    gestures_by_id_dir = '../data/gestures_by_id'
    words_by_id_dir = '../data/words_by_id'
    mixed_by_id_dir = '../data/mixed_by_id'

    gesture_output_dir = '../data/gesture'
    word_output_dir = '../data/word'
    mixed_output_dir = '../data/mixed'
    for path in [gesture_output_dir, word_output_dir, mixed_output_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Conduct main task
    prepare_train_test(gestures_by_id_dir, gesture_output_dir, seed=args.seed, shuffle_over='files')
    prepare_train_test(words_by_id_dir, word_output_dir, seed=args.seed, shuffle_over='lines')
    prepare_train_test(mixed_by_id_dir, mixed_output_dir, seed=args.seed, shuffle_over='lines')

    # Conduct additional tasks
    if args.additional_task == 'compress-gestures':
        compressed_gesture_output_dir = '../data/gesture_compressed'
        if not os.path.exists(compressed_gesture_output_dir):
            os.makedirs(compressed_gesture_output_dir)
        compress_gestures(gesture_output_dir, compressed_gesture_output_dir)
    elif args.additional_task == 'train-tokenizer':
        tokenizer_output_file = os.path.join(word_output_dir, 'word_level_tokenizer.json')
        train_wordlevel_tokenizer(input_path=word_output_dir, output_file=tokenizer_output_file)
    elif args.additional_task == 'process-single-quotes':
        pass


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)