import glob
import os
import random
import re
import argparse
from typing import Tuple, List
# from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=3, help='the seed used for shuffling data')


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


def _file_to_line(input_file) -> str:
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    gestures = ' '.join(lines)
    return gestures


def prepare_train_test(data_dir, output_dir, seed: int, test_ratio:float = 0.2):
    data_files = [fname for fname in glob.glob(os.path.join(data_dir, '*.txt')) if ('train' not in fname and 'test' not in fname)]
    random.seed(seed)
    random.shuffle(data_files)
    all_filelines = list(map(_file_to_line, data_files))
    test_count = int(len(all_filelines) * test_ratio)
    test_data = all_filelines[:test_count]
    train_data = all_filelines[test_count:]

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for line in train_data:
            f.write(line + '\n')
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for line in test_data:
            f.write(line + '\n')


def main(args):
    gesture_dir = '../data/gestures_by_id'
    gesture_output_dir = '../data/gesture'
    compressed_gesture_output_dir = '../data/gesture_compressed'
    if not os.path.exists(gesture_output_dir):
        os.makedirs(gesture_output_dir)
    if not os.path.exists(compressed_gesture_output_dir):
        os.makedirs(compressed_gesture_output_dir)
    prepare_train_test(gesture_dir, gesture_output_dir, seed=args.seed)
    compress_gestures(gesture_output_dir, compressed_gesture_output_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)