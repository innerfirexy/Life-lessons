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
    pass


def file_to_line(input_file) -> str:
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    gestures = ' '.join(lines)
    return gestures


def prepare_gesture(data_dir, output_dir, seed: int, test_ratio:float = 0.2):
    data_files = [fname for fname in glob.glob(os.path.join(data_dir, '*.txt')) if ('train' not in fname and 'test' not in fname)]
    random.seed(seed)
    random.shuffle(data_files)
    all_filelines = list(map(file_to_line, data_files))
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
    if not os.path.exists(gesture_output_dir):
        os.makedirs(gesture_output_dir)
    prepare_gesture(gesture_dir, gesture_output_dir, seed=args.seed)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)