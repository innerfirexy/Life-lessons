import re
import webvtt
import glob
import os
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import itertools
import argparse
from collections import Counter
from typing import Tuple, Union, List


parser = argparse.ArgumentParser()
parser.add_argument('--input_gesture_dir', type=str, help='Input directory that contains gesture token files, which can be either .pkl or .csv')
parser.add_argument('--input_vtt_dir', type=str, help='Input directory that contains the automatically generated subtitle files in .vtt format')
parser.add_argument('--output_word_dir', type=str, help='Output directory to save the extracted word data file')
parser.add_argument('--output_gesture_dir', type=str, help='Output directory to save the extracted gesture data file')
parser.add_argument('--output_mixed_dir', type=str, help='Output directory to save the extracted mixed data file')
parser.add_argument('--input_gesture_file', type=str, help='An input file that contains gesture token files, .pkl or .csv')
parser.add_argument('--input_vtt_file', type=str, help='An input file that contains the automatically generated subtitle files in .vtt format')


# Interpolate missing frames with None
def interpolate_gesture_tokens(gesture_file, unk=0) -> List[int]:
    """
    gesture_file: A file containing the tokenized gestures obtained from running tokenize_gest_grid.py
        Can be either a .pkl or .csv file
    """
    df = pd.read_csv(gesture_file)
    labels = df['label'].tolist()
    frame_indices = df['frame_idx'].tolist()
    min_frame_idx = 0
    max_frame_idx = frame_indices[-1]
    frame_indices = set(frame_indices)
    full_frame_indices = list(range(min_frame_idx, max_frame_idx+1))

    interpolated_tokens = []
    i = 0
    for frame_idx in full_frame_indices:
        if frame_idx in frame_indices:
            interpolated_tokens.append(labels[i])
            i += 1
        else:
            interpolated_tokens.append(unk)
    try:
        assert len(interpolated_tokens) == len(full_frame_indices)
    except AssertionError:
        print(f'problem in {gesture_file}')
        raise
    return interpolated_tokens 


def get_token_by_timespan(start_time: Union[float, str], end_time: Union[float, str], interpolated_tokens, unk = 0, fps: int = 24):
    """
    start_time: float, starting time of a token in seconds
    end_time: float, ending time of a token
    labels_map: list[int], a list of interpolated labels
    """
    def str_to_float(time_str: str):
        ms = datetime.strptime(time_str, "%H:%M:%S.%f").microsecond
        s = datetime.strptime(time_str, "%H:%M:%S.%f").second
        m = datetime.strptime(time_str, "%H:%M:%S.%f").minute
        h = datetime.strptime(time_str, "%H:%M:%S.%f").hour
        return ms/1e6 + s + m*60 + h*3600
    if isinstance(start_time, str):
        start_time = str_to_float(start_time)
    if isinstance(end_time, str):
        end_time = str_to_float(end_time)

    start_frame_idx = int(start_time * fps)
    end_frame_idx = int(end_time * fps)
    if end_frame_idx <= start_frame_idx:
        end_frame_idx = start_frame_idx + 1 # Solve cases where start_time >= end_time
    try:
        assert end_frame_idx < len(interpolated_tokens)
    except AssertionError:
        # print('end_frame_index:', end_frame_idx)
        # print('len(labels_list):', len(labels_list))
        return unk

    token_slice = interpolated_tokens[start_frame_idx: end_frame_idx]
    c = Counter(token_slice)
    try:
        token, _ = c.most_common(1)[0]
    except IndexError:
        print('label_slice:', token_slice)
        print('Counter:', c)
        raise
    return token


def extract_words_gestures_from_vtt(vtt_file: str, interpolated_tokens: List[int]) -> List[Tuple[Union[str, int]]]:
    """
    vtt_file: An .vtt file that contains the automatic generated subscript from a youtube video
    interpolated_tokens: the interpolated gesture tokens from the same video, which is obtaiend from interpolate_gesture_tokens()
    """
    vtt = webvtt.read(vtt_file)
    all_gestures = []
    all_words = []
    for i in range(len(vtt.captions)):
        for s in vtt.captions[i].lines:
            ss = re.findall('<(.+?)>', s)
            if len(ss) == 0:
                continue
            ss.insert(0, vtt.captions[i].start)
            ss.insert(len(ss), vtt.captions[i].end)

            # Use re.split
            words = re.split(r'<\d\d:\d\d:\d\d\.\d+><c>', s) #<\d\d:\d\d:\d\d\.\d+><c>|</c><\d\d:\d\d:\d\d\.\d+>
            words_cleaned = []
            for w in words:
                if w.endswith('</c>'):
                    words_cleaned.append(w[:-4].strip())
                else:
                    words_cleaned.append(w.strip())

            time_intervals = [x for x in ss if 'c' not in x]
            assert len(time_intervals) == len(words_cleaned) + 1
            gestures = []
            for i, w in enumerate(words_cleaned):
                start = time_intervals[i]
                end = time_intervals[i+1]
                try:
                    token = get_token_by_timespan(start, end, interpolated_tokens)
                except Exception:
                    print(start, end, f'@{vtt_file}')
                    raise
                gestures.append(token)
            all_gestures.append(tuple(gestures))
            all_words.append(tuple(words_cleaned))
    return all_gestures, all_words 


def process_single_file(gesture_file, vtt_file, output_word_dir=None, output_gesture_dir=None, output_mixed_dir=None):
    # double check if the video ids are matched
    file_id1, _ = os.path.splitext(os.path.basename(gesture_file))
    file_id2 = re.split(r'.en', os.path.basename(vtt_file))[0]
    assert file_id1 == file_id2

    intp_tokens = interpolate_gesture_tokens(gesture_file)
    gestures, words = extract_words_gestures_from_vtt(vtt_file, intp_tokens)
    if output_word_dir:
        os.makedirs(output_word_dir, exist_ok=True)
        output_words_file = os.path.join(output_word_dir, file_id1 + '.txt')
        with open(output_words_file, 'w') as f:
            for item in words:
                f.write(' '.join(item) + '\n')
    if output_gesture_dir:
        os.makedirs(output_gesture_dir, exist_ok=True)
        output_gestures_file = os.path.join(output_gesture_dir, file_id1 + '.txt')
        with open(output_gestures_file, 'w') as f:
            for item in gestures:
                f.write(' '.join(map(str, item)) + '\n')
    if output_mixed_dir:
        os.makedirs(output_mixed_dir, exist_ok=True)
        output_mixed_file = os.path.join(output_mixed_dir, file_id1 + '.txt')
        with open(output_mixed_file, 'w') as f:
            for item in zip(words, gestures):
                ws, tkns = item
                f.write('\t'.join([' '.join(ws), ' '.join(map(str, tkns))]) + '\n')


def test():
    gest_file = '../data/gestures_grid_3x3/0C_jkAilRG4.csv'
    vtt_file = '../data/raw_videos/0C_jkAilRG4.en.vtt'
    intp_tokens = interpolate_gesture_tokens(gest_file)
    gestures, words = extract_words_gestures_from_vtt(vtt_file, intp_tokens)
    print(len(gestures))
    print(len(words))
    print(gestures[-1])
    print(words[-1])
    process_single_file(gest_file, vtt_file, output_word_dir='../data/words_by_id', output_gesture_dir='../data/gestures_by_id',
        output_mixed_dir='../data/mixed_by_id')


def main(args):
    # Single processing
    if args.input_gesture_file:
        assert args.input_vtt_file is not None
        process_single_file(args.input_gesture_file, args.input_vtt_file, args.output_word_dir, args.output_gesture_dir, args.output_mixed_dir)
    # Batch processing
    if args.input_gesture_dir:
        assert args.input_vtt_dir is not None
        input_gesture_files = glob.glob(os.path.join(args.input_gesture_dir, '*.csv'))
        input_gesture_files = sorted(input_gesture_files)
        input_vtt_files = glob.glob(os.path.join(args.input_vtt_dir, '*.vtt'))
        input_vtt_files = sorted(input_vtt_files)
        print(f'# of input gesture files: {len(input_gesture_files)}')
        print(f'# of input vtt files: {len(input_vtt_files)}')

        gesture_file_ids = []
        for g_file in input_gesture_files:
            file_id, _ = os.path.splitext(os.path.basename(g_file))
            gesture_file_ids.append(file_id)
        vtt_file_ids = []
        for v_file in input_vtt_files:
            file_id = re.split(r'.en', os.path.basename(v_file))[0]
            vtt_file_ids.append(file_id)
        common_ids = set(gesture_file_ids).intersection(set(vtt_file_ids))
        strange_g_ids = list(set(gesture_file_ids) - common_ids)
        strange_v_ids = list(set(vtt_file_ids) - common_ids)
        print(f'Strange gesture ids: {strange_g_ids}')
        print(f'Strange vtt ids: {strange_v_ids}')
        print(f'# of matched file pairs: {len(common_ids)}')

        for file_id in tqdm(common_ids):
            g_file = os.path.join(args.input_gesture_dir, file_id + '.csv')
            v_file = os.path.join(args.input_vtt_dir, file_id + '.en.vtt')
            process_single_file(g_file, v_file, output_word_dir=args.output_word_dir, output_gesture_dir=args.output_gesture_dir, output_mixed_dir=args.output_mixed_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


####
# 2022.12.13 log
# of input gesture files: 71
# of input vtt files: 66
# Strange gesture ids: ['XZ5MlLQCQrs', 'Rx14L3TXSWA', 'ukNrZuVb4Ew', 'aL3RERxiYwo', 'iJIa9YOLxIs', 'Ygmqh6nyaRU', '9gnhFpRTPR8', 'DcSDlJuqCT4', 'T7sBbdYt7hw']
# Strange vtt ids: ['WQWiLZ1M6xw', 'iNtLNde_v1E', 'hXkiAfjFtgU', 'YXvXxZtseo0']
# of matched file pairs: 62
