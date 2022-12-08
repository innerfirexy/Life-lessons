import re
import webvtt
import glob
import os
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import itertools
from collections import Counter
from typing import Tuple, Union, List


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


def process_single_file(gesture_file, vtt_file, output_words_dir=None, output_gestures_dir=None, output_mixed_dir=None):
    # double check if the video ids are matched
    file_id1, _ = os.path.splitext(os.path.basename(gesture_file))
    file_id2 = re.split(r'.en', os.path.basename(vtt_file))[0]
    assert file_id1 == file_id2

    intp_tokens = interpolate_gesture_tokens(gesture_file)
    gestures, words = extract_words_gestures_from_vtt(vtt_file, intp_tokens)
    if output_words_dir:
        os.makedirs(output_words_dir, exist_ok=True)
        output_words_file = os.path.join(output_words_dir, file_id1 + '.txt')
        with open(output_words_file, 'w') as f:
            for item in words:
                f.write(' '.join(item) + '\n')
    if output_gestures_dir:
        os.makedirs(output_gestures_dir, exist_ok=True)
        output_gestures_file = os.path.join(output_gestures_dir, file_id1 + '.txt')
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
    process_single_file(gest_file, vtt_file, output_words_dir='../data/words_by_id', output_gestures_dir='../data/gestures_by_id',
        output_mixed_dir='../data/mixed_by_id')


if __name__ == '__main__':
    test()