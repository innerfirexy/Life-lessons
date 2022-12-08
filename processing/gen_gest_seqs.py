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
def interpolate_gesture_tokens(csv_file, unk=0) -> List[int]:
    """
    csv_file: str, a csv file in `label_dir`
    """
    df = pd.read_csv(csv_file)
    labels = df['label'].tolist()
    frame_indices = df['frame_idx'].tolist()
    min_frame_idx = 0
    max_frame_idx = frame_indices[-1]
    frame_indices = set(frame_indices)
    full_frame_indices = list(range(0, max_frame_idx+1))

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
        print(f'problem in {csv_file}')
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
    pass