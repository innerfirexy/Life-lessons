import numpy as np
import pandas as pd
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
import time
import json
from tqdm import tqdm
from argparse import ArgumentParser
warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path

def run(data):
    video_id, args = data
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
       download(video_id.split('#')[0], args)

def read_metadata(metadata_path):
    metadata_files = glob.glob(os.path.join(metadata_path, '*.md'))
    # print(len(metadata_files))

    video_ids = []
    durations = []
    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            for line in f:
                if not line.startswith('{'):
                    continue
                meta = json.loads(line.strip())
                video_ids.append(meta['id'])
                durations.append(meta['duration'])

    df = pd.DataFrame.from_dict({'video_id': video_ids, 'duration':durations})

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='youtube-video', help='Path to youtube videos')
    parser.add_argument("--metadata", default='../metadata', help='Path to metadata')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--youtube", default='./youtube-dl', help='Path to youtube-dl')
 
    args = parser.parse_args()
    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)

    df = read_metadata(args.metadata)
    video_ids = set(df['video_id'])

    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        None  
    """
    """
