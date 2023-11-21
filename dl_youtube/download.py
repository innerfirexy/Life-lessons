import json
import queue
import multiprocessing as mp
import sys
import argparse
import os
import time
import glob
import itertools
from typing import List
from utils import Task, DownloadTask

# Fix mp.Queue.qsize() problem on MacOS & Windows
import platform
if platform.system() == 'Darwin' or 'Windows':
    from FixedQueue import Queue
else:
    from multiprocessing.queues import Queue


# Create argument parser
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--play_list_file', type=str, help='a file containing the urls for a list of videos to be downloaded')
    parser.add_argument('--play_list_folder', type=str, help='a folder containing multiple play list files')
    parser.add_argument('--play_list_file_format', type=str, choices=['.md', '.txt'], default='.md')
    parser.add_argument('--download_path', type=str, default='./tmp')
    parser.add_argument('--video_ids', type=str, help='this option overwrites --play_list_file and --play_list_folder. Multiple video ids should be separated by space')
    parser.add_argument('--num_workers', type=int, default=2)
    return parser


def download_task_worker(tasks_assigned, tasks_done, tasks_failed, next_tasks_assigned=None):
    while True:
        try:
            task = tasks_assigned.get_nowait()
        except queue.Empty:
            break
        else:
            # do the task
            task.execute()
            if task.done:
                tasks_done.put(task)
                if next_tasks_assigned is not None:
                    # Create parse task and add it to the queue
                    next_task = task.create_next_task()
                    next_tasks_assigned.put(next_task)
            else:
                tasks_failed.put(task)
    return True


def load_tasks_from_playlist_file(play_list_file: str, args) -> List[Task]:
    video_urls = []
    with open(play_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            if line.startswith('{'):
                json_obj = json.loads(line)
                if json_obj['duration'] is not None: # duration is null for private videos
                    video_urls.append(json_obj['url'])
    tasks = []
    for url in video_urls:
        tasks.append(DownloadTask(download_path=args.download_path, url=url, args=args))
    return tasks


def load_tasks_from_folder(play_list_folder: str, args) -> List[Task]:
    loaded_tasks = []
    play_list_files = glob.glob(os.path.join(play_list_folder, '*' + args.play_list_file_format)) # default: '*.md'
    for pl_file in play_list_files:
        tasks = load_tasks_from_playlist_file(pl_file, args)
        loaded_tasks.append(tasks)
    loaded_tasks = list(itertools.chain.from_iterable(loaded_tasks))
    return loaded_tasks


def create_tasks_from_video_ids(video_ids: List[str], args) -> List[Task]:
    base_url = 'https://www.youtube.com/watch?v='
    tasks = []
    for vid in video_ids:
        url = base_url + vid
        tasks.append(DownloadTask(download_path=args.download_path, url=url, args=args))
    return tasks


def main(args):
    if args.play_list_folder:
        dl_tasks = load_tasks_from_folder(args.play_list_folder, args)
    elif args.play_list_file:
        dl_tasks = load_tasks_from_playlist_file(args.play_list_file, args)
    elif args.video_ids:
        video_ids = args.video_ids.strip().split(' ')
        video_ids = [vid for vid in video_ids if vid]
        dl_tasks = create_tasks_from_video_ids(video_ids, args)

    dl_tasks_assigned = Queue()
    dl_tasks_done = Queue()
    dl_tasks_failed = Queue()
    for t in dl_tasks:
        dl_tasks_assigned.put(t)
    num_dl_tasks = len(dl_tasks)

    processes = []
    for _ in range(args.num_workers):
        p = mp.Process(target = download_task_worker, args=(dl_tasks_assigned, dl_tasks_done, dl_tasks_failed))
        processes.append(p)
        p.start()
    
    while True:
        time.sleep(1)
        num_tasks_done = dl_tasks_done.qsize()
        num_tasks_failed = dl_tasks_failed.qsize()
        num_tasks_remain = num_dl_tasks - num_tasks_done - num_tasks_failed
        sys.stdout.write(f'\r Remaining tasks #: {num_tasks_remain} | Done: {num_tasks_done} | Failed: {num_tasks_failed}')
        sys.stdout.flush()
        if num_tasks_remain == 0:
            print('='*12)
            print('All download tasks done')
            break
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)