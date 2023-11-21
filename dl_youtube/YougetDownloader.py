import os
import sys
import glob
import json
import time
import queue
import argparse
import multiprocessing as mp
from typing import List
from utils import Task, YougetDownloadTask

# Fix mp.Queue.qsize() problem on MacOS & Windows
import platform

if platform.system() == "Darwin" or "Windows":
    from FixedQueue import Queue
else:
    from multiprocessing.queues import Queue


# Create argument parser
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="a folder containing multiple files",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="a file containing the urls to be downloaded, overwrites --folder",
    )
    parser.add_argument(
        "--urls",
        type=str,
        help="Multiple video urls should be separated by space, overwrites --file and --folder",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="if specified, you-get will be called with --playlist argument, which will download all videos in the playlist if possible",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        help="a file containing the cookies for the website",
    )
    parser.add_argument("--download_path", type=str, default="./tmp")
    parser.add_argument("--num_workers", type=int, default=2)
    return parser


def download_task_worker(
    tasks_assigned, tasks_done, tasks_failed, next_tasks_assigned=None
):
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


def load_tasks_from_folder(folder: str, args) -> List[Task]:
    tasks = []
    for file in glob.glob(os.path.join(folder, f"*.json")):
        tasks += load_tasks_from_file(file, args)
    return tasks


def load_tasks_from_file(file: str, args) -> List[Task]:
    video_urls = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            if line.startswith("{"):
                json_obj = json.loads(line)
                video_urls.append(json_obj["url"])
    tasks = []
    for url in video_urls:
        tasks.append(YougetDownloadTask(url, args.download_path, args))
    return tasks


def load_tasks_from_urls(urls: str, args) -> List[Task]:
    tasks = []
    video_urls = urls.strip().split(' ')
    video_urls = [vid for vid in video_urls if vid]
    for url in video_urls:
        tasks.append(YougetDownloadTask(url, args.download_path, args))
    return tasks


def main(args):
    if args.folder is not None:
        dl_tasks = load_tasks_from_folder(args.folder, args)
    if args.file is not None:
        dl_tasks = load_tasks_from_file(args.file, args)
    if args.urls is not None:
        dl_tasks = load_tasks_from_urls(args.urls, args)
    if dl_tasks is None:
        raise ValueError("No tasks loaded")

    dl_tasks_assigned = Queue()
    dl_tasks_done = Queue()
    dl_tasks_failed = Queue()
    for t in dl_tasks:
        dl_tasks_assigned.put(t)
    num_dl_tasks = len(dl_tasks)

    processes = []
    for _ in range(args.num_workers):
        p = mp.Process(
            target=download_task_worker,
            args=(dl_tasks_assigned, dl_tasks_done, dl_tasks_failed),
        )
        processes.append(p)
        p.start()

    last_num_tasks_done = 0
    last_num_tasks_failed = 0
    last_num_tasks_remain = 0
    while True:
        time.sleep(1)
        num_tasks_done = dl_tasks_done.qsize()
        num_tasks_failed = dl_tasks_failed.qsize()
        num_tasks_remain = num_dl_tasks - num_tasks_done - num_tasks_failed
        if last_num_tasks_done != num_tasks_done or last_num_tasks_failed != num_tasks_failed or last_num_tasks_remain != num_tasks_remain:
            sys.stdout.write(
                f"\r Remaining tasks #: {num_tasks_remain} | Done: {num_tasks_done} | Failed: {num_tasks_failed}"
            )
            sys.stdout.flush()
            last_num_tasks_done = num_tasks_done
            last_num_tasks_failed = num_tasks_failed
            last_num_tasks_remain = num_tasks_remain
        if num_tasks_remain == 0:
            print("\n" + "=" * 12)
            print("All download tasks done")
            break
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
