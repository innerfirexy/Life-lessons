from argparse import Namespace
import subprocess
import pickle
import cv2
import re
import os
import face_recognition
from tqdm import tqdm


class Task:
    def __init__(self):
        self.done = False
        self.status = None
    def set_done(self):
        self.done = True


class DownloadTask(Task):
    command = 'yt-dlp'
    def __init__(self, download_path: str, url: str, args: Namespace):
        super(DownloadTask, self).__init__()
        self.download_path = download_path
        self.url = url
        self.args = args
        if re.search(r'(?<=\?v\=).+', url) is None:
            self.video_id = url
        else:
            self.video_id = re.search(r'(?<=\?v\=).+', url).group(0)
        self.downloaded_video_file = os.path.join(self.download_path, self.video_id + '.mp4')
        self.n_trials = 0

    def execute(self):
        if os.path.exists(self.downloaded_video_file):
            self.set_done()
            self.status = 'success'
            return 
        # Run download command
        ret = subprocess.run([DownloadTask.command, self.url, '--paths', self.download_path, 
        '--output', '%(id)s.%(ext)s', '--format', 'mp4', '--write-auto-subs',
        '--quiet'])
        if ret.returncode == 0:
            self.set_done()
            self.status = 'success'
        else:
            self.status = 'failure'
        self.n_trials += 1
        return ret

    def create_next_task(self):
        try:
            parse_task = ParseTask(input_file=self.downloaded_video_file, sample_interval=self.args.sample_interval, 
            batch_mode=self.args.batch_mode, batch_size=self.args.batch_size, delete_after_done=self.args.delete_after_done, video_id=self.video_id)
        except Exception:
            print('Error in creating task for {}'.format(self.downloaded_video_file))
            raise
        return parse_task


class ParseTask(Task):
    def __init__(self, input_file: str, sample_interval: int, batch_mode: bool, batch_size: int,
                 delete_after_done: bool, video_id: str):
        """
        :param input_file: Path of input video file
        :param sample_interval: Interval of sampling the input video, measured by number of frames
        :param delete_after_done: If True, delete the input video after the parsing is successfully done
        :param kwargs:
        """
        super(ParseTask, self).__init__()
        self.input_file = input_file
        self.sample_interval = sample_interval
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.delete_after_done = delete_after_done
        self.parse_result = None
        self.video_id = video_id
        self._init_output_path(input_file)
    
    def _init_output_path(self, input_file: str):
        output_filename, _ = os.path.splitext(input_file)
        self.output_path = output_filename + '.pkl'
    
    def _delete_input_file(self):
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
    
    def _save_parse_result(self):
        pickle.dump(self.parse_result, open(self.output_path, 'wb'))
    
    def get_log_str(self):
        numbers_of_faces = self.parse_result[1]
        one_face_ratio = numbers_of_faces.count(1) / len(numbers_of_faces)
        zero_face_ratio = numbers_of_faces.count(0) / len(numbers_of_faces)
        return f'{self.video_id},{self.duration},{zero_face_ratio},{one_face_ratio}\n'

    def execute(self, pid:int=None):
        # print(f'Worker-{pid} started parsing {self.input_file}')
        try:
            video_capture = cv2.VideoCapture(self.input_file)
        except Exception:
            self.status = 'failure'
            return 1
        else:
            total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            self.total_frame_count = total_frame_count
            self.duration = total_frame_count / fps
            if self.batch_mode:
                # Use CUDA for faster processing
                # https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py
                frames = []
                current_frame_index = -1
                frame_indices = []
                number_of_faces_in_frames = []
                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    current_frame_index += 1
                    if (current_frame_index + 1) % self.sample_interval == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        frame_indices.append(current_frame_index)
                    if len(frames) == self.batch_size \
                            or current_frame_index == total_frame_count - 1: # Last frame
                        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)
                        for face_locations in batch_of_face_locations:
                            num_faces = len(face_locations)
                            number_of_faces_in_frames.append(num_faces)
                        frames = []
                self.status = 'success'
                self.set_done()
                self.parse_result = (frame_indices, number_of_faces_in_frames)
            else:
                # Process frames one by one
                # https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py
                current_frame_index = -1
                frame_indices = []
                number_of_faces_in_frames = []
                pbar = tqdm(total=total_frame_count//self.sample_interval, desc=f'Worker-{pid}', position=pid, leave=False)
                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    current_frame_index += 1
                    if (current_frame_index + 1) % self.sample_interval == 0:
                        frame_indices.append(current_frame_index)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(frame)
                        num_faces = len(face_locations)
                        number_of_faces_in_frames.append(num_faces)
                        pbar.update(1)
                self.status = 'success'
                self.set_done()
                self.parse_result = (frame_indices, number_of_faces_in_frames)
                self._save_parse_result()
                pbar.close()

            video_capture.release()
            # Delete video if needed
            if self.delete_after_done:
                self._delete_input_file()
            return 0