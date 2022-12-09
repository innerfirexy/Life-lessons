import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import glob
import pickle
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock


# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


###
# Arg parse
###
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='Input directory that contains extracted keypoints files (e.g., .pkl)')
parser.add_argument('--output_dir', type=str, help='Output directory to save labeled data')
parser.add_argument('--input', type=str, help='A single input file (.mp4)')
parser.add_argument('--output', type=str, help='An output file name (.pkl)')
parser.add_argument('--annotate', action='store_true')
parser.add_argument('--multiprocessing', '-mp', action='store_true')
parser.add_argument('--num_workers', type=int, default=2)


def check_args(args):
    if args.input_dir:
        # Check input_dir
        if not os.path.exists(args.input_dir):
            print('--input_dir path does not exist.')
            return False
        elif not os.path.isdir(args.input_dir):
            print('--input_dir is not a valid folder.')
            return False
        # Check output_dir
        if args.output_dir is None:
            print('--output_dir needs be specified.')
            return False
        elif not os.path.exists(args.output_dir):
            print('--output_dir path does not exist. Please manually create it.')
            return False
        elif not os.path.isdir(args.output_dir):
            print('--output_dir is not a valid folder.')
            return False
        # Check annotate
        if args.annotate:
            print('--annotated is suggested off in batch processing mode')
            return False
        return True
    elif args.input:
        # Check input
        if not os.path.exists(args.input):
            print('--input path does not exist.')
            return False
        # Check output
        if args.output is None:
            print('--output needs be specified.')
            return False
        return True
    else:
        print('Either one of --input_dir and --input needs be specified.')
        return False


##
# For multiprocessing
# Adopted from https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
def worker(pid: int, input_file: str, args):
    in_file_name, _ = os.path.splitext(os.path.basename(input_file))
    out_file = os.path.join(args.output_dir, in_file_name + '.pkl')
    res = body_pose(input_file=input_file, verbose=True, pid=pid)
    with open(out_file, 'wb') as f:
        pickle.dump(res, f)
    return True


##
# Run Mediapipe on single input video file
def body_pose(input_file: str, annotated_output: str = None, verbose = False, pid: int = 1):
    cap = cv2.VideoCapture(input_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if annotated_output:
        output = cv2.VideoWriter(annotated_output,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, size)
    result_list = []
    pbar = None
    if verbose:
        pbar = tqdm(total=n_frames, desc='# {}'.format(pid).zfill(2), position=pid)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # print("Ignoring empty camera frame.")
                break # For camera use, replace with `continute`
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # No need to flip horizontally

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)
            poses = []
            if results.pose_landmarks:
                for i in range(33):
                    poses.append(results.pose_landmarks.landmark[i].x)
                    poses.append(results.pose_landmarks.landmark[i].y)
                    poses.append(results.pose_landmarks.landmark[i].z)
                poses = np.reshape(poses, (33, 3))
                result_list.append((frame_idx, poses))

            # #Draw the pose annotation on the image. (Not necessary)
            if annotated_output:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert RGB to BGR
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow('MediaPipe Pose', image)
                output.write(image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            # Progress bar
            frame_idx += 1
            if pbar:
                pbar.update(1)

    cap.release()
    if pbar:
        pbar.close()
    if annotated_output:
        output.release()
    cv2.destroyAllWindows()

    return result_list



def main(args):
    if args.input:
        if args.annotate:
            input_filename, _ = os.path.splitext(args.input)
            annotated_file = input_filename + '_annotated.mp4'
        else:
            annotated_file = None
        res = body_pose(input_file=args.input, annotated_output=annotated_file, verbose=True)
        with open(args.output, 'wb') as f:
            pickle.dump(res, f)
    if args.input_dir:
        input_files = glob.glob(os.path.join(args.input_dir, '*.mp4'))
        if args.multiprocessing:
            # Multiprocessing method adopted from https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
            freeze_support() # For Windows support
            pool = Pool(processes=args.num_workers, initargs=(RLock(),), initializer=tqdm.set_lock)
            jobs = [pool.apply_async(worker, args=(i+1,fname,args)) for i, fname in enumerate(input_files)]
            count = 0
            for job in jobs:
                if job.get():
                    count += 1
            pool.close()
            print(f'\n{count} jobs done.')
        else:
            for in_file in tqdm(input_files):
                in_file_name, _ = os.path.splitext(os.path.basename(in_file))
                out_file = os.path.join(args.output_dir, in_file_name + '.pkl')
                res = body_pose(input_file=in_file, verbose=True)
                with open(out_file, 'wb') as f:
                    pickle.dump(res, f)


if __name__ == '__main__':
    args = parser.parse_args()
    assert(check_args(args))
    main(args)