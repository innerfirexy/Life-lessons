import pickle
from typing import List, Tuple
import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

###
# Arg parse
###
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, help='The input video file (.mp4)')
parser.add_argument('--gesture_file', type=str, help='The input file (.pkl or .csv) that contains the gesture token and grids\' coordinates for each frame, which is generated from tokenize_gest_grid.py')
parser.add_argument('--output', type=str, help='An output file (.mp4) to write the video with grids plotted')


def load_gesture_data(gesture_file: str, args):
    if not os.path.exists(gesture_file):
        print(f'gesture_file {gesture_file} does not exist.')
        raise FileNotFoundError
    try:
        with open(gesture_file, 'rb') as f:
            gestures_list, grids_list = pickle.load(f)
        assert isinstance(grids_list[0], tuple)
        assert len(grids_list[0]) == 2
    except Exception:
        print(f'gesture_file {gesture_file} corrupted or in wrong format.')
        raise
    else:
        # Calcuate N, the number of splits along width or height (N x N)
        # the formula is: dim = (N+1)*2, i.e., N = dim // 2 - 1
        dim = len(grids_list[0][1])
        N = dim // 2 - 1
        vars(args)['N'] = N
        print(f'gesture_file loaded: dim={dim}, N={args.N}')
        return gestures_list, grids_list


def annotate_gestures_grids(input_file, output_file, gestures_list: List[Tuple], grids_list: List[Tuple], N: int):
    """
    :param input_file:
    :param output_file:
    :param gestures_list:
    :param grids_list:
    :return:
    """
    assert(len(gestures_list) == len(grids_list))
    assert(len(grids_list[0][1]) == 2*(N+1))
    # Convert gestures_list to a dict. E.g., [(0, 9, 8, 80), (1, 9, 8, 80), ...] => {0: (9,8,80), 1:(9,8,80), ...}
    gestures_dict = {it[0]: it[1:] for it in gestures_list if it[1] is not None} 
    # Also convert grids to a dict that shares the same keys as gestures_dict
    grids_dict = {it[0]: it[1] for it in grids_list if it[1] is not None}

    cap = cv2.VideoCapture(input_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output = cv2.VideoWriter(output_file,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, size)
    # Green color in BGR,
    color = (0, 255, 0)
    thickness = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    pbar = tqdm(total=n_frames)
    frame_idx = -1
    while cap.isOpened():
        frame_idx += 1
        success, image = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            break
        if frame_idx not in gestures_dict:
            continue

        xs = grids_dict[frame_idx][:N+1]
        ys = grids_dict[frame_idx][N+1:]
        l_label, r_label, label = gestures_dict[frame_idx]
        l_row = np.ceil(l_label / N)
        l_col = l_label - (l_row-1) * N
        r_row = np.ceil(r_label / N)
        r_col = r_label - (r_row-1) * N
        grid_w = xs[1] - xs[0]
        grid_h = ys[1] - ys[0]

        l_label_x = xs[0] + (l_col - 0.5) * grid_w
        l_label_y = ys[0] + (l_row - 0.5) * grid_h
        l_label_x = int(l_label_x * w)
        l_label_y = int(l_label_y * h)
        r_label_x = xs[0] + (r_col - 0.5) * grid_w
        r_label_y = ys[0] + (r_row - 0.5) * grid_h
        r_label_x = int(r_label_x * w)
        r_label_y = int(r_label_y * h)
        image = cv2.putText(image, str(l_label), (l_label_x, l_label_y), fontFace=font, fontScale=font_scale, color=color, thickness=thickness)
        image = cv2.putText(image, str(r_label), (r_label_x, r_label_y), fontFace=font, fontScale=font_scale, color=color, thickness=thickness)

        for i in range(N+1):
            start = (int(xs[i]*w), int(ys[0]*h))
            end = (int(xs[i]*w), int(ys[-1]*h))
            image = cv2.line(image, start, end, color, thickness)
        for j in range(N+1):
            start = (int(xs[0]*w), int(ys[j]*h))
            end = (int(xs[-1]*w), int(ys[j]*h))
            image = cv2.line(image, start, end, color, thickness)

        image = cv2.putText(image, 'Left label: ' + str(l_label), (10, 50), fontFace=font, fontScale=font_scale, color=color, thickness=thickness)
        image = cv2.putText(image, 'Right label: ' + str(r_label), (10, 100), fontFace=font, fontScale=font_scale, color=color, thickness=thickness)
        image = cv2.putText(image, 'Label: ' + str(label), (10, 150), fontFace=font, fontScale=font_scale, color=color, thickness=thickness)

        output.write(image)
        pbar.update(1)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main(args):
    gestures_list, grids_list = load_gesture_data(args.gesture_file, args)
    # print(len(gestures_list), len(grids_list))
    annotate_gestures_grids(args.input_video, args.output, gestures_list, grids_list, args.N)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)