import numpy as np
import pandas as pd
import argparse
import os
import glob
import pickle
from tqdm import tqdm


###
# Arg parse
###
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='Input directory that contains extracted keypoints files (e.g., .pkl)')
parser.add_argument('--output_dir', type=str, help='Output directory to save tokenized data')
parser.add_argument('--input', type=str, help='A single input file (.pkl)')
parser.add_argument('--output', type=str, help='An output file name')
parser.add_argument('--output_ext', type=str, default='.pkl', choices=['.pkl', '.csv'])
parser.add_argument('--N', type=int, default=3, help='The N value for creating NxN grids')
parser.add_argument('--return_type', type=str, default='list', choices=['list', 'df'])
parser.add_argument('--return_grids', action='store_true')
parser.add_argument('--multiprocessing', '-mp', action='store_true')


def check_args(args):
    # Check output extension
    if args.output_ext == '.csv':
        if args.return_type == 'list' or args.return_grids == True:
            print('--output_ext can only be .csv when --return_type is \'df\' and --return_grids is False')
            return False
    # Check input and output paths
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


###
# Functions
###
def get_labels(data_list, N = 3, hw_ratio = 720/1280, return_type = 'list', return_grids = False):
    """
    data_list: list[tuple], in which tuple[0] is the frame_idx, tuple[1] is a (33,3) np array
    grid: int, number of grids. E.g., grid = 3 results in a 3 by 3 split
    return_type: 'list' | 'df'
    """
    assert isinstance(data_list, list)

    results = []
    grids = []
    for frame_idx, data in data_list:
        nose_x = data[0][0]
        l_shoulder_x = data[11][0]
        r_shoulder_x = data[12][0]
        l_hip_x = data[23][0]
        r_hip_x = data[24][0]

        # y1, y2 = 0.001, 0.999 # This is the simplist method
        l_eye_y = data[2][1]
        r_eye_y = data[5][1]
        mid_eye_y = (l_eye_y + r_eye_y) / 2
        nose_y = data[0][1]
        nose_eye_diff = nose_y - mid_eye_y
        y1 = max(0.001, mid_eye_y - nose_eye_diff * 2) # x2 because: https://www.artyfactory.com/portraits/pencil-portraits/proportions-of-a-head.html
        l_hip_y = data[23][1]
        r_hip_y = data[24][1]
        mid_hip_y = (l_hip_y + r_hip_y) / 2
        y2 = min(0.999, mid_hip_y)

        grids_height = y2 - y1
        grids_width = grids_height * hw_ratio # normalized by hw_ratio

        xc = (nose_x + (l_shoulder_x + r_shoulder_x)/2 + (l_hip_x + r_hip_x)/2) / 3
        # print(xc)
        x1 = xc - 0.5 * grids_width + 0.001
        x2 = xc + 0.5 * grids_width - 0.001
        if x1 <= 0.0 or x2 >= 1.0:
            results.append((frame_idx, None, None, None))
            continue


        # Compute grids for current frame
        if return_grids:
            cur_grid = [x1]
            for i in range(1,N):
                cur_grid.append(x1 + (x2-x1) * i/N)
            cur_grid.append(x2)
            cur_grid.append(y1)
            for j in range(1,N):
                cur_grid.append(y1 + (y2-y1) * j/N)
            cur_grid.append(y2)
            grids.append(tuple(cur_grid))

        # l_wrist_x, l_wrist_y = data[15][0], data[15][1]
        # r_wrist_x, r_wrist_y = data[16][0], data[16][1]
        l_pinky_x, l_pinky_y = data[17][0], data[17][1]
        r_pinky_x, r_pinky_y = data[18][0], data[18][1]
        l_index_x, l_index_y = data[19][0], data[19][1]
        r_index_x, r_index_y = data[20][0], data[20][1]
        l_thumb_x, l_thumb_y = data[21][0], data[21][1]
        r_thumb_x, r_thumb_y = data[22][0], data[22][1]
        l_hand_x = (l_pinky_x + l_index_x + l_thumb_x) / 3
        l_hand_y = (l_pinky_y + l_index_y + l_thumb_y) / 3
        r_hand_x = (r_pinky_x + r_index_x + r_thumb_x) / 3
        r_hand_y = (r_pinky_y + r_index_y + r_thumb_y) / 3

        l_col = np.floor(min(max(l_hand_x - x1, 0), x2-x1) / grids_width * N) + 1
        r_col = np.floor(min(max(r_hand_x - x1, 0), x2-x1) / grids_width * N) + 1
        l_row = np.floor(min(max(l_hand_y - y1, 0), y2-y1) * N) + 1
        r_row = np.floor(min(max(r_hand_y - y1, 0), y2-y1) * N) + 1

        l_label = int((l_row - 1)*N + l_col)
        r_label = int((r_row - 1)*N + r_col)
        label = (l_label - 1)*N*N + r_label

        results.append((frame_idx, l_label, r_label, label))
        # results.append((frame_idx, l_label, r_label, l_label*r_label))  # This needs be fixed

    if return_type == 'list':
        if return_grids:
            return results, grids
        else:
            return results
    elif return_type == 'df':
        results_cleaned = list(filter(lambda x: x[1] != None, results))
        data = np.array(results_cleaned, dtype=[('frame_idx', 'i4'), ('l_label', 'i4'), ('r_label', 'i4'), ('label', 'i4')])
        df = pd.DataFrame.from_records(data)
        if return_grids:
            return df, grids
        else:
            return df


def main(args):
    # Single processing
    if args.input:
        with open(args.input, 'rb') as fr:
            kp_data = pickle.load(fr)
        res = get_labels(kp_data, N=args.N, return_type=args.return_type, return_grids=args.return_grids)
        print(res)
        if args.output_ext == '.pkl':
            with open(args.output, 'wb') as f:
                pickle.dump(res, f)
        elif args.output_ext == '.csv':
            assert isinstance(res, pd.DataFrame)
            res.to_csv(args.output, index=False)
    # Batch processing
    if args.input_dir:
        input_files = glob.glob(os.path.join(args.input_dir, '*.pkl'))
        if args.multiprocessing:
            pass
        else:
            for in_file in tqdm(input_files):
                with open(in_file, 'rb') as f:
                    kp_data = pickle.load(f)
                in_file_name, _ = os.path.splitext(os.path.basename(in_file))
                out_file = os.path.join(args.output_dir, in_file_name + args.output_ext)
                res = get_labels(kp_data, N=args.N, return_type=args.return_type, return_grids=args.return_grids)
                if args.output_ext == '.pkl':
                    with open(out_file, 'wb') as f:
                        pickle.dump(res, f)
                elif args.output_ext == '.csv':
                    assert isinstance(res, pd.DataFrame)
                    res.to_csv(out_file, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    assert(check_args(args))
    main(args)