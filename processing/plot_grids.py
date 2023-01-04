from typing import List, Tuple
import cv2

def annotate_labels_grids(input_file, output_file, labels: List[Tuple], grids: List[Tuple], N: int):
    """
    :param input_file:
    :param output_file:
    :param labels:
    :param grids:
    :param N: Same N as in get_labels()
    :return:
    """
    assert(len(labels) == len(grids))
    assert(len(grids[0]) == 2*(N+1))
    # Convert labels to a dict. E.g., [(0, 9, 8, 80), (1, 9, 8, 80), ...] => {0: (9,8,80), 1:(9,8,80), ...}
    labels_dict = {it[0]: it[1:] for it in labels}
    # Also convert grids to a dict that shares the same keys as labels_dict
    grids_dict = {it[0]: grids[i] for i, it in enumerate(labels)}

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
    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            break
        if frame_idx not in labels_dict:
            continue

        xs = grids_dict[frame_idx][:N+1]
        ys = grids_dict[frame_idx][N+1:]
        l_label, r_label, label = labels_dict[frame_idx]
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
        frame_idx += 1
        pbar.update(1)
        if cv2.waitKey(1) & 0xFF == 27:
            break