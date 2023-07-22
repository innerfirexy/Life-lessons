# Life-lessons
A dataset of first-person monologue videos/transcript/annotations about "life lessons" in various domains. The main purpose is for multi-modal language analysis and modeling.

## Published papers
[Findings of ACL 2023: Spontaneous gestures encoded by hand positions improve language models: An Information-Theoretic motivated study](https://aclanthology.org/2023.findings-acl.600/)
```
@inproceedings{xu-cheng-2023-spontaneous,
    title = "Spontaneous gestures encoded by hand positions improve language models: An Information-Theoretic motivated study",
    author = "Xu, Yang  and
      Cheng, Yang",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.600",
    pages = "9409--9424"
}
```

[COLING 2022: Gestures Are Used Rationally: Information Theoretic Evidence from Neural Sequential Models](https://aclanthology.org/2022.coling-1.12/)
```
@inproceedings{xu-etal-2022-gestures,
    title = "Gestures Are Used Rationally: Information Theoretic Evidence from Neural Sequential Models",
    author = "Xu, Yang  and
      Cheng, Yang  and
      Bhatia, Riya",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.12",
    pages = "134--140"
}
```


## Meta data
Included in the folder `metadata`

## Download videos/transcripts
Run the script `dl_videos.py` in folder `dl_youtube`, which is adapted from the repo [video-processing](https://github.com/AliaksandrSiarohin/video-preprocessing)
```
cd dl_youtube
python dl_videos.py --video_folder PATH_TO_OUTPUT
```

## Step 1. Body key points extraction
Run the script `extract_body_kp.py` in folder `processing` to process multiple videos in batch:
```
python extract_body_kp.py --input_dir YOUR_INPUT_PATH --output_dir YOUR_OUTPUT_PATH
```
Or, you can change `--input_dir` to `--input`, and `--output_dir` to `--output`, if you just want to process a single video file.

Use the `--annotate` argument to generate an output video with all 33 key points annotated. 
For example, after you run
```
python extract_body_kp.py --input foo.mp4 --annotated
```
There will be a video file named `foo_annotated.mp4` saved to the folder, which looks like this:

![foo.mp4](images/test_annotated_large.gif)

The main output file is in `.pkl` format, which is a serialized object of type `List[Tuple]`. Each `Tuple` object has two items: `(frame_index: int, poses: np.ndarray)`. Here `frame_index` is the index of video frame (from 0 to the maximum length -1) that contains a valid detected human body, so for frames where no human body is detected, their indices are skipped. `poses` is a numpy array of shape `(33, 3)`, in which `33` is the number of key points, and `3` stands for the `x`, `y`, and `z` coordinates. 

For detailed explanations on the format of the key points coordinates in the output, please refer to the APIs of [mediapipe](https://google.github.io/mediapipe/solutions/pose.html).


## Step 2. Encode gestures (continuous position => discrete label)
Encode gestures based on hand positions.
Using the key points coordinates returned from step 1, encode the gesture in each frame using the spatial information. 

Run the script `encode_gesture_grid.py` in folder `processing`:

## Step 3. Sample gestures (word sequence => gesture sequence)