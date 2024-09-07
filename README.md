# Life-lessons
A dataset of first-person monologue videos/transcript/annotations about "life lessons" in various domains. The main purpose is for multi-modal language analysis and modeling.

## Published papers
[Findings of ACL 2024: How Much Does Nonverbal Communication Conform to Entropy Rate Constancy?: A Case Study on Listener Gaze in Interaction](https://aclanthology.org/2024.findings-acl.210/)
```
@inproceedings{wang-etal-2024-much,
    title = "How Much Does Nonverbal Communication Conform to Entropy Rate Constancy?: A Case Study on Listener Gaze in Interaction",
    author = "Wang, Yu  and
      Xu, Yang  and
      Skantze, Gabriel  and
      Buschmeier, Hendrik",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.210",
    pages = "3533--3545"
}
```

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
Run script `dl_videos.py` in `dl_youtube` folder, which is adapted from the repo [video-processing](https://github.com/AliaksandrSiarohin/video-preprocessing)
```
cd dl_youtube
python dl_videos.py --video_folder PATH_TO_OUTPUT
```

## Step 1. Body key points extraction
Run `extract_body_kp.py` in `processing` folder to process multiple videos in batch:
```
python extract_body_kp.py --input_dir YOUR_INPUT_PATH --output_dir YOUR_OUTPUT_PATH
```
Or change `--input_dir` to `--input`, and `--output_dir` to `--output`, for processing a single video file.

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

Run script `encode_gesture_grid.py` in `processing` folder:
```
python encode_gesture_grid.py --input_dir YOUR_INPUT_PATH --output_dir YOUR_OUTPUT_PATH
```
Use argument `--return_type` to select two reutrn types, `'df'` or `'list'`. By default, `--return_type='df'`, and the output format is `.csv`. When `--return_type='list'` is specified, the corresponding output format is `.pkl`, for storing a serialized Python list of tuples. 

The detailed output formats are as follows:
- `--return_type='df'`: output is saved to an $$M\times 4$$ csv file, or a `.pkl` file if `--return_grids` is set. In the output csv file, each row corresponds to a annotated frame. The first column is the frame index (an integer starting from 0). The remaining three columns are the left-hand token, right-hand token, and the overall gesture token, respectively. 
- `--return_type='list'`: output is saved as a serialized Python pickle object that contains one or two lists, depending on whether `--return_grids` is set or not.
- `--return_grids`: If not set, the returned list is a list of 4-element tuples, containing the frame index and three gesture tokens for a frame (similar to the 4 columns described above); If set, then the first returned list is the same, and the second list is a list of `(x,y)` coordinates of all grids. 

## Step 3. Sample gestures (word sequence => gesture sequence)
Run script `get_gesture_seqs.py` in `processing` folder:
```
python get_gesture_seqs.py --input_gesture_dir YOUR_GESTURE_PATH \
      --input_vtt_dir YOUR_VTT_PATH \
      --output_word_dir YOUR_WORD_PATH \
      --output_gesture_dir YOUR_GESTURE_PATH \
      --output_mixed_dir YOUR_MIXED_PATH
```
TBD