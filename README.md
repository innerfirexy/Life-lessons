# Life-lessons
A dataset of first-person monologue videos/transcript/annotations about "life lessons" in various domains. The main purpose is for multi-modal language analysis and modeling.

## Meta data
Included in the folder `metadata`

## Download videos/transcripts
Run the script `dl_videos.py` in folder `dl_youtube`, which is adapted from the repo [video-processing](https://github.com/AliaksandrSiarohin/video-preprocessing)
```
cd dl_youtube
python dl_videos.py --video_folder PATH_TO_OUTPUT
```

## Body key points extraction
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

The main output file is in `.pkl` format, which is a serialized object of type `List[Tuple]`. Each `Tuple` object has two items: `(frame_index: int, poses: np.ndarray)`. 


## Label gestures (tokenize)

## Sample gestures (word => gesture mapping)