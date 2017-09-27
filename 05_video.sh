#!/bin/bash
python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=project_video.mp4 --video_file_out=segmented.mp4

