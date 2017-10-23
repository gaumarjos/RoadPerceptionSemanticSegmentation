#!/bin/bash
python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=project_video.mp4 --video_file_out=segmented.mp4

python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/20170905_DrivePX/quick_test01.mp4 --video_start_second 0 --video_end_second 60 --video_file_out=../videos/20170905_DrivePX/quick_test01_segmented_2.mp4
python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/20170905_DrivePX/quick_test02.mp4 --video_file_out=../videos/20170905_DrivePX/quick_test02_segmented.mp4
