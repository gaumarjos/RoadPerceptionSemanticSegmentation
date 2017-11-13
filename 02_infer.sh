#!/bin/bash
#python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=project_video.mp4 --video_file_out=segmented.mp4

#python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/20170905_DrivePX/quick_test01.mp4 --video_start_second 0 --video_end_second 60 --video_file_out=../videos/20170905_DrivePX/quick_test01_segmented_2.mp4
#python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/20170905_DrivePX/quick_test02.mp4 --video_file_out=../videos/20170905_DrivePX/quick_test02_segmented.mp4


# Model freezing and optimization
#python main.py freeze --ckpt_dir=ckpt/Training_B --frozen_model_dir=frozen_model
#python main.py optimise --frozen_model_dir=frozen_model --optimised_model_dir=optimised_model

# Video segmentation
#python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/20170905_DrivePX/dw_20170905_123726_0.000000_0.000000/video_1.h264_undist.mp4 --video_start_second 0 --video_end_second 10 --video_file_out=../videos/20170905_DrivePX/dw_20170905_123726_0.000000_0.000000/video_1.h264_undist_segmentedB_intermediate_10s.mp4


# Training A
#python main.py freeze --ckpt_dir=ckpt/Training_A --frozen_model_dir=frozen_model
#python main.py optimise --frozen_model_dir=frozen_model --optimised_model_dir=optimised_model
#python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/20170905_DrivePX/dw_20170905_123726_0.000000_0.000000/video_1.h264_undist.mp4 --video_file_out=../videos/20170905_DrivePX/dw_20170905_123726_0.000000_0.000000/video_1.h264_undist_segmentedA.mp4


# Urban Cologne videos
# TRAINING="Training_A"
TRAINING=""
IMAGES=1
VIDEO=0

MASTERVIDEOFOLDER="20171103_DrivePX_City"
VIDEOFOLDER="dw_20171102_181952_0.000000_0.000000"
# VIDEONAME="video_1.h264_undist"
VIDEONAME="video_4.h264"

EXTENSION="mp4"
SUFFIX="_segmented"
VIDEOFILEINPUT=$VIDEONAME.$EXTENSION
VIDEOFILEOUTPUT=$VIDEONAME$SUFFIX.$EXTENSION

python main.py freeze --ckpt_dir=ckpt/$TRAINING --frozen_model_dir=frozen_model
python main.py optimise --frozen_model_dir=frozen_model --optimised_model_dir=optimised_model

if [ $IMAGES -eq 1 ]
then
  echo "Processing images in the test set"
  python main.py predict --gpu=1 --xla=2 --model_dir=optimised_model
fi

if [ $VIDEO -eq 1 ]
then
  echo "Processing $VIDEOFILEINPUT --> $VIDEOFILEOUTPUT"
  python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/$MASTERVIDEOFOLDER/$VIDEOFOLDER/$VIDEOFILEINPUT --video_file_out=../videos/$MASTERVIDEOFOLDER/$VIDEOFOLDER/$VIDEOFILEOUTPUT
fi

