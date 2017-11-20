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
TRAINING="mapillary_training_b"
IMAGES=0
VIDEO=1
OPTIMISE=0

MASTERVIDEOFOLDER="20171103_DrivePX_City"
:'
VIDEOFOLDERS=( "dw_20171102_182846_0.000000_0.000000"     # Aachener Str. direction centre
               "dw_20171102_185404_0.000000_0.000000"     # Crossing cyclist
               "dw_20171102_185840_0.000000_0.000000" )   # Small neighbourhood
'
VIDEOFOLDERS=( "dw_20171102_180413_0.000000_0.000000"     # Durener
               "dw_20171102_181333_0.000000_0.000000"     # Small neighbourhood
               "dw_20171102_183737_0.000000_0.000000")    # Deutz bridge
VIDEONAME="video_4.h264_256x512"

EXTENSION="mp4"
SUFFIX="_segmented_mapillary_b"
VIDEOFILEINPUT=$VIDEONAME.$EXTENSION
VIDEOFILEOUTPUT=$VIDEONAME$SUFFIX.$EXTENSION

if [ $OPTIMISE -eq 1 ]
then
  python main.py freeze --ckpt_dir=ckpt/$TRAINING --frozen_model_dir=frozen_model
  python main.py optimise --frozen_model_dir=frozen_model --optimised_model_dir=optimised_model
fi

if [ $IMAGES -eq 1 ]
then
  echo "Processing images in the test set"
  python main.py predict --gpu=1 --xla=2 --model_dir=optimised_model
fi

if [ $VIDEO -eq 1 ]
then
  for VIDEOFOLDER in "${VIDEOFOLDERS[@]}"
  do
    echo "[$VIDEOFOLDER] processing $VIDEOFILEINPUT --> $VIDEOFILEOUTPUT"
    python main.py video --gpu=1 --xla=2 --model_dir=optimised_model --video_file_in=../videos/$MASTERVIDEOFOLDER/$VIDEOFOLDER/$VIDEOFILEINPUT --video_file_out=../videos/$MASTERVIDEOFOLDER/$VIDEOFOLDER/$VIDEOFILEOUTPUT
    # --video_start_second 0 --video_end_second 60
  done
fi

#
