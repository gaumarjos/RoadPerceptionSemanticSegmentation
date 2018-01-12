import numpy as np
import cv2
import os
from os.path import basename
import os.path
import glob
import time
import pickle
from collections import deque
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tqdm
from moviepy.editor import VideoFileClip

import tty
import sys
import termios


def crop_and_resize(img):
    desired_w = 1440
    desired_h = 896

    crop_left = 280
    crop_right = desired_w + crop_left
    crop_top = 65
    crop_bottom = desired_h + crop_top

    img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]
    img_resized = cv2.resize(img_cropped, (desired_w, desired_h), interpolation=cv2.INTER_LINEAR)
    
    return img_resized


def extract_frames_from_video(folder, input_folder, left_basename, right_basename, frame_secs, i_start=1, output_folder="", preprocess=False):
    left_clip = VideoFileClip(folder + input_folder + left_basename + ".mp4")
    right_clip = VideoFileClip(folder + input_folder + right_basename + ".mp4")

    duration = np.min([left_clip.duration, right_clip.duration])
    print("{} frames will be extracted, the video is {}s long".format(len(frame_secs), duration))

    for i, t in enumerate(frame_secs):
        i_label = i
        print('frame {}'.format(i))
        left_name = folder + output_folder + "{}_{:03d}.png".format(left_basename, i_label+i_start)
        right_name = folder + output_folder  + "{}_{:03d}.png".format(right_basename, i_label+i_start)
        left_clip.save_frame(left_name, t=t)
        right_clip.save_frame(right_name, t=t)
        
        # Create preprocessed version if needed
        if preprocess==1:
            left = cv2.imread(left_name)
            right = cv2.imread(right_name)
            left_cropped = crop_and_resize(left)
            right_cropped = crop_and_resize(right)
            cv2.imwrite(left_name[:-4] + "_cropped.png", left_cropped)
            cv2.imwrite(right_name[:-4] + "_cropped.png", right_cropped)

        if preprocess==2:
            left = cv2.imread(left_name)
            right = cv2.imread(right_name)
            left_resized = cv2.resize(left, (int(1920/2), int(1208/2)), interpolation=cv2.INTER_LINEAR)
            right_resized = cv2.resize(right, (int(1980/2), int(1208/2)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(left_name[:-4] + "_resized.png", left_resized)
            cv2.imwrite(right_name[:-4] + "_resized.png", right_resized)


def preprocess_video(folder, left_basename, right_basename):
    left_clip = VideoFileClip(folder + left_basename + ".mp4")
    right_clip = VideoFileClip(folder + right_basename + ".mp4")

    left_white_clip = left_clip.fl_image(crop_and_resize).subclip(0, 10)
    left_white_output = folder + left_basename + '_preprocessed.mp4'
    left_white_clip.write_videofile(left_white_output, audio=False)

    right_white_clip = right_clip.fl_image(crop_and_resize).subclip(0, 10)
    right_white_output = folder + right_basename + '_preprocessed.mp4'
    right_white_clip.write_videofile(right_white_output, audio=False)




# car 1
"""
frame_secs = [14, 20, 40, 44, 51, 60, 60+7, 60+15, 60+23, 60+29, 60+40, 60+52, 60+55, 120+3, 120+14, 120+18, 120+24, 120+24, 120+42, 120+50, 180, 180+4, 180+11, 180+27, 180+33, 180+40, 180+44]
"""
# car2
"""
frame_secs = [1, 11, 16, 27, 31, 60+1, 60+4, 60+12, 60+16, 60+29, 60+35, 60+45, 60+55, 120+4, 120+13, 120+18, 120+20, 120+23, 120+28,
              120+33, 120+37, 120+40, 120+45, 120+52, 120+55, 120+59, 180+3, 180+10, 180+14, 180+18, 180+22, 180+30, 180+35, 180+41,
              180+44, 180+50, 180+53]
frame_secs = [2, 14, 27, 30, 36, 47, 54, 60+3]
"""

# Relative to folder 20171201_stereo_1st_calibration_at_TMG
#calibration_frame_secs = [1, 6, 9, 17, 22, 25, 32, 34, 47, 58, 60+8, 60+13, 60+20, 60+23, 60+30, 60+50, 60+55, 60+58, 120+1, 120+10, 120+13, 120+16, 120+18]
#test_frame_secs = test_frame_secs = [44, 50, 60+5, 60+27, 60+45, 60+47, 120+37, 120+42, 120+49, 180+47, 180+49, 180+59, 240+8, 240+32, 240+41]

# Relative to folder 20171215_stereo_i3
#calibration_frame_secs = [1, 6, 9, 17, 22, 25, 32, 34, 47, 58, 60+8, 60+13, 60+20, 60+23, 60+30, 60+50, 60+55, 60+58, 120+1, 120+10, 120+13, 120+16, 120+18]
#test_frame_secs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20 ,22, 24, 26, 28, 30, 32 ,34, 36, 38, 40, 42, 44, 46, 48, 50, 52]

if 1:
    master_folder = '20180111_stereo_calibration_60deg_120mm/'
    """
    extract_frames_from_video(master_folder,
                              input_folder='calibration_videos/',
                              left_basename='calibration_left',
                              right_basename='calibration_right',
                              frame_secs=[0, 15, 22, 30, 35, 40, 44, 50, 55, 58, 60+5, 60+18, 60+22, 60+27, 60+33, 60+38, 60+40, 60+42, 60+45, 60+48, 60+50, 60+52, 120+2, 120+6, 120+10, 120+14, 120+18, 120+22, 120+26, 120+30, 120+33, 120+36, 120+40, 120+44, 180+0, 180+4, 180+8, 180+11, 180+15, 180+18, 180+38, 180+45, 180+55, 240+8, 240+15, 240+18, 240+22, 240+25, 240+30, 240+35, 240+46, 300+12],
                              i_start=0,
                              output_folder='calibration_frames/',
                              preprocess=0)

    extract_frames_from_video(master_folder,
                              input_folder='distance_videos/',
                              left_basename='distance_left',
                              right_basename='distance_right',
                              frame_secs=[3, 22, 35, 48, 60+3, 60+19, 60+32, 60+45, 60+59, 120+16, 120+31, 120+47, 180+2, 180+17, 180+31, 180+47, 240+2],
                              i_start=0,
                              output_folder='distance_frames/',
                              preprocess=0)
    """
    extract_frames_from_video(master_folder,
                              input_folder='test_videos/',
                              left_basename='test_left',
                              right_basename='test_right',
                              frame_secs=[3*60+46, 3*60+54, 4*60+53, 10*60+45, 13*60+18, 13*60+19, 13*60+22, 13*60+24, 13*60+38, 13*60+56, 15*60+13, 16*60+17, 19*60+30, 20*60+31, 21*60+4, 21*60+6, 21*60+16, 21*60+27, 21*60+52, 22*60+14, 22*60+57, 24*60+21],
                              i_start=0,
                              output_folder='test_frames/',
                              preprocess=0)

if 0:
    master_folder = '20171220_stereo_calibration_120deg_390mm/'
    i=240
    extract_frames_from_video(master_folder,
                              input_folder='calibration_videos/',
                              left_basename='calibration_left',
                              right_basename='calibration_right',
                              # frame_secs=[3, 9, 18, 34, 38, 41, 48, 52, 54, 56, 60, 60+5, 60+9, 60+15, 60+18, 60+21, 60+24,  60+27, 60+30, 60+33, 60+36, 60+41, 60+44, 60+47, 60+50, 60+53, 60+56, 60+60, 60+64, 60+70, 60+74, 60+77, 60+83, 60+86, 60+90, 60+93, 60+98, 60+103, 60+107, 180+8, 180+12, 180+16, 180+20, 180+26, 180+29, 180+32, 180+39, 180+42, 180+47, 180+50], # BIG BOARD
                              frame_secs=[i+27, i+31, i+34, i+37, i+40, i+46, i+49, i+54, i+57, i+60, i+63, i+65, i+68, i+71, i+74, i+77, i+79, i+82, i+86, i+90, i+94, i+96, i+99, i+103, i+106, i+109, i+113, i+120, i+124, i+127, i+129, i+138, i+140, i+146, i+149, i+151, i+153, i+165], # schmall BOARD
                              i_start=1,
                              output_folder='calibration_frames_small/',
                              preprocess=1)

    extract_frames_from_video(master_folder,
                              input_folder='distance_indoor_videos/',
                              left_basename='distance_indoor_left',
                              right_basename='distance_indoor_right',
                              frame_secs=[60+16, 60+39, 60+52, 120+14, 120+32, 120+51, 180+10, 180+27, 180+44, 180+56, 240+14, 240+30, 240+46, 300+3],
                              i_start=1,
                              output_folder='distance_indoor_frames/',
                              preprocess=1)

    extract_frames_from_video(master_folder,
                              input_folder='distance_outdoor_videos/',
                              left_basename='distance_outdoor_left',
                              right_basename='distance_outdoor_right',
                              frame_secs=[12, 24, 38, 56, 60+8, 60+24, 60+41, 60+59, 120+14, 120+30, 180+10, 180+27, 180+43, 180+59, 240+22, 240+41, 300+3, 300+31, 420+2],
                              i_start=1,
                              output_folder='distance_outdoor_frames/',
                              preprocess=1)

if 0:
    preprocess_video(folder,
                     left_basename='test_left',
                     right_basename='test_right')

