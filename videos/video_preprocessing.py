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

"""
def imread_crop(filename, ratio):
    img = cv2.imread(filename)
    x = img.shape[0]
    y = img.shape[1]
    newx = int((1.-ratio) * x / 2)
    newy = int((1.-ratio) * y / 2)
    img_cropped = img[newx:x-newx, newy:y-newy, :]
    return img_cropped
"""
    
    
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
        if preprocess:
            left = cv2.imread(left_name)
            right = cv2.imread(right_name)
            left_cropped = crop_and_resize(left)
            right_cropped = crop_and_resize(right)
            cv2.imwrite(left_name[:-4] + "_cropped.png", left_cropped)
            cv2.imwrite(right_name[:-4] + "_cropped.png", right_cropped)


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

# Relative to folder 20171201_stereo_2nd_calibration_at_TMG
if 1:
    master_folder = '20180109_stereo_60_calibration/'

    """
    extract_frames_from_video(master_folder,
                              input_folder='calibration_videos/',
                              left_basename='calibration_left',
                              right_basename='calibration_right',
                              frame_secs=[0, 7, 11, 15, 21, 32, 37, 46, 49, 60+1, 60+15, 60+20, 60+23, 60+28, 60+34, 60+39, 60+43, 60+47, 60+52, 60+58, 60+59, 120+6, 120+12, 120+15, 120+19, 120+25, 120+30, 120+32, 120+37, 120+40, 120+43, 120+53, 120+56, 180+0, 180+4, 180+8, 180+12, 180+16, 180+23, 180+24, 180+26, 180+31, 180+37, 180+39,180+41, 180+43, 180+45, 180+47, 180+52, 180+57, 180+59, 240+1, 240+5, 240+7, 240+9, 240+11, 240+13, 240+15, 240+17, 240+19, 240+21, 240+23, 240+25, 240+27, 240+29, 240+35, 240+39, 240+42, 240+44, 240+46, 240+48, 240+55], # BIG BOARD Closer from 180+23
                              i_start=1,
                              output_folder='calibration_frames/',
                              preprocess=0)
    """

    extract_frames_from_video(master_folder,
                              input_folder='distance_videos/',
                              left_basename='distance_left',
                              right_basename='distance_right',
                              frame_secs=[1, 9, 18, 32, 44, 58, 1*60+13, 1*60+32, 1*60+57, 2*60+10, 3*60+40, 3*60+52, 4*60+5, 4*60+36, 4*60+54, 5*60+12, 6*60+18, 7*60+22, 7*60+44, 8*60+0, 8*60+17, 8*60+32, 8*60+40, 8*60+51, 9*60+1, 9*60+10, 9*60+24],
                              i_start=0,
                              output_folder='distance_frames/',
                              preprocess=0)

"""
    master_folder = '20171220_stereo_2nd_calibration_at_TMG/'
    i=240
    extract_frames_from_video(master_folder,
                              input_folder='calibration_videos/',
                              left_basename='calibration_left',
                              right_basename='calibration_right',
                              #frame_secs=[3, 9, 18, 34, 38, 41, 48, 52, 54, 56, 60, 60+5, 60+9, 60+15, 60+18, 60+21, 60+24,  60+27, 60+30, 60+33, 60+36, 60+41, 60+44, 60+47, 60+50, 60+53, 60+56, 60+60, 60+64, 60+70, 60+74, 60+77, 60+83, 60+86, 60+90, 60+93, 60+98, 60+103, 60+107, 180+8, 180+12, 180+16, 180+20, 180+26, 180+29, 180+32, 180+39, 180+42, 180+47, 180+50], # BIG BOARD
                              #frame_secs=[i+27, i+31, i+34, i+37, i+40, i+46, i+49, i+54, i+57, i+60, i+63, i+65, i+68, i+71, i+74, i+77, i+79, i+82, i+86, i+90, i+94, i+96, i+99, i+103, i+106, i+109, i+113, i+120, i+124, i+127, i+129, i+138, i+140, i+146, i+149, i+151, i+153, i+165], # schmall BOARD
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
"""

if 0:
    preprocess_video(folder,
                     left_basename='test_left',
                     right_basename='test_right')

