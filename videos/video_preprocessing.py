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
    crop_left = 280
    crop_right = 1440+crop_left
    crop_top = 0
    crop_bottom = 896
    desired_w = 1440
    desired_h = 896

    img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]
    img_resized = cv2.resize(img_cropped, (desired_w, desired_h), interpolation=cv2.INTER_LINEAR)
    
    return img_resized


def extract_frames_from_video(folder, frame_secs, left_basename, right_basename, i_start=1, output_folder="", preprocess=False):
    left_clip = VideoFileClip(folder + left_basename + ".mp4")
    right_clip = VideoFileClip(folder + right_basename + ".mp4")

    duration = np.min([left_clip.duration, right_clip.duration])
    print("{} frames will be extracted, the video is {}s long".format(len(frame_secs), duration))

    for i, t in enumerate(frame_secs):
        i_label = i
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



folder = '20171201_stereo_TMG/'
# car2
# toskip = [5, 7, 8, 9, 10, 13, 27, 28, 29, 30, 38, 40, 41, 42, 43, 44]
toskip = []

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
# car 3
calibration_frame_secs = [1, 6, 9, 17, 22, 25, 32, 34, 47, 58, 60+8, 60+13, 60+20, 60+23, 60+30, 60+50, 60+55, 60+58, 120+1, 120+10, 120+13, 120+16, 120+18]
test_frame_secs = [44, 50, 60+5, 60+27, 60+45, 60+47, 120+37, 120+42, 120+49, 180+47, 180+49, 180+59, 240+8, 240+32, 240+41]

if 1:
    extract_frames_from_video(folder,
                              calibration_frame_secs,
                              left_basename='calibration_left',
                              right_basename='calibration_right',
                              i_start=1,
                              output_folder='calibration_frames/',
                              preprocess=1)
                              
    extract_frames_from_video(folder,
                              test_frame_secs,
                              left_basename='test_left',
                              right_basename='test_right',
                              i_start=1,
                              output_folder='test_frames/',
                              preprocess=1)

if 0:
    preprocess_video(folder,
                     left_basename='test_left',
                     right_basename='test_right')

