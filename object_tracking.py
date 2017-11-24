import os.path
import os
import warnings
from distutils.version import LooseVersion
import shutil
import time
import argparse
import glob
import re
import random
from timeit import default_timer as timer
import math
import pickle
import cv2
import scipy.ndimage.measurements as scipymeas
from collections import deque


import scipy.misc
import scipy.io as sio
import numpy as np

class Tracker():
    def __init__(self, label_nr):
        # Label
        self.label_nr = label_nr

        # Best heatmap
        self.heatmap = None

        # Heatmaps FIFO length
        self.heatmap_fifo_length = 3

        # Heatmaps FIFO
        self.heatmap_fifo = deque(maxlen=self.heatmap_fifo_length)

        # Centroid FIFO (list of lists)
        # self.centroid_fifo = deque(maxlen=self.heatmap_fifo_length)

        # Gaussian blur kernel size
        # self.blur_kernel = 3

        # Threshold for heatmap
        self.threshold = 2

        assert self.heatmap_fifo_length > self.threshold

    def update_heatmap(self, this_frame_heatmap):
        # Keep that in memory
        self.heatmap_fifo.append(np.asarray(this_frame_heatmap, dtype=np.int32))

        # Update the current heatmap with all the elements from the heatmap history
        self.heatmap = np.sum(np.array(self.heatmap_fifo), axis=0)
        self.heatmap[self.heatmap < self.threshold] = 0
        
