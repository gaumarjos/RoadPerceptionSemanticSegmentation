import scipy
import scipy.misc
import numpy as np
from PIL import Image
import os.path
import re
import os
import warnings
from distutils.version import LooseVersion
import shutil
import time
import glob
import mapillary_labels
import cv2

images_path_pattern = '../mapillary/data/train/*.jpg'
labels_path_pattern = '../mapillary/data/train/*.png'

desired_h = 1024
desired_w = 2048
desired_ratio = desired_w / desired_h


def change_color(fromimage, toimage, fromcolor, tocolor):
    r1, g1, b1 = fromcolor  # Original value
    red, green, blue = fromimage[:,:,0], fromimage[:,:,1], fromimage[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    toimage[mask] = tocolor
    # r2, g2, b2 = tocolor    # Value that we want to replace it with
    # toimage[:,:,:3][mask] = [r2, g2, b2]
    return np.sum(mask)


image_paths = glob.glob(images_path_pattern)
label_paths = {re.sub('png', 'jpg', os.path.basename(path)): path for path in glob.glob(labels_path_pattern)}
assert len(image_paths) == len(label_paths)

for i, image_file in enumerate(image_paths):
    start_time = time.time()
    
    # Output filenames
    output_image_file = os.path.splitext(image_file)[0] + '_image.png'
    gt_image_file = label_paths[os.path.basename(image_file)]
    output_gt_image_file = os.path.splitext(gt_image_file)[0] + '_gt.png'

    # Read images
    image = scipy.misc.imread(image_file)
    gt_image = scipy.misc.imread(gt_image_file)
    
    print("({}/{}) Processing {} ({})...".format(i+1, len(image_paths), gt_image_file, gt_image.shape))

    # Crop both image and gt to be the desired_ratio
    h = gt_image.shape[0]
    w = gt_image.shape[1]
    ratio = w/h
    if ratio > desired_ratio:
        tocrop = int((w - h * desired_ratio) / 2)
        w = w - 2 * tocrop
        #print("Need to crop horizontally, {}px per side".format(tocrop))
        image    =    image[0:h, tocrop:tocrop+w]
        gt_image = gt_image[0:h, tocrop:tocrop+w]
    elif ratio < desired_ratio:
        tocrop = int((h - w / desired_ratio) / 2)
        h = h - 2 * tocrop
        #print("Need to crop vertically, {}px per side".format(tocrop))
        image    =    image[tocrop:tocrop+h, 0:w]
        gt_image = gt_image[tocrop:tocrop+h, 0:w]
    else:
        pass
        #print("NO need to crop")
    #print(image.shape)
    #print(gt_image.shape)

    # Resize it to be desired_h * desired_w
    image_res = cv2.resize(image, (desired_w, desired_h), interpolation=cv2.INTER_NEAREST)
    gt_image_res = cv2.resize(gt_image, (desired_w, desired_h), interpolation=cv2.INTER_NEAREST)

    # Create gt empty output
    gt_image_bw = np.zeros((gt_image_res.shape[0], gt_image_res.shape[1]), dtype=np.uint8)

    # Scroll through all possible labels and paint the output accordingly
    num_classes = len(mapillary_labels.labels)
    for i in range(num_classes):
        fromcolor = mapillary_labels.labels[i].color
        tocolor = mapillary_labels.labels[i].trainId
        n_changed_px = change_color(gt_image_res, gt_image_bw, fromcolor, tocolor)
        print("    ({:5.2f}%)    RGB {:3}, {:3}, {:3} --> LABEL {:3}    {}".format(n_changed_px/gt_image_bw.size*100,
                                                                            fromcolor[0], fromcolor[1], fromcolor[2],
                                                                            tocolor,
                                                                            mapillary_labels.labels[i].name))

    # Save both the resized image and the gt image with B/W labelling
    scipy.misc.imsave(output_image_file, image_res)
    scipy.misc.imsave(output_gt_image_file, gt_image_bw)

    # Just check
    check = scipy.misc.imread(output_gt_image_file)
    # assert np.max(gt_image_bw) == np.max(check)
    
    # Compute processing duration
    duration = time.time() - start_time
    print("Duration: {:4.1f}s".format(duration))

