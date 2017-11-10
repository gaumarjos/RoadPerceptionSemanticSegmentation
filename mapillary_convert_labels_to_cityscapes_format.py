import numpy as np
import os.path
import re
import os
import time
import glob
import cv2
from collections import deque
import mapillary_labels


def apply_id(labelled_image):
    r, g, b = labelled_image[:,:,2], labelled_image[:,:,1], labelled_image[:,:,0]  # Untested, just make sure you use BGR is loading images with cv2
    bw_image = np.zeros((labelled_image.shape[0], labelled_image.shape[1]), dtype=np.uint8)
    for label_id, label in enumerate(mapillary_labels.labels):
        r1, g1, b1 = label.color_prepr
        mask = (r == r1) & (g == g1) & (b == b1)
        bw_image[mask] = label.trainId
        if 0:
            print("    RGB {:3}, {:3}, {:3} --> LABEL {:3}    {}".format(r1, g1, b1,
                                                                         label.trainId,
                                                                         label.name))
    return bw_image


def process_folder(images_input_path_pattern,  # where to find the images
                   instances_input_path,       # where to find the b/w instances
                   images_output_path,         # where to save the processed images
                   instances_output_path):     # where to save the b/w processed instances

    # Find images to convert (start from the images and then check if the corresponding instance exists)
    image_paths = glob.glob(images_input_path_pattern)
    instance_paths = []
    for path in image_paths:
        proposed = instances_input_path + os.path.splitext(os.path.basename(path))[0] + ".png"
        assert os.path.isfile(proposed)
        instance_paths.append(proposed)

    # Used to estimate the remaining time
    processing_times = deque([], maxlen=10)

    for i, image_file in enumerate(image_paths):
        start_time = time.time()

        # Input/output filenames
        instance_file = instance_paths[i]
        output_image_file = images_output_path + os.path.splitext(os.path.basename(image_file))[0] + '_image.png'
        output_instance_file = instances_output_path + os.path.splitext(os.path.basename(image_file))[0] + '_instance.png'

        # Check if these output files already exists
        if os.path.isfile(output_image_file) and os.path.isfile(output_instance_file):
            print("({}/{}) Pair {} - {} already exists, skip.".format(i+1, len(image_paths),
                                                                      output_image_file,
                                                                      output_instance_file))

        else:
            # Read images
            image = cv2.imread(image_file)
            instance = cv2.imread(instance_file)
            assert image.shape[0] == instance.shape[0] and image.shape[1] == instance.shape[1] and image.shape[2] == 3

            print("({}/{}) Processing the pair {} ({}) - {} ({})...".format(i+1, len(image_paths),
                                                                            image_file, image.shape,
                                                                            instance_file, instance.shape))

            # Crop both image and gt to be the desired_ratio
            h = image.shape[0]
            w = image.shape[1]
            ratio = w/h
            # find_horizon(image, output_stats_file)
            if ratio > desired_ratio:
                tocrop = int(w - h * desired_ratio)
                tocrop_left = int(0.5 * tocrop)
                w = w - tocrop
                #print("Need to crop horizontally, {}px per side".format(tocrop))
                image = image[0:h, tocrop_left:tocrop_left+w]
                instance = instance[0:h, tocrop_left:tocrop_left+w]
            elif ratio < desired_ratio:
                tocrop = int(h - w / desired_ratio)
                tocrop_top = int(desired_top_crop_ratio * tocrop)
                h = h - tocrop
                #print("Need to crop vertically, {}px per side".format(tocrop))
                image = image[tocrop_top:tocrop_top+h, 0:w]
                instance = instance[tocrop_top:tocrop_top+h, 0:w]
            else:
                pass
                #print("No need to crop")

            # Resize it to be desired_h * desired_w
            image_res = cv2.resize(image, (desired_w, desired_h), interpolation=cv2.INTER_NEAREST)
            instance_res = cv2.resize(instance, (desired_w, desired_h), interpolation=cv2.INTER_NEAREST)
            #image_res = image
            #instance_res = instance

            # Create instance --> not used as we're working directly with instances
            # instance_res = apply_id(label_res)

            # Make sure they're 8 bit
            image_res = image_res.astype(dtype=np.uint8)
            instance_res = instance_res.astype(dtype=np.uint8)

            # Save both the resized image and the gt image with B/W labelling
            cv2.imwrite(output_image_file, image_res)
            cv2.imwrite(output_instance_file, instance_res)

            # Compute processing duration
            duration = time.time() - start_time
            processing_times.append(duration)
            avg_duration = np.mean(np.asarray(processing_times))
            remaining_time = int((len(image_paths) - (i+1)) * avg_duration)
            remaining_time_m, remaining_time_s = divmod(remaining_time, 60)
            remaining_time_h, remaining_time_m = divmod(remaining_time_m, 60)
            print("Remaining time: {:02d}:{:02d}:{:02d}".format(remaining_time_h, remaining_time_m, remaining_time_s))


if __name__ == '__main__':
    #desired_h = 1024
    #desired_w = 2048
    desired_h = 256
    desired_w = 512
    desired_ratio = desired_w / desired_h
    desired_top_crop_ratio = 0.66  # 1 = crop only top, 0 = crop only bottom, 0.5 = equal
    assert desired_top_crop_ratio <= 1.0 and desired_top_crop_ratio >= 0.0

    images_input_path_pattern = '../mapillary/data/training/images/*.jpg'
    instances_input_path = '../mapillary/data/training/instances/'
    images_output_path = '../mapillary/data/training/images_processed/'
    instances_output_path = '../mapillary/data/training/instances_processed/'
    process_folder(images_input_path_pattern,
                   instances_input_path,
                   images_output_path,
                   instances_output_path)

