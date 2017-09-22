import re
import random
import os.path
import shutil
import zipfile
import time
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))



def gen_batch_function_cityscapes(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size, n_samples_max=None):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'leftImg8bit/train/*/*_leftImg8bit.png'))
        label_paths = {
            re.sub('_gtFine_labelTrainIds', '_leftImg8bit', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gtFine/train/*/*_gtFine_labelTrainIds.png'))}
        #background_color = np.array([0, 0, 0])
        num_classes = 35 # 0..34

        random.shuffle(image_paths)
        count = 0
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_image = gt_image.reshape(*gt_image.shape, 1)
                tmp = []
                for label in range(num_classes):
                  color = np.array([label, label, label])
                  gt_bg = np.all(gt_image == color, axis=2)
                  gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                  tmp.append(gt_bg)
                gt_image = np.concatenate(tmp, axis=2)

                images.append(image)
                gt_images.append(gt_image)
                count += 1
                if n_samples_max is not None and n_samples_max == count:
                    break

            yield np.array(images), np.array(gt_images)
            if n_samples_max is not None and n_samples_max == count:
                break
    return get_batches_fn


