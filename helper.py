import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
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


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

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


def gen_test_output(sess, logits, keep_prob, image_pl, data_path, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_path: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    from labels import labels, trainId2label

    for image_file in tqdm(glob(data_path)):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        result_im = scipy.misc.toimage(image)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        num_classes = im_softmax[0].shape[1]
        softmax_result = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes)
        #im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        for label in range(num_classes):
            segmentation = (softmax_result[:,:,label] > 0.5).reshape(image_shape[0], image_shape[1], 1)
            color = trainId2label[label].color
            mask = np.dot(segmentation, np.array([color + (127,)]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            result_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(result_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)
    for name, image in tqdm(image_outputs):
        scipy.misc.imsave(os.path.join(output_dir, name), image)
    return output_dir