import os.path
import warnings
from distutils.version import LooseVersion
import shutil
import time
import argparse
import glob
import re
import random

import tensorflow as tf
from tqdm import tqdm
import scipy.misc
import numpy as np

import helper
import cityscape_labels
import fcn8vgg16

"""Semantic Segmentation with Fully Convolutional Networks

Architecture as in https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

Trained on Citiscapes data https://www.cityscapes-dataset.com/
"""



def load_trained_vgg_vars(sess):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :return: dict name/value where value is pre-trained array
    """
    # Download pretrained vgg model
    vgg_path = 'pretrained_vgg'
    helper.maybe_download_pretrained_vgg(vgg_path)
    # load model
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # extract variables
    graph = tf.get_default_graph()
    variables = [op for op in graph.get_operations() if op.op_def and op.op_def.name[:5] == 'Varia']
    # filter out relevant variables and change names
    var_values = {}
    for var in variables:
        name = var.name
        tensor = tf.get_default_graph().get_tensor_by_name(name + ':0')
        value = sess.run(tensor)
        name = name.replace('filter', 'weights')
        name = name.replace('fc', 'conv')
        var_values[name] = value
    return var_values


def get_train_batch_generator_cityscapes(images_path_pattern, labels_path_pattern, image_shape):
    """
    Generate function to create batches of training data
    :param images_path_pattern: path pattern for images
    :param labels_path_pattern: path pattern for labels
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = glob.glob(images_path_pattern)
    label_paths = {re.sub('_gtFine_labelTrainIds', '_leftImg8bit', os.path.basename(path)): path
                        for path in glob.glob(labels_path_pattern)}
    num_classes = len(cityscape_labels.labels)
    num_samples = len(image_paths)
    assert len(image_paths) == len(label_paths)

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        random.shuffle(image_paths)
        count = 0
        for batch_i in range(0, num_samples, batch_size):
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

            yield np.array(images), np.array(gt_images)
    return get_batches_fn, num_samples


def train(args, image_shape):
    # tensorflow GPU config
    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': args.gpu})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem

    # extract pre-trained VGG weights
    with tf.Session(config=config) as sess:
        var_values = load_trained_vgg_vars(sess)
    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        # define our FCN
        images_shape = (None,)+image_shape+(3,)
        num_classes = len(cityscape_labels.labels)
        labels_shape = (None,)+image_shape+(num_classes,)
        model = fcn8vgg16.FCN8_VGG16(images_shape, labels_shape)

        # variables initialization
        sess.run(tf.global_variables_initializer())
        model.restore_variables(sess, var_values)

        # Create batch generator
        # TODO: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        train_batches_fn, num_samples = get_train_batch_generator_cityscapes(args.images_paths,
                                                                             args.labels_paths,
                                                                             image_shape)

        run_name = "/ep{}_b{}_lr{:.6f}_kp{}".format(args.epochs, args.batch_size, args.learning_rate, args.keep_prob)
        start_time = time.time()

        final_loss = model.train(sess, args.epochs, args.batch_size,
                                 train_batches_fn, num_samples,
                                 args.keep_prob, args.learning_rate,
                                 args.ckpt_dir, args.summary_dir+run_name)

        # Make folder for current run
        output_dir = os.path.join(args.runs_dir, time.strftime("%Y%m%d_%H%M%S"))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # save training details to text file
        with open(os.path.join(output_dir, "params.txt"), "w") as f:
            f.write('keep_prob={}\n'.format(args.keep_prob))
            f.write('images_paths={}\n'.format(args.images_paths))
            f.write('num_samples={}\n'.format(num_samples))
            f.write('batch_size={}\n'.format(args.batch_size))
            f.write('epochs={}\n'.format(args.epochs))
            f.write('gpu={}\n'.format(args.gpu))
            f.write('gpu_mem={}\n'.format(args.gpu_mem))
            f.write('learning_rate={}\n'.format(args.learning_rate))
            f.write('final_loss={}\n'.format(final_loss))
            duration = time.time() - start_time
            f.write('total_time_hrs={}\n'.format(duration/3600))

        # save model
        """ save trained model using SavedModelBuilder """
        if args.model_dir is None:
            model_dir = os.path.join(output_dir, 'model')
        else:
            model_dir = args.model_dir
        print('saving trained model to {}'.format(model_dir))
        model.save_model(sess, model_dir)



def predict(args, image_shape):
    # tensorflow GPU config
    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': args.gpu})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem

    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        # define our FCN
        images_shape = (None,)+image_shape+(3,)
        num_classes = len(cityscape_labels.labels)
        labels_shape = (None,)+image_shape+(num_classes,)
        model = fcn8vgg16.FCN8_VGG16(images_shape, labels_shape)

        # load saved model
        if args.model_dir is None:
            model_dir = 'trained_model'
        else:
            model_dir = args.model_dir
        model.load_model(sess, model_dir)

        # Make folder for current run
        output_dir = os.path.join(args.runs_dir, time.strftime("%Y%m%d_%H%M%S"))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print('Predicting on test images {} to: {}'.format(args.images_paths, output_dir))

        transparency_level = 56
        for image_file in tqdm(glob.glob(args.images_paths)):
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            result_im = scipy.misc.toimage(image)
            predicted_class = model.predict_one(sess, image)
            for label in range(num_classes):
                segmentation = np.expand_dims(predicted_class[:,:,label], axis=2)
                color = cityscape_labels.trainId2label[label].color
                mask = np.dot(segmentation, np.array([color + (transparency_level,)]))
                mask = scipy.misc.toimage(mask, mode="RGBA")
                result_im.paste(mask, box=None, mask=mask)
            segmented_image = np.array(result_im)
            scipy.misc.imsave(os.path.join(output_dir, os.path.basename(image_file)), segmented_image)



if __name__ == '__main__':

    train_images_path_pattern = '../cityscapes/data/leftImg8bit/train/*/*_leftImg8bit.png'
    train_labels_path_pattern = '../cityscapes/data/gtFine/train/*/*_gtFine_labelTrainIds.png'
    test_images_path_pattern = '../cityscapes/data/leftImg8bit/test/*/*.png'

    parser = argparse.ArgumentParser()
    parser.add_argument('action', help='what to do: train/predict', type=str, choices=['train','predict'])
    parser.add_argument('-g', '--gpu', help='number of GPUs to use. default 0 (use CPU)', type=int, default=0)
    parser.add_argument('--gpu_mem', help='GPU memory fraction to use. default 0.9', type=float, default=0.9)
    parser.add_argument('-ep', '--epochs', help='training epochs. default 0', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', help='training batch size. default 5', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', help='training learning rate. default 0.0001', type=float, default=0.0001)
    parser.add_argument('-kp', '--keep_prob', help='training dropout keep probability. default 0.9', type=float, default=0.9)
    parser.add_argument('-rd', '--runs_dir', help='training runs directory. default runs', type=str, default='runs')
    parser.add_argument('-cd', '--ckpt_dir', help='training checkpoints directory. default ckpt', type=str, default='ckpt')
    parser.add_argument('-sd', '--summary_dir', help='training tensorboard summaries directory. default summaries', type=str, default='summaries')
    parser.add_argument('-md', '--model_dir', help='model directory. default None - model directory is created in runs. needed for predict', type=str, default=None)
    parser.add_argument('-ip', '--images_paths', help="images path/file pattern. e.g. 'train/img*.png'", type=str, default=None)
    parser.add_argument('-lp', '--labels_paths', help="label images path/file pattern. e.g. 'train/label*.png'", type=str, default=None)
    args = parser.parse_args()

    if args.images_paths is None:
        if args.action=='train':
            args.images_paths = train_images_path_pattern
        else:
            args.images_paths = test_images_path_pattern
    if args.labels_paths is None and args.action=='train':
        args.labels_paths = train_labels_path_pattern

    print("action={}".format(args.action))
    print("gpu={}".format(args.gpu))
    print('keep_prob={}'.format(args.keep_prob))
    print('images_paths={}'.format(args.images_paths))
    print('batch_size={}'.format(args.batch_size))
    print('epochs={}'.format(args.epochs))
    print('learning_rate={}'.format(args.learning_rate))

    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if args.action=='train' and not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    image_shape = (256, 512)

    if args.action=='train':
        train(args, image_shape)
    else:
        predict(args, image_shape)

    # TODO: Apply the trained model to a video
