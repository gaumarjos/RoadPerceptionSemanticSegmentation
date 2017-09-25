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

import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util
from tqdm import tqdm
import scipy.misc
import numpy as np
from moviepy.editor import VideoFileClip

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
    config = session_config(args)

    # extract pre-trained VGG weights
    with tf.Session(config=config) as sess:
        var_values = load_trained_vgg_vars(sess)
    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        # define our FCN
        num_classes = len(cityscape_labels.labels)
        model = fcn8vgg16.FCN8_VGG16(num_classes)

        # variables initialization
        sess.run(tf.global_variables_initializer())
        model.restore_variables(sess, var_values)

        # Create batch generator
        # TODO: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        train_batches_fn, num_samples = get_train_batch_generator_cityscapes(args.images_paths,
                                                                             args.labels_paths,
                                                                             image_shape)
        time_str = time.strftime("%Y%m%d_%H%M%S")
        run_name = "/{}_ep{}_b{}_lr{:.6f}_kp{}".format(time_str, args.epochs, args.batch_size, args.learning_rate, args.keep_prob)
        start_time = time.time()

        final_loss = model.train(sess, args.epochs, args.batch_size,
                                 train_batches_fn, num_samples,
                                 args.keep_prob, args.learning_rate,
                                 args.ckpt_dir, args.summary_dir+run_name)

        # Make folder for current run
        output_dir = os.path.join(args.runs_dir, time_str)
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


def predict_image(sess, model, image, colors_dict):
    # run TF prediction
    start_time = timer()
    predicted_class = model.predict_one(sess, image)
    predicted_class = np.array(predicted_class, dtype=np.uint8)
    duration = timer() - start_time
    tf_time_ms = int(duration * 1000)

    # overlay on image
    start_time = timer()
    result_im = scipy.misc.toimage(image)
    for label in range(len(colors_dict)):
        segmentation = np.expand_dims(predicted_class[:, :, label], axis=2)
        mask = np.dot(segmentation, colors_dict[label])
        mask = scipy.misc.toimage(mask, mode="RGBA")
        # paste (from PIL) seem to take time (or rather toimage calls to convert to PIL format).
        # in the future need to try this to speed up
        # https://stackoverflow.com/questions/19561597/pil-image-paste-on-another-image-with-alpha
        result_im.paste(mask, box=None, mask=mask)
    segmented_image = np.array(result_im)
    duration = timer() - start_time
    img_time_ms = int(duration * 1000)

    return segmented_image, tf_time_ms, img_time_ms

def session_config(args):
    # tensorflow GPU config
    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': args.gpu})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
    # playing with JIT level, this can be set to ON_1 or ON_2
    if args.xla is not None:
        if args.xla==1:
            jit_level = tf.OptimizerOptions.ON_1 # this works on Ubuntu tf1.3 but does not improve performance
        if args.xla==2:
            jit_level = tf.OptimizerOptions.ON_2
        config.graph_options.optimizer_options.global_jit_level = jit_level


def predict_files(args, image_shape):
    tf.reset_default_graph()
    with tf.Session(config=session_config(args)) as sess:
        model = fcn8vgg16.FCN8_VGG16(define_graph=False)
        model.load_model(sess, 'trained_model' if args.model_dir is None else args.model_dir)

        # Make folder for current run
        output_dir = os.path.join(args.runs_dir, time.strftime("%Y%m%d_%H%M%S"))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        print('Predicting on test images {} to: {}'.format(args.images_paths, output_dir))

        colors = get_colors()

        images_pbar = tqdm(glob.glob(args.images_paths),
                            desc='Predicting (last tf call __ ms)',
                            unit='images')
        tf_total_duration = 0.
        img_total_duration = 0.
        tf_count = 0.
        img_count = 0.
        for image_file in images_pbar:
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

            segmented_image, tf_time_ms, img_time_ms = predict_image(sess, model, image, colors)

            if tf_count>0:
                tf_total_duration += tf_time_ms
            tf_count += 1
            tf_avg_ms = int(tf_total_duration/(tf_count-1 if tf_count>1 else 1))

            if img_count>0:
                img_total_duration += img_time_ms
            img_count += 1
            img_avg_ms = int(img_total_duration/(img_count-1 if img_count>1 else 1))

            images_pbar.set_description('Predicting (last tf call {} ms, avg tf {} ms, last img {} ms, avg {} ms)'.format(
                tf_time_ms, tf_avg_ms, img_time_ms, img_avg_ms))
            # tf timings:
            #    mac cpu inference is  670ms on trained but unoptimized graph. tf 1.3
            # ubuntu cpu inference is 1360ms on pip tf-gpu 1.3.
            # ubuntu cpu inference is  560ms on custom built tf-gpu 1.3 (cuda+xla).
            # ubuntu gpu inference is   18ms on custom built tf-gpu 1.3 (cuda+xla). 580ms total per image. 1.7 fps
            # quantize_weights increases inference to 50ms
            # final performance on ubuntu/1080ti with ssd, including time to load/save is 3 fps

            scipy.misc.imsave(os.path.join(output_dir, os.path.basename(image_file)), segmented_image)


def freeze_graph(args):
    # based on https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    if args.ckpt_dir is None:
        print("for freezing need --ckpt_dir")
        return
    if args.frozen_model_dir is None:
        print("for freezing need --frozen_model_dir")
        return

    checkpoint = tf.train.get_checkpoint_state(args.ckpt_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    print("freezing from {}".format(input_checkpoint))
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    print("{} ops in the input graph".format(len(input_graph_def.node)))

    output_node_names = "predictions/prediction_class"

    # freeze graph
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # use a built-in TF helper to export variables to constants
        output_graph_def = tf_graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

    print("{} ops in the frozen graph".format(len(output_graph_def.node)))

    if os.path.exists(args.frozen_model_dir):
        shutil.rmtree(args.frozen_model_dir)

    # save model in same format as usual
    print('saving frozen model as saved_model to {}'.format(args.frozen_model_dir))
    model = fcn8vgg16.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(output_graph_def, name='')
    with tf.Session() as sess:
        model.save_model(sess, args.frozen_model_dir)

    print('saving frozen model as graph.pb (for transforms) to {}'.format(args.frozen_model_dir))
    with tf.gfile.GFile(args.frozen_model_dir+'/graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


def optimise_graph(args):
    """ optimize frozen graph for inference """
    if args.frozen_model_dir is None:
        print("for optimise need --frozen_model_dir")
        return
    if args.optimised_model_dir is None:
        print("for optimise need --optimised_model_dir")
        return

    print('calling c++ implementation of graph transform')
    os.system('./optimise.sh {}'.format(args.frozen_model_dir))

    # reading optimised graph
    tf.reset_default_graph()
    gd = tf.GraphDef()
    output_graph_file = args.frozen_model_dir+"/optimised_graph.pb"
    with tf.gfile.Open(output_graph_file, 'rb') as f:
        gd.ParseFromString(f.read())
    tf.import_graph_def(gd, name='')
    print("{} ops in the optimised graph".format(len(gd.node)))

    # save model in same format as usual
    shutil.rmtree(args.optimised_model_dir)
    print('saving optimised model as saved_model to {}'.format(args.optimised_model_dir))
    model = fcn8vgg16.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(gd, name='')
    with tf.Session() as sess:
        model.save_model(sess, args.optimised_model_dir)
    shutil.move(args.frozen_model_dir+'/optimised_graph.pb', args.optimised_model_dir)


def predict_video(args):
    if args.video_file_in is None:
        print("for video processing need --video_file_in")
        return
    if args.video_file_out is None:
        print("for video processing need --video_file_out")
        return

    def process_frame(image):
        # this function expects color images
        segmented_image, tf_time_ms, img_time_ms = predict_image(sess, model, image, colors)
        return segmented_image

    tf.reset_default_graph()
    with tf.Session(config=session_config(args)) as sess:
        model = fcn8vgg16.FCN8_VGG16(define_graph=False)
        model.load_model(sess, 'trained_model' if args.model_dir is None else args.model_dir)
        print('Running on video {}, output to: {}'.format(args.video_file_in, args.video_file_out))
        colors = get_colors()
        input_clip = VideoFileClip(args.video_file_in)
        annotated_clip = input_clip.fl_image(process_frame)
        annotated_clip.write_videofile(args.video_file_out, audio=False)
        # results on ubuntu/1080ti. with GPU 4.8fps. with CPU the same??
        # mac cpu



def get_colors():
    num_classes = len(cityscape_labels.labels)
    colors = {}
    transparency_level = 128
    for label in range(num_classes):
        color = cityscape_labels.trainId2label[label].color
        colors[label] = np.array([color + (transparency_level,)], dtype=np.uint8)
    return colors


def check_tf():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if args.action=='train' and not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    # set tf logging
    tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        help='what to do: train/predict/freeze/optimise/video',
                        type=str,
                        choices=['train','predict', 'freeze', 'optimise', 'video'])
    parser.add_argument('-g', '--gpu', help='number of GPUs to use. default 0 (use CPU)', type=int, default=0)
    parser.add_argument('-gm','--gpu_mem', help='GPU memory fraction to use. default 0.9', type=float, default=0.9)
    parser.add_argument('-x','--xla', help='XLA JIT level. default None', type=int, default=None, choices=[1,2])
    parser.add_argument('-ep', '--epochs', help='training epochs. default 0', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', help='training batch size. default 5', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', help='training learning rate. default 0.0001', type=float, default=0.0001)
    parser.add_argument('-kp', '--keep_prob', help='training dropout keep probability. default 0.9', type=float, default=0.9)
    parser.add_argument('-rd', '--runs_dir', help='training runs directory. default runs', type=str, default='runs')
    parser.add_argument('-cd', '--ckpt_dir', help='training checkpoints directory. default ckpt', type=str, default='ckpt')
    parser.add_argument('-sd', '--summary_dir', help='training tensorboard summaries directory. default summaries', type=str, default='summaries')
    parser.add_argument('-md', '--model_dir', help='model directory. default None - model directory is created in runs. needed for predict', type=str, default=None)
    parser.add_argument('-fd', '--frozen_model_dir', help='model directory for frozen graph. for freeze', type=str, default=None)
    parser.add_argument('-od', '--optimised_model_dir', help='model directory for optimised graph. for optimize', type=str, default=None)
    parser.add_argument('-ip', '--images_paths', help="images path/file pattern. e.g. 'train/img*.png'", type=str, default=None)
    parser.add_argument('-lp', '--labels_paths', help="label images path/file pattern. e.g. 'train/label*.png'", type=str, default=None)
    parser.add_argument('-vi', '--video_file_in', help="mp4 video file to process", type=str, default=None)
    parser.add_argument('-vo', '--video_file_out', help="mp4 video file to save results", type=str, default=None)
    args = parser.parse_args()
    return args




if __name__ == '__main__':

    train_images_path_pattern = '../cityscapes/data/leftImg8bit/train/*/*_leftImg8bit.png'
    train_labels_path_pattern = '../cityscapes/data/gtFine/train/*/*_gtFine_labelTrainIds.png'
    test_images_path_pattern = '../cityscapes/data/leftImg8bit/test/*/*.png'

    args = parse_args()

    if args.images_paths is None:
        if args.action=='train':
            args.images_paths = train_images_path_pattern
        else:
            args.images_paths = test_images_path_pattern
    if args.labels_paths is None and args.action=='train':
        args.labels_paths = train_labels_path_pattern

    check_tf()
    # fix plugin for moviepy
    import imageio
    imageio.plugins.ffmpeg.download()

    # this is image size to be read and trained on. predict also uses this
    image_shape = (256, 512)

    print("action={}".format(args.action))
    print("gpu={}".format(args.gpu))
    if args.action=='train':
        print('keep_prob={}'.format(args.keep_prob))
        print('images_paths={}'.format(args.images_paths))
        print('batch_size={}'.format(args.batch_size))
        print('epochs={}'.format(args.epochs))
        print('learning_rate={}'.format(args.learning_rate))
        train(args, image_shape)
    elif args.action=='predict':
        print('images_paths={}'.format(args.images_paths))
        predict_files(args, image_shape)
    elif args.action == 'freeze':
        freeze_graph(args)
    elif args.action == 'optimise':
        optimise_graph(args)
    elif args.action == 'video':
        predict_video(args, image_shape)
