"""Semantic Segmentation with Fully Convolutional Networks

Architecture as in https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

Trained on Citiscapes data https://www.cityscapes-dataset.com/
Trained on Mapillary data (added by Stefano Salati)
"""


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

import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util as tf_graph_util
from tqdm import tqdm
import scipy.misc
import scipy.io as sio
import numpy as np
from moviepy.editor import VideoFileClip

import helper
import fcn8vgg16
import camera_calibration
import object_tracking

# dataset = "cityscapes"
dataset = "mapillary"
if dataset == "cityscapes":
    import cityscape_labels as dataset_labels
elif dataset == "mapillary":
    import mapillary_labels as dataset_labels

USE_TF_BATCHING = True


def load_trained_vgg_vars(sess):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :return: dict name/value where value is pre-trained array
    """
    # Download pretrained vgg model
    vgg_path = 'pretrained_vgg/vgg'
    
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


def get_train_batch_generator(images_path_pattern, labels_path_pattern, image_shape):
    """
    Generate function to create batches of training data
    This function is ignored if USE_TF_BATCHING is True, as in that case batching operations are dealt with by Tensorflow itself
    :param images_path_pattern: path pattern for images
    :param labels_path_pattern: path pattern for labels
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = glob.glob(images_path_pattern)
    if dataset == "cityscapes":
        label_paths = {re.sub('_gtFine_labelTrainIds', '_leftImg8bit', os.path.basename(path)): path
                            for path in glob.glob(labels_path_pattern)}
    elif dataset == "mapillary":
        label_paths = {re.sub('_instance', '_image', os.path.basename(path)): path
                            for path in glob.glob(labels_path_pattern)}
    num_classes = len(dataset_labels.labels)
    num_samples = len(image_paths)
    assert len(image_paths) == len(label_paths)
    
    print("num_classes={}".format(num_classes))
    print("num_samples={}".format(num_samples))

    """
    # Original version, for reference
     def get_batches_fn(batch_size):
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
    """

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
                # start_time = time.time()
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imread(image_file)
                gt_image = scipy.misc.imread(gt_image_file)

                # If gt_image has 3 channels instead of the necessary 1, just remove the others, it's BW anyway
                if len(gt_image.shape) > 2:
                    gt_image = gt_image[:,:,0]

                # Resize images only if necessary
                if image.shape[0] != image_shape[0] or image.shape[1] != image_shape[1]:
                    image = scipy.misc.imresize(image, image_shape)                         # 256x512x3
                if gt_image.shape[0] != image_shape[0] or gt_image.shape[1] != image_shape[1]:
                    gt_image = scipy.misc.imresize(gt_image, image_shape)                   # 256*512*1, the value is the label

                """
                # https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

                # Augmentation by translating vertically --> it looked like a smart idea but the result is rather rubbish...
                tx = 0
                ty = random.randint(0,20)
                M = np.float32([[1,0,tx],[0,1,ty]])
                # scipy.misc.imsave('augmentation_test/trans_{}_a.png'.format(count), image)
                image = cv2.warpAffine(image, M, image_shape[::-1])
                gt_image = cv2.warpAffine(gt_image, M, image_shape[::-1])
                # scipy.misc.imsave('augmentation_test/trans_{}_b.png'.format(count), image)
                """

                gt_image = gt_image.reshape(*gt_image.shape, 1)
                tmp = []
                for label in range(num_classes):
                    # STE: I commented these two lines cos they're unnecessarily slow and "color" has dimension 1, not 3
                    # color = np.array([label, label, label])
                    # gt_bg = np.all(gt_image == color, axis=2)
                    gt_bg = np.all(gt_image == label, axis=2)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    tmp.append(gt_bg)
                gt_image = np.concatenate(tmp, axis=2)

                # duration = time.time() - start_time
                # print('Pre-processing time per image: {}'.format(duration))

                images.append(image)
                gt_images.append(gt_image)
                count += 1

            yield np.array(images), np.array(gt_images)
    return get_batches_fn, num_samples


def train(args, image_shape):
    # DOCUMENTATION
    # https://www.tensorflow.org/api_docs/python/tf/train/batch
    # https://www.tensorflow.org/programmers_guide/datasets
    # http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
    # https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html

    config = session_config(args)

    # extract pre-trained VGG weights
    with tf.Session(config=config) as sess:
        var_values = load_trained_vgg_vars(sess)
    tf.reset_default_graph()
    
    # This portion of the code is enabled only if USE_TF_BATCHING is enabled
    if USE_TF_BATCHING:
        # Load paths and images in memory (they're only string, there's space)
        images_path_pattern = args.images_paths
        labels_path_pattern = args.labels_paths                                              
        image_paths = glob.glob(images_path_pattern)
        if dataset == "cityscapes":
            label_paths_tmp = {re.sub('_gtFine_labelTrainIds', '_leftImg8bit', os.path.basename(path)): path
                                for path in glob.glob(labels_path_pattern)}
        elif dataset == "mapillary":
            label_paths_tmp = {re.sub('_instance', '_image', os.path.basename(path)): path
                                for path in glob.glob(labels_path_pattern)}
        num_classes = len(dataset_labels.labels)
        num_samples = len(image_paths)
        assert len(image_paths) == len(label_paths_tmp)

        print("num_classes={}".format(num_classes))
        print("num_samples={}".format(num_samples))

        label_paths = []
        for image_file in image_paths:
            label_paths.append(label_paths_tmp[os.path.basename(image_file)])
            # Now we have a list of image_paths and one of label_paths

        # Convert string into tensors
        all_images = ops.convert_to_tensor(image_paths, dtype=dtypes.string)
        all_labels = ops.convert_to_tensor(label_paths, dtype=dtypes.string)

        # Create input queues
        train_input_queue = tf.train.slice_input_producer([all_images, all_labels],
                                                          shuffle=True)

        # Process path and string tensor into an image and a label
        train_image = tf.image.decode_png(tf.read_file(train_input_queue[0]), channels=3)
        train_label = tf.image.decode_png(tf.read_file(train_input_queue[1]), channels=1)

        # Define tensor shape
        train_image.set_shape([256, 512, 3])
        train_label.set_shape([256, 512, 1])

        # One hot on the label
        train_label_one_hot = tf.one_hot(tf.squeeze(train_label), num_classes)

        # Collect batches of images before processing
        train_image_batch, train_label_batch = tf.train.batch([train_image, train_label_one_hot],
                                                              batch_size=args.batch_size,
                                                              allow_smaller_final_batch=True,
                                                              num_threads=1)

        def dummy_batch_fn(sess):
            tmp_image_batch, tmp_label_batch = sess.run([train_image_batch, train_label_batch])
            """
            print(debug)
            for i in range(tmp_label_batch.shape[0]):
                print("R={:6d}    S={:6d}    T={:6d}".format(np.count_nonzero(tmp_label_batch[i,:,:,13]),
                                                             np.count_nonzero(tmp_label_batch[i,:,:,27]),
                                                             np.count_nonzero(tmp_label_batch[i,:,:,30])))
            """
            return np.array(tmp_image_batch), np.array(tmp_label_batch)


    # TF session
    with tf.Session(config=config) as sess:
        # define our FCN
        num_classes = len(dataset_labels.labels)
        model = fcn8vgg16.FCN8_VGG16(num_classes)

        # variables initialization
        sess.run(tf.global_variables_initializer())
        model.restore_variables(sess, var_values)

        if USE_TF_BATCHING:
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
        else:
            train_batches_fn, num_samples = get_train_batch_generator(args.images_paths,
                                                                  args.labels_paths,
                                                                  image_shape)

        # Details for saving
        time_str = time.strftime("%Y%m%d_%H%M%S")
        run_name = "/{}_ep{}_b{}_lr{:.6f}_kp{}".format(time_str, args.epochs, args.batch_size, args.learning_rate, args.keep_prob)
        start_time = time.time()

        if USE_TF_BATCHING:
            final_loss = model.train2(sess, args.epochs, args.batch_size,
                                      dummy_batch_fn, num_samples,
                                      args.keep_prob, args.learning_rate,
                                      args.ckpt_dir, args.summary_dir+run_name)
        else:
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
        if args.model_dir is None:
            model_dir = os.path.join(output_dir, 'model')
        else:
            model_dir = args.model_dir
        print('saving trained model to {}'.format(model_dir))
        model.save_model(sess, model_dir)

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()

def predict_image(sess, model, image, colors_dict, trackers=[]):
    # this image size is arbitrary and may break middle of decoder in the network.
    # need to feed FCN images sizes in multiples of 32
    image_shape = [x for x in image.shape]
    fcn_shape = [x for x in image.shape]
    # should be bigger and multiple of 32 for fcn to work
    fcn_shape[0] = math.ceil(fcn_shape[0] / 32) * 32
    fcn_shape[1] = math.ceil(fcn_shape[1] / 32) * 32
    tmp_image = np.zeros(fcn_shape, dtype=np.uint8)
    tmp_image[0:image_shape[0], 0:image_shape[1], :] = image

    # run TF prediction
    start_time = timer()
    predicted_class = model.predict_one(sess, tmp_image)
    predicted_class = np.array(predicted_class, dtype=np.uint8)
    duration = timer() - start_time
    tf_time_ms = int(duration * 1000)

    def generate_bboxes(labels):
        bboxes = []
        # centroids = []
        for labelnr in range(1, labels[1] + 1):
            nonzero = (labels[0] == labelnr).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)
            """
            centroid = ((np.min(nonzerox) + np.max(nonzerox)) / 2,
                        (np.min(nonzeroy) + np.max(nonzeroy)) / 2)
            centroids.append(centroid)
            """
        return bboxes

    def draw_boxes(img, bboxes, label_nr, color=(255,255,255), thick=1):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
            cv2.putText(imcopy,
                        dataset_labels.id2name[label_nr],
                        bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
        # Return the image copy with boxes drawn
        return imcopy

    # overlay on image
    start_time = timer()
    if image_painting_style == 0:
        stack = np.zeros(tmp_image.shape, tmp_image.dtype)
        for label in range(len(colors_dict)):
            active = np.expand_dims(predicted_class[:, :, label], axis=2)
            layer = np.full(tmp_image.shape, colors_dict[label][0,0:3])
            layer = cv2.bitwise_and(layer, layer, mask=active)
            stack = cv2.addWeighted(stack, 1, layer, 1, 0)
        segmented_image = stack.copy()

    elif image_painting_style == 1:
        stack = np.zeros(tmp_image.shape, tmp_image.dtype)
        for label in range(len(colors_dict)):
            active = np.expand_dims(predicted_class[:, :, label], axis=2)
            layer = np.full(tmp_image.shape, colors_dict[label][0,0:3])
            layer = cv2.bitwise_and(layer, layer, mask=active)
            stack = cv2.addWeighted(stack, 1, layer, 1, 0)
        segmented_image = cv2.addWeighted(tmp_image, 1, stack, 0.5, 0)

    elif image_painting_style == 2:
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

    # Bounding boxes
    for tracker in trackers:
        aoi = predicted_class[:, :, tracker.label_nr]
        # labelled_aoi = scipymeas.label(aoi)     # labelled_aoi will contain a list of cars (adjacent cars are treated as one single car, this needs to be improved) TODO
        tracker.update_heatmap(aoi)
        bboxes = generate_bboxes(scipymeas.label(tracker.heatmap))
        # bombo = np.asarray(colors_dict[tracker.label_nr][0,:3], dtype=np.int32)
        segmented_image = draw_boxes(segmented_image, bboxes, tracker.label_nr)

    duration = timer() - start_time
    img_time_ms = int(duration * 1000)

    out = segmented_image[0:image_shape[0], 0:image_shape[1], :]

    return out, tf_time_ms, img_time_ms

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
    print('called c++ implementation of graph transform')

    # reading optimised graph
    tf.reset_default_graph()
    gd = tf.GraphDef()
    output_graph_file = args.frozen_model_dir+"/optimised_graph.pb"
    with tf.gfile.Open(output_graph_file, 'rb') as f:
        gd.ParseFromString(f.read())
    tf.import_graph_def(gd, name='')
    print("{} ops in the optimised graph".format(len(gd.node)))

    # save model in same format as usual
    shutil.rmtree(args.optimised_model_dir, ignore_errors=True)
    #if not os.path.exists(args.optimised_model_dir):
    #    os.makedirs(args.optimised_model_dir)

    print('saving optimised model as saved_model to {}'.format(args.optimised_model_dir))
    model = fcn8vgg16.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(gd, name='')
    with tf.Session() as sess:
        model.save_model(sess, args.optimised_model_dir)
    shutil.move(args.frozen_model_dir+'/optimised_graph.pb', args.optimised_model_dir)


def predict_video(args, image_shape=None, force_reshape=True):
    if args.video_file_in is None:
        print("for video processing need --video_file_in")
        return
    if args.video_file_out is None:
        print("for video processing need --video_file_out")
        return

    # Initializing tracker object to track movable vehicles
    tracker_labels = [19, 52, 54, 55, 56, 57, 59, 60, 61, 62]
    trackers = []
    for tracker_label in tracker_labels:
        trackers.append(object_tracking.Tracker(tracker_label))
    
    # The actual frame processing is dealt with in this function
    def process_frame(image):
        segmented_image, tf_time_ms, img_time_ms = predict_image(sess, model, image, colors, trackers)
        return segmented_image
        
    def process_frame_with_reshape(image):
        if image_shape is not None:
            # Apply intrisic camera calibration (undistort) --> not needed as I've already undistorted the test videos beforehand
            # image = camera_calibration.undistort_image(image, mtx, dist)  # adds about 14% of overhead
            image = scipy.misc.imresize(image, image_shape)
        segmented_image, tf_time_ms, img_time_ms = predict_image(sess, model, image, colors, trackers)
        return segmented_image

    tf.reset_default_graph()
    with tf.Session(config=session_config(args)) as sess:
        model = fcn8vgg16.FCN8_VGG16(define_graph=False)
        model.load_model(sess, 'trained_model' if args.model_dir is None else args.model_dir)
        print('Running on video {}, output to: {}'.format(args.video_file_in, args.video_file_out))
        colors = get_colors()
        input_clip = VideoFileClip(args.video_file_in)
        
        if force_reshape:
            if args.video_start_second is None or args.video_end_second is None:
                annotated_clip = input_clip.fl_image(process_frame_with_reshape)
            else:
                annotated_clip = input_clip.fl_image(process_frame_with_reshape).subclip(args.video_start_second,args.video_end_second)
        else:
            if args.video_start_second is None or args.video_end_second is None:
                annotated_clip = input_clip.fl_image(process_frame)
            else:
                annotated_clip = input_clip.fl_image(process_frame).subclip(args.video_start_second,args.video_end_second)
                
        annotated_clip.write_videofile(args.video_file_out, audio=False)
        # for half size
        # ubuntu/1080ti. with GPU ??fps. with CPU the same??
        # mac/cpu 1.8s/frame
        # full size 1280x720
        # ubuntu/gpu 1.2s/frame i.e. 0.8fps :(
        # ubuntu/cpu 1.2fps
        # mac cpu 6.5sec/frame


def get_colors():
    num_classes = len(dataset_labels.labels)
    colors = {}
    transparency_level = 128
    for label in range(num_classes):
        color = tuple(dataset_labels.trainId2label[label].color)
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
    parser.add_argument('-vs', '--video_start_second', help="video start second", type=int, default=None)
    parser.add_argument('-ve', '--video_end_second', help="video end second", type=int, default=None)
    args = parser.parse_args()
    return args




if __name__ == '__main__':

    if dataset == "cityscapes":
        train_images_path_pattern = '/home/ele_16/Documents/CarND/ste/cityscapes/data/leftImg8bit/train/*/*_leftImg8bit.png'
        train_labels_path_pattern = '/home/ele_16/Documents/CarND/ste/cityscapes/data/gtFine/train/*/*_gtFine_labelTrainIds.png'
        test_images_path_pattern  = '/home/ele_16/Documents/CarND/ste/cityscapes/data/leftImg8bit/test/*/*.png'
    elif dataset == "mapillary":
        train_images_path_pattern = '../mapillary/data/training/images_processed_256x512/*_image.png'
        train_labels_path_pattern = '../mapillary/data/training/instances_processed_256x512/*_instance.png'
        # test_images_path_pattern  = '../mapillary/data/testing/*.jpg'  # Keep in mind these have all different sizes
        test_images_path_pattern  = '../videos/20171201_stereo_TMG/test_frames/*_cropped.png'  # To be used with the images recorded in Cologne

    # This enables a faster (but slightly uglier, less saturated) code to paint on the output images and videos.
    # The idea is to disable it when preparing images and videos for presentations.
    # 0: just label colors
    # 1: fast painting
    # 2: slow painting (better saturation)
    image_painting_style = 0

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


    print("action={}".format(args.action))
    print("gpu={}".format(args.gpu))
    if args.action=='train':
        print('keep_prob={}'.format(args.keep_prob))
        print('images_paths={}'.format(args.images_paths))
        print('batch_size={}'.format(args.batch_size))
        print('epochs={}'.format(args.epochs))
        print('learning_rate={}'.format(args.learning_rate))
        # this is image size to be read and trained on. predict also uses this
        # cityscapes size is 2048x1024. looks like the ratio should stay or decoder fails
        image_shape = (256, 512)
        train(args, image_shape)
    elif args.action=='predict':
        print('images_paths={}'.format(args.images_paths))
        #image_shape = (256, 512)
        image_shape = (320, 512)
        predict_files(args, image_shape)
    elif args.action == 'freeze':
        freeze_graph(args)
    elif args.action == 'optimise':
        optimise_graph(args)
    elif args.action == 'video':
        #image_shape = None
        #image_shape = (int(720/2), int(1280/2))  # Original code
        #image_shape = (256, 512)                  # My version to have the same form factor of the train images
        image_shape = (1440, 896)
        #predict_video(args, image_shape, False if image_shape == (256, 512) else True)
        predict_video(args, image_shape, False)
