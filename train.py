import os.path
import warnings
from distutils.version import LooseVersion
import shutil
import time

import tensorflow as tf
from tqdm import tqdm
import scipy.misc

import helper
import fcn8vgg16

"""Semantic Segmentation with Fully Convolutional Networks

Architecture as in https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

Trained on Citiscapes data https://www.cityscapes-dataset.com/
"""

_n_samples = 2975
_keep_probability_value = 0.9
_learning_rate_value = 0.0001
_gpu_count = 0
_gpu_mem_fraction = 0.9
_epochs = 0
_batch_size = 5
num_classes = 35
image_shape = (256, 512)
cityscapes_data_dir = '../cityscapes/data'
runs_dir = './runs_city2'
ckpt_dir = 'ckpt2'
summaries_dir = 'summaries'
test_data_dir = os.path.join(cityscapes_data_dir, 'leftImg8bit/test/*/*.png')
_model_dir = './data'


def load_trained_vgg_vars(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    all_ops = graph.get_operations()
    vars = [op for op in all_ops if op.op_def and op.op_def.name[:5] == 'Varia']

    var_values = {}
    for var in vars:
        name = var.name
        tensor = tf.get_default_graph().get_tensor_by_name(name + ':0')
        value = sess.run(tensor)
        name = name.replace('filter', 'weights')
        name = name.replace('fc', 'conv')
        var_values[name] = value

    return var_values


def train():
    # tensorflow GPU config
    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': _gpu_count})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = _gpu_mem_fraction

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(_model_dir)
    # extract pre-trained VGG weights
    with tf.Session(config=config) as sess:
        var_values = load_trained_vgg_vars(sess, os.path.join(_model_dir, 'vgg'))
    tf.reset_default_graph()

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function_cityscapes(os.path.join(cityscapes_data_dir, ''), image_shape)

    with tf.Session(config=config) as sess:
        # define our FCN
        images_shape = (None,)+image_shape+(3,)
        labels_shape = (None,)+image_shape+(num_classes,)
        model = fcn8vgg16.FCN8_VGG16(images_shape, labels_shape)

        # variables initialization
        sess.run(tf.global_variables_initializer())
        model.restore_variables(sess, var_values)

        # TODO: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        final_loss = model.train(sess, _epochs, _batch_size, get_batches_fn,
                                 _n_samples,
                                 _keep_probability_value, _learning_rate_value,
                                 ckpt_dir, summaries_dir)

        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # save training details to text file
        with open(os.path.join(output_dir, "params.txt"), "w") as f:
            f.write('keep_prob={}\n'.format(_keep_probability_value))
            f.write('data_dir={}\n'.format(cityscapes_data_dir))
            f.write('n_samples={}\n'.format(_n_samples))
            f.write('batch={}\n'.format(_batch_size))
            f.write('epochs={}\n'.format(_epochs))
            f.write('use_gpu={}\n'.format(_gpu_count))
            f.write('gpu_mem={}\n'.format(_gpu_mem_fraction))
            f.write('lr={}\n'.format(_learning_rate_value))
            f.write('n_samples={}\n'.format(_n_samples))
            f.write('final_loss={}\n'.format(final_loss))

        # save model
        """ save trained model using SavedModelBuilder """
        model_dir = os.path.join(output_dir, 'model')
        print('saving SavedModel into {}'.format(model_dir))
        model.save_model(model_dir)

        # Run NN on test images and save them to HD
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = model.predict(sess, test_data_dir)
        for name, image in tqdm(image_outputs):
            scipy.misc.imsave(os.path.join(output_dir, name), image)



def predict():
    # tensorflow GPU config
    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': _gpu_count})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = _gpu_mem_fraction

    tf.reset_default_graph()

    with tf.Session(config=config) as sess:
        # define our FCN
        images_shape = (None,)+image_shape+(3,)
        labels_shape = (None,)+image_shape+(num_classes,)
        model = fcn8vgg16.FCN8_VGG16(images_shape, labels_shape)

        # variables initialization
        model_dir = 'runs_city2/test/model'
        model.load_model(sess, model_dir)

        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Run NN on test images and save them to HD
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = model.predict(sess, test_data_dir)
        for name, image in tqdm(image_outputs):
            scipy.misc.imsave(os.path.join(output_dir, name), image)

        # TODO: Apply the trained model to a video



if __name__ == '__main__':

    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    train()
    #predict()
