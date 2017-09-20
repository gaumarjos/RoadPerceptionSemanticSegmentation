import tensorflow as tf
import math
from tqdm import tqdm
import os
import scipy.misc
from glob import glob
import numpy as np


class FCN8_VGG16:
    def __init__(self, images_shape, labels_shape):
        self._images_shape = images_shape
        self._labels_shape = labels_shape
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob', shape=[])
        self._learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self._create_input_pipeline()
        with tf.name_scope("encoder_vgg16"):
            self._create_vgg16_conv_layers()
            self._create_vgg16_fc_conv_layers()
        self._create_decoder()
        self._create_optimizer()
        #if weights is not None and sess is not None:
        #    self.load_weights(weights, sess)

    def train(self, sess, epochs, batch_size, get_batches_fn, n_samples, keep_prob_value, learning_rate):
        """
        Train neural network and print out the loss during training.
        :param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
        """
        # save intermediate checkpoint during training
        saver = tf.train.Saver()  # by default saves all variables
        if not os.path.exists('ckpt2'):
            os.makedirs('ckpt2')
        checkpoint_dir = 'ckpt2/model.ckpt'
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored from checkpoint {}".format(ckpt.model_checkpoint_path))

        for epoch in range(epochs):
            # running optimization in batches of training set
            n_batches = int(math.ceil(float(n_samples) / batch_size))
            batches_pbar = tqdm(get_batches_fn(batch_size, n_samples),
                                desc='Train Epoch {:>2}/{} (loss _.___)'.format(epoch + 1, epochs),
                                unit='batches',
                                total=n_batches)
            n = 0.
            l = 0.
            for images, labels in batches_pbar:
                feed_dict = {self._images: images,
                             self._labels: labels,
                             self._keep_prob: keep_prob_value,
                             self._learning_rate: learning_rate}
                _, loss = sess.run([self._optimizer, self._loss],  # , self._summaries
                                   feed_dict=feed_dict)
                n += len(images)
                l += loss * len(images)
                batches_pbar.set_description(
                    'Train Epoch {:>2}/{} (loss {:.3f})'.format(epoch + 1, epochs, l / n))
                # write training summaries for tensorboard every so often
                # step = self._global_step.eval(session=self._session)
                # if step % 5 == 0:
                #    summary_writer.add_summary(summaries, global_step=step)
                # if i % 100 == 99:  # Record execution stats
                #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #     run_metadata = tf.RunMetadata()
                #     summary, _ = sess.run([merged, train_step],
                #                           feed_dict=feed_dict(True),
                #                           options=run_options,
                #                           run_metadata=run_metadata)
                #     train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                #     train_writer.add_summary(summary, i)
                #     print('Adding run metadata for', i)
                # else:  # Record a summary
                #     summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                #     train_writer.add_summary(summary, i)

            l /= n_samples
            # batches_pbar.set_description("loss over last epoch {}".format(l))

            save_path = saver.save(sess, checkpoint_dir)  # , global_step=self._global_step)
            # print("checkpoint saved to {}".format(save_path))
        return l

    def predict(self, sess, data_path):
        """
        Generate test output using the test images
        :param sess: TF session
        :param data_path: Path to the folder that contains the datasets
        :param image_shape: Tuple - Shape of image
        :return: Output for for each test image
        """
        from labels import labels, trainId2label
        transparency_level = 56
        num_classes = self._labels_shape[-1]
        image_shape = self._images_shape[1:3]
        for image_file in tqdm(glob(data_path)):
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            result_im = scipy.misc.toimage(image)

            im_softmax = sess.run( [tf.nn.softmax(self._logits)], {self._keep_prob: 1.0, self._images: [image]})
            softmax_result = im_softmax[0].reshape(image_shape[0], image_shape[1], num_classes)
            for label in range(num_classes):
                segmentation = (softmax_result[:, :, label] > 0.5).reshape(image_shape[0],
                                                                           image_shape[1], 1)
                color = trainId2label[label].color
                mask = np.dot(segmentation, np.array([color + (transparency_level,)]))
                mask = scipy.misc.toimage(mask, mode="RGBA")
                result_im.paste(mask, box=None, mask=mask)
            yield os.path.basename(image_file), np.array(result_im)

    def _create_input_pipeline(self):
        # define input placeholders in the graph
        with tf.name_scope("data"):
            self._images = tf.placeholder(tf.uint8, name='images', shape=self._images_shape)
            tf.summary.image('input_images', self._images, 3)
            self._labels = tf.placeholder(tf.uint8, name='labels', shape=self._labels_shape)
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            self._images_float = tf.image.convert_image_dtype(self._images, tf.float32)
            self._images_std = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self._images_float)
            self._labels_float = tf.cast(self._labels, tf.float32)

    def _create_vgg16_conv_layers(self):
        self._parameters = []

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self._images_std, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # pool3
        self._pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self._pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # pool4
        self._pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self._pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)
            self._parameters += [kernel, biases]

        # pool5
        self._pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

    def _create_vgg16_fc_conv_layers(self):
        # here we create first two FC layers of VGG16, but as 1x1 convolutions

        # fc1 -> conv6
        with tf.name_scope('conv6') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self._pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(out, name='relu')
            conv6 = tf.nn.dropout(relu, self._keep_prob, name=scope)
            self._parameters += [kernel, biases]

        # fc2 -> conv7
        with tf.name_scope('conv7') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(out, name='relu')
            self._conv7 = tf.nn.dropout(relu, self._keep_prob, name=scope)
            self._parameters += [kernel, biases]

    def _create_decoder(self):
        num_classes = self._labels_shape[-1]
        with tf.name_scope("decoder"):
            conv_1x1 = tf.layers.conv2d(self._conv7, num_classes, kernel_size=1,
                                        strides=(1, 1), padding='SAME',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        name='conv_1x1')
            # in the paper 'initialise to bilinear interpolation'. here we do random initialization
            up4 = tf.layers.conv2d_transpose(conv_1x1, 512,
                                             kernel_size=4, strides=2, padding='SAME',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name='up4')
            skip4 = tf.add(up4, self._pool4, name='skip4')
            up3 = tf.layers.conv2d_transpose(skip4, 256,
                                             kernel_size=4, strides=2, padding='SAME',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name='up3')
            skip3 = tf.add(up3, self._pool3, name='skip3')
            self._output = tf.layers.conv2d_transpose(skip3, num_classes,
                                             kernel_size=16, strides=8, padding='SAME',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             name='output')

    def _create_optimizer(self):
        num_classes = self._labels_shape[-1]
        self._logits = tf.reshape(self._output, (-1, num_classes), name='logits')
        # TODO: use weighted loss based on how our classes are represented?
        self._loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=self._labels_float),
                                name="loss")
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

