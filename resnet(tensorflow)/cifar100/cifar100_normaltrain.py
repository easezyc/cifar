# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import resnet_model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='./tmp/cifar100_data',
                    help='The path to the CIFAR-100 data directory.')

parser.add_argument('--model_dir', type=str, default='./tmp/cifar100_model_res32_wordL2',
                    help='The directory where the model will be stored.')

parser.add_argument('--log_dir', type=str, default="./log/cifar100_log_res%d_origin_weightdecay",
                    help='The directory where the log will be stored.')

parser.add_argument('--resnet_size', type=int, default=104,
                    help='The size of the ResNet model to use.')

parser.add_argument('--lr_boundaries', type=list, default=[50, 100, 150, 200, 250, 300, 350],
                    help='The boundaries for learning rate.')

parser.add_argument('--lr_values', type=list, default=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002],
                    help='The multiple value for learning rate.')

parser.add_argument('--train_epochs', type=int, default=400,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')


parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 100
d_cifar100 = {}

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 5e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

def cal_d_cifar():#计算d_cifar
    global  d_cifar100
    cifar100 = pd.read_csv('./cifar100_numindex_vec.csv', header=0, encoding='utf8')
    for i in range(100):
        d_cifar100[cifar100.iloc[i, 0]] = (np.array(cifar100.iloc[i, 1:301]) + 1)/2

l_map = []
file_map = open("map.txt", "r")
while True:
    line = file_map.readline().strip("\n")
    if not line:
        break
    l_map.append(int(line))
file_map.close()
np_map = np.array(l_map)
cal_d_cifar()
np_cifar100 = np.zeros([100, 300])
for i in range(100):
    np_cifar100[i, :] = d_cifar100[i]
np_cifar100 = np_cifar100.astype("float32")
print(np_cifar100.dtype)
tensor_cifar100 = tf.convert_to_tensor(np_cifar100)
tensor_l_map = tf.convert_to_tensor(np_map)

def change_labelto_wordvec(label):
    return d_cifar100[label]

def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 2
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-100-binary')

  assert os.path.exists(data_dir), (
      'Run cifar100_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'train.bin')
    ]
  else:
    return [os.path.join(data_dir, 'test.bin')]


def parse_record(raw_record):
  """Parse CIFAR-10 image and label from a raw record."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  coarse_label_bytes = 1
  fine_label_bytes = 1
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = coarse_label_bytes + fine_label_bytes + image_bytes

  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)
  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[1], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)
  #label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[coarse_label_bytes + fine_label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = record_dataset(get_filenames(is_training, data_dir))

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance. Because CIFAR-10
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training), label))

  dataset = dataset.prefetch(2 * batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Batch results by up to batch_size, and then fetch the tuple from the
  # iterator.
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def cifar100_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  tf.summary.image('images', features, max_outputs=6)

  network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES, params['data_format'])

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  tf.identity(logits, name='logits')
  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)
  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  tf.identity(loss, name='loss')
  tf.summary.scalar('loss', loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 128
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    #boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 180, 300]]
    #values = [initial_learning_rate * decay for decay in [0.2, 0.04, 0.02, 0.01]]

    boundaries = [int(batches_per_epoch * epoch) for epoch in FLAGS.lr_boundaries]
    values = [initial_learning_rate * decay for decay in FLAGS.lr_values]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])
  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.log_dir,
      summary_op=tf.summary.merge_all())
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics,
      training_hooks=[summary_hook])


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  cifar_classifier = tf.estimator.Estimator(
      model_fn=cifar100_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size': FLAGS.resnet_size,
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    cifar_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])
    tensor_to_log = {
        'cross_entropy': 'cross_entropy'
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=10000 / 128)
    # Evaluate the model and print results
    eval_results = cifar_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size),
        hooks=[logging_hook])
    print(eval_results)

    #eval_results = cifar_classifier.predict(input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    #arr = np.zeros((1, 100))
    #for i in range(100):
    #    t = next(eval_results)
    #    t = np.reshape(t, (1, 100))
    #    arr = np.concatenate((arr, t), axis=0)
    #print(max(arr),min(arr))

def save_first_model():
    wordmodel_dir = "./tmp/cifar100_model_res%d_origin_weightdecay%f_1" % (FLAGS.resnet_size, _WEIGHT_DECAY)
    network = resnet_model.cifar10_resnet_v2_generator(
        FLAGS.resnet_size, _NUM_CLASSES, FLAGS.data_format)
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels = tf.placeholder(tf.float32, [None, _NUM_CLASSES])
    logits = network(x, True)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')

    # Add weight decay to the loss.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    initial_learning_rate = 0.1
    batches_per_epoch = _NUM_IMAGES['train'] / 128
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    ckpt = tf.train.get_checkpoint_state(wordmodel_dir)
    model_restore_path = ckpt.model_checkpoint_path
    all_var = tf.trainable_variables()
    restorelist = []
    for key in all_var:
        # print(key.name)
        if not "dense" in key.name:  # and not "batch_normalization" in key.name:
            print(key.name)
            restorelist.append(key)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(restorelist)
    saver.restore(sess, model_restore_path)
    saver = tf.train.Saver()
    save_path = saver.save(sess, FLAGS.model_dir + "/model.ckpt")
    sess.close()
    print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.model_dir = "./tmp/cifar100_model_res%d_origin_weightdecay%f_1" % (FLAGS.resnet_size, _WEIGHT_DECAY)
  FLAGS.log_dir = "./log/cifar100_log_res%d_origin_weightdecay%f_1" % (FLAGS.resnet_size, _WEIGHT_DECAY)
  main(unparsed)
  FLAGS.model_dir = "./tmp/cifar100_model_res%d_origin_weightdecay%f_2" % (FLAGS.resnet_size, _WEIGHT_DECAY)
  FLAGS.log_dir = "./log/cifar100_log_res%d_origin_weightdecay%f_2" % (FLAGS.resnet_size, _WEIGHT_DECAY)
  save_first_model()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
