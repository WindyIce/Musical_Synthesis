from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data_normalizer
import layers
import networks
import tensorflow.compat.v1 as tf


def _num_filters_fn(block_id, **kwargs):
  """Computes number of filters of block `block_id`."""
  return networks.num_filters(block_id, kwargs['fmap_base'],
                              kwargs['fmap_decay'], kwargs['fmap_max'])


def generator_fn_specgram(inputs, **kwargs):
  """Builds generator network."""
  # inputs = (noises, one_hot_labels)
  with tf.variable_scope('generator_cond'):
    z = tf.concat(inputs, axis=1)
  if kwargs['to_rgb_activation'] == 'tanh':
    to_rgb_activation = tf.tanh
  elif kwargs['to_rgb_activation'] == 'linear':
    to_rgb_activation = lambda x: x
  fake_images, end_points = networks.generator(
      z,
      kwargs['progress'],
      lambda block_id: _num_filters_fn(block_id, **kwargs),
      kwargs['resolution_schedule'],
      num_blocks=kwargs['num_blocks'],
      kernel_size=kwargs['kernel_size'],
      colors=2,
      to_rgb_activation=to_rgb_activation,
      simple_arch=kwargs['simple_arch'])
  shape = fake_images.shape
  normalizer = data_normalizer.registry[kwargs['data_normalizer']](kwargs)
  fake_images = normalizer.denormalize_op(fake_images)
  fake_images.set_shape(shape)
  return fake_images, end_points


def discriminator_fn_specgram(images, **kwargs):
  """Builds discriminator network."""
  shape = images.shape
  normalizer = data_normalizer.registry[kwargs['data_normalizer']](kwargs)
  images = normalizer.normalize_op(images)
  images.set_shape(shape)
  logits, end_points = networks.discriminator(
      images,
      kwargs['progress'],
      lambda block_id: _num_filters_fn(block_id, **kwargs),
      kwargs['resolution_schedule'],
      num_blocks=kwargs['num_blocks'],
      kernel_size=kwargs['kernel_size'],
      simple_arch=kwargs['simple_arch'])
  with tf.variable_scope('discriminator_cond'):
    x = tf.layers.flatten(end_points['last_conv'])
    end_points['classification_logits'] = layers.custom_dense(
        x=x, units=kwargs['num_tokens'], scope='classification_logits')
  return logits, end_points


g_fn_registry = {
    'specgram': generator_fn_specgram,
}


d_fn_registry = {
    'specgram': discriminator_fn_specgram,
}
