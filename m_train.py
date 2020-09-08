
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import json
import os
import time

from absl import logging
import absl.flags
import data_helpers
import data_normalizer
import flags as lib_flags
import model as lib_model
import train_util
import util
import tensorflow.compat.v1 as tf


absl.flags.DEFINE_string('hparams', '{}', 'Flags dict as JSON string.')
absl.flags.DEFINE_string('config', 'mel_prog_hires', 'Name of config module.')
FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def init_data_normalizer(config):
  """Initializes data normalizer."""
  normalizer = data_normalizer.registry[config['data_normalizer']](config)
  if normalizer.exists():
    return

  if config['task'] == 0:
    tf.reset_default_graph()
    data_helper = data_helpers.registry[config['data_type']](config)
    real_images, _ = data_helper.provide_data(batch_size=10)

    # Save normalizer.
    # Note if normalizer has been saved, save() is no-op. To regenerate the
    # normalizer, delete the normalizer file in train_root_dir/assets
    normalizer.save(real_images)
  else:
    while not normalizer.exists():
      time.sleep(5)


def run(config):
  """Entry point to run training."""
  init_data_normalizer(config)

  stage_ids = train_util.get_stage_ids(**config)
  if not config['train_progressive']:
    stage_ids = list(stage_ids)[-1:]

  # Train one stage at a time
  for stage_id in stage_ids:
    batch_size = train_util.get_batch_size(stage_id, **config)
    tf.reset_default_graph()
    with tf.device(tf.train.replica_device_setter(config['ps_tasks'])):
      model = lib_model.Model(stage_id, batch_size, config)
      model.add_summaries()
      print('Variables:')
      for v in tf.global_variables():
        print('\t', v.name, v.get_shape().as_list())
      logging.info('Calling train.train')
      train_util.train(model, **config)

def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True
  # Set hyperparams from json args and defaults
  flags = lib_flags.Flags()
  # Config hparams
  if FLAGS.config:
    config_module = importlib.import_module(
        'magenta.models.gansynth.configs.{}'.format(FLAGS.config))   # TODO: Magenta not needed
    flags.load(config_module.hparams)
  # Command line hparams
  flags.load_json(FLAGS.hparams)
  # Set default flags
  lib_model.set_flags(flags)

  print('Flags:')
  flags.print_values()

  # Create training directory
  flags['train_root_dir'] = util.expand_path(flags['train_root_dir'])
  if not tf.gfile.Exists(flags['train_root_dir']):
    tf.gfile.MakeDirs(flags['train_root_dir'])

  # Save the flags to help with loading the model latter
  fname = os.path.join(flags['train_root_dir'], 'experiment.json')
  with tf.gfile.Open(fname, 'w') as f:
    json.dump(flags, f)  # pytype: disable=wrong-arg-types

  # Run training
  run(flags)


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
