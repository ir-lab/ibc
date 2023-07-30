# coding=utf-8
# Copyright 2022 The Reach ML Authors.
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

"""Main binary to train a Behavioral Cloning agent."""
#  pylint: disable=g-long-lambda
import collections
import datetime
import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
from ibc.ibc import tasks
from ibc.ibc.agents import ibc_policy  # pylint: disable=unused-import
from ibc.ibc.eval import eval_env as eval_env_module
from ibc.ibc.train import get_agent as agent_module
from ibc.ibc.train import get_cloning_network as cloning_network_module
from ibc.ibc.train import get_data as data_module
from ibc.ibc.train.get_eval_actor import EvalActor
from ibc.ibc.train import get_learner as learner_module
from ibc.ibc.train import get_normalizers as normalizers_module
from ibc.ibc.train import get_sampling_spec as sampling_spec_module
from ibc.ibc.utils import make_video as video_module
import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.specs import BoundedArraySpec, ArraySpec
from tf_agents.specs import tensor_spec
import numpy as np
from tf_agents.trajectories import TimeStep
from gail.policyopt import util
import gym
import random
import csv

flags.DEFINE_string('tag', None,
										'Tag for the experiment. Appended to the root_dir.')
flags.DEFINE_bool('add_time', False,
									'If True current time is added to the experiment path.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')
flags.DEFINE_bool('shared_memory_eval', False,
									'If true the eval_env uses shared_memory.')
flags.DEFINE_bool('video', False,
									'If true, write out one rollout video after eval.')
flags.DEFINE_multi_enum(
		'task', None,
		(tasks.IBC_TASKS + tasks.D4RL_TASKS + tasks.GYM_TASKS),
		'If True the reach task is evaluated.')
flags.DEFINE_boolean('viz_img', default=False,
										 help='Whether to save out imgs of what happened.')
flags.DEFINE_bool('skip_eval', False,
									'If true the evals are skipped and instead run from '
									'policy_eval binary.')
flags.DEFINE_bool('multi_gpu', False,
									'If true, run in multi-gpu setting.')

flags.DEFINE_enum('device_type', 'gpu', ['gpu', 'tpu'],
									'Where to perform training.')

flags.DEFINE_integer('limit_trajs', -1, 'Tag for the experiment. Appended to the root_dir.')

FLAGS = flags.FLAGS
VIZIER_KEY = 'success'



@gin.configurable
def train_eval(
		task=None,
		dataset_path=None,
		root_dir=None,
		# 'ebm' or 'mse' or 'mdn'.
		loss_type=None,
		# Name of network to train. see get_cloning_network.
		network=None,
		# Training params
		batch_size=512,
		num_iterations=20000,
		learning_rate=1e-3,
		decay_steps=100,
		replay_capacity=100000,
		eval_interval=1000,
		eval_loss_interval=100,
		eval_episodes=1,
		fused_train_steps=100,
		sequence_length=2,
		uniform_boundary_buffer=0.05,
		for_rnn=False,
		flatten_action=True,
		dataset_eval_fraction=0.0,
		goal_tolerance=0.02,
		tag=None,
		add_time=True,
		seed=0,
		viz_img=False,
		skip_eval=False,
		num_envs=1,
		shared_memory_eval=False,
		image_obs=False,
		strategy=None,
		# Use this to sweep amount of tfrecords going into training.
		# -1 for 'use all'.
		limit_trajs=-1,
		use_warmup=False,
		runs=1):
	"""Trains a BC agent on the given datasets."""
	if task is None:
		raise ValueError('task argument must be set.')
	logging.info(('Using task:', task))

	#tf.random.set_seed(seed)
	if not tf.io.gfile.exists(root_dir):
		tf.io.gfile.makedirs(root_dir)

	# Logging.
	if tag:
		root_dir = os.path.join(root_dir, tag)
	if add_time:
		current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
		root_dir = os.path.join(root_dir, current_time)

	for run in range(runs):
		# Define eval env.
		env_name = eval_env_module.get_env_name(task, shared_memory_eval, image_obs)
		logging.info(('Got env name:', env_name))
		eval_env = eval_env_module.get_eval_env(
					env_name, sequence_length, goal_tolerance, num_envs)
		logging.info(('Got eval_env:', eval_env))

		obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
				spec_utils.get_tensor_specs(eval_env))
		
		obs_tensor_spec = tensor_spec.BoundedTensorSpec(shape=obs_tensor_spec.shape, dtype=tf.float32, minimum=obs_tensor_spec.minimum, maximum=obs_tensor_spec.maximum, name=obs_tensor_spec.name)
		
		action_tensor_spec = tensor_spec.BoundedTensorSpec(shape=action_tensor_spec.shape, dtype=tf.float32, minimum=action_tensor_spec.minimum, maximum=action_tensor_spec.maximum, name=action_tensor_spec.name)
		
		# Compute normalization info from training data.
		create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
				dataset_path,
				sequence_length,
				replay_capacity,
				batch_size,
				for_rnn,
				dataset_eval_fraction,
				flatten_action,
				max_trajs=limit_trajs)
		train_data, _ = create_train_and_eval_fns_unnormalized()
		(norm_info, norm_train_data_fn) = normalizers_module.get_normalizers(
				train_data, batch_size, env_name)

		# Create normalized training data.
		if not strategy:
			strategy = tf.distribute.get_strategy()
		per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
		create_train_and_eval_fns = data_module.get_data_fns(
				dataset_path,
				sequence_length,
				replay_capacity,
				per_replica_batch_size,
				for_rnn,
				dataset_eval_fraction,
				flatten_action,
				norm_function=norm_train_data_fn,
				max_trajs=limit_trajs)

		# Create normalization layers for obs and action.
		with strategy.scope():
			# Create train step counter.
			train_step = train_utils.create_train_step()

			# Define action sampling spec.
			action_sampling_spec = sampling_spec_module.get_sampling_spec(
					action_tensor_spec,
					min_actions=norm_info.min_actions,
					max_actions=norm_info.max_actions,
					uniform_boundary_buffer=uniform_boundary_buffer,
					act_norm_layer=norm_info.act_norm_layer)

			# This is a common opportunity for a bug, having the wrong sampling min/max
			# so log this.
			logging.info(('Using action_sampling_spec:', action_sampling_spec))

			# Define keras cloning network.
			cloning_network = cloning_network_module.get_cloning_network(
					network,
					obs_tensor_spec,
					action_tensor_spec,
					norm_info.obs_norm_layer,
					norm_info.act_norm_layer,
					sequence_length,
					norm_info.act_denorm_layer)

			# Define tfagent.
			agent = agent_module.get_agent(loss_type,
					time_step_tensor_spec,
					action_tensor_spec,
					action_sampling_spec,
					norm_info.obs_norm_layer,
					norm_info.act_norm_layer,
					norm_info.act_denorm_layer,
					learning_rate,
					use_warmup,
					cloning_network,
					train_step,
					decay_steps)

			# Define bc learner.
			bc_learner = learner_module.get_learner(
					loss_type,
					root_dir,
					agent,
					train_step,
					create_train_and_eval_fns,
					fused_train_steps,
					strategy,
					checkpoint_interval=500)

		logging.info('Saving operative-gin-config.')
		with tf.io.gfile.GFile(
				os.path.join(root_dir, 'operative-gin-config.txt'), 'wb') as f:
			f.write(gin.operative_config_str())
		# Main train and eval loop.
		rows = []
		t_total = 0
		fields = ['iter','ttotal','reduced_loss']
		while train_step.numpy() < num_iterations:
			# Run bc_learner for fused_train_steps.
			with util.Timer() as t_all :
				reduced_loss = training_step(agent, bc_learner, fused_train_steps, train_step)
			t_total += t_all.dt
			rows.append([train_step.numpy(),t_total,reduced_loss])
		csv_name = os.path.join(root_dir,f"log.csv")
		print(csv_name)
		with open(csv_name, 'w') as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow(fields)
			csvwriter.writerows(rows)


def training_step(agent, bc_learner, fused_train_steps, train_step):
	"""Runs bc_learner for fused training steps."""
	reduced_loss_info = None
	if not hasattr(agent, 'ebm_loss_type') or agent.ebm_loss_type != 'cd_kl':
		reduced_loss_info = bc_learner.run(iterations=fused_train_steps)
	else:
		for _ in range(fused_train_steps):
			# I think impossible to do this inside tf.function.
			agent.cloning_network_copy.set_weights(
					agent.cloning_network.get_weights())
			reduced_loss_info = bc_learner.run(iterations=1)
	if reduced_loss_info:
		# Graph the loss to compare losses at the same scale regardless of
		# number of devices used.
		with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(
				True):
			tf.summary.scalar(
					'reduced_loss', reduced_loss_info.loss, step=train_step)
	return reduced_loss_info.loss.numpy()

def main(_):
	logging.set_verbosity(logging.INFO)

	gin.add_config_file_search_path(os.getcwd())
	gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings,
			# TODO(coreylynch): This is a temporary
			# hack until we get proper distributed
			# eval working. Remove it once we do.
			skip_unknown=True)

	# For TPU, FLAGS.tpu will be set with a TPU address and FLAGS.use_gpu
	# will be False.
	# For GPU, FLAGS.tpu will be None and FLAGS.use_gpu will be True.
	strategy = strategy_utils.get_strategy(
			tpu=FLAGS.tpu, use_gpu=FLAGS.use_gpu)

	task = FLAGS.task or gin.REQUIRED
	# If setting this to True, change `my_rangea in mcmc.py to `= range`
	tf.config.experimental_run_functions_eagerly(False)
	train_eval(
			task=task[0],
			tag=FLAGS.tag,
			add_time=FLAGS.add_time,
			viz_img=FLAGS.viz_img,
			skip_eval=FLAGS.skip_eval,
			shared_memory_eval=FLAGS.shared_memory_eval,
			strategy=strategy)


if __name__ == '__main__':
	multiprocessing.handle_main(functools.partial(app.run, main))