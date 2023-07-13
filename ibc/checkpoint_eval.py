
"""Evaluates TF-Agents policies."""
import functools
import os
import shutil
import datetime
import signal

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

import gin
# Need import to get env resgistration.
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from ibc.environments.block_pushing import block_pushing_discontinuous
from ibc.environments.block_pushing import block_pushing_multimodal
from ibc.environments.collect.utils import get_oracle as get_oracle_module
from ibc.environments.particle import particle  # pylint: disable=unused-import
from ibc.environments.particle import particle_oracles
from ibc.ibc.eval import eval_env as eval_env_module
import ibc.ibc.agents.ibc_policy
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.metrics import py_metrics
from tf_agents.train.utils import train_utils
from ibc.ibc.train.get_eval_actor import EvalActor
# Need import to get tensorflow_probability registration.
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import example_encoding_dataset


flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

flags.DEFINE_integer('num_episodes', 5, 'Number of episodes to evaluate.')
flags.DEFINE_integer('history_length', None,
                     'If set the previous n observations are stacked.')
flags.DEFINE_bool('video', False,
                  'If true record a video of the evaluations.')
flags.DEFINE_bool('viz_img', False,
                  'If true records an img of evaluation trajectories.')
flags.DEFINE_string('output_path', '/tmp/ibc/policy_eval/',
                    'Path to save videos at.')
flags.DEFINE_enum(
    'task', None,
    ['REACH', 'PUSH', 'INSERT', 'REACH_NORMALIZED', 'PUSH_NORMALIZED',
     'PARTICLE', 'PUSH_DISCONTINUOUS', 'PUSH_MULTIMODAL','bimanual_v1','quad_insert2_v11'],
    'Which task of the enum to evaluate.')
flags.DEFINE_bool('use_image_obs', False,
                  'Whether to include image observations.')
flags.DEFINE_bool('flatten_env', False,
                  'If True the environment observations are flattened.')
flags.DEFINE_bool('shared_memory', False,
                  'If True the connection to pybullet uses shared memory.')
flags.DEFINE_string('model_path', None,
                    'Path to the model_policy to evaluate.')
flags.DEFINE_string('saved_model_path', None,
                    'Path to the saved_model policy to eval.')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to the checkpoint to evaluate.')
flags.DEFINE_string('checkpoint', None,
                    'checkpoint to evaluate.')
flags.DEFINE_enum('policy', None, [
    'random', 'oracle_reach', 'oracle_push', 'oracle_reach_normalized',
    'oracle_push_normalized', 'particle_green_then_blue'
], 'Static policies to evaluate.')
flags.DEFINE_string(
    'dataset_path', None,
    'If set a dataset of the policy evaluation will be saved '
    'to the given path.')
flags.DEFINE_integer('replicas', None,
                     'Number of parallel replicas generating evaluations.')

MAX_RUNTIME_SECONDS = 900

class timeout:
    def __init__(self, seconds=1, error_message='Timeout Error'):
        self.seconds = seconds
        self.error_message = error_message
        self.failured = False
    def handle_timeout(self, signum, frame):
        self.failured = True
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def evaluate(task,
             use_image_obs,
             shared_memory,
             flatten_env,
             model_path=None,
             saved_model_path=None,
             checkpoint_path=None,
             checkpoint=None,
             static_policy=None,
             dataset_path=None,
             history_length=None,
             video=False,
             viz_img=False,
             output_path=None,
             shared_memory_eval = False,
             image_obs = False,
             sequence_length = 2,
             goal_tolerance = 0.02,
             num_envs=1,
             eval_episodes=10,
             strategy=None,):
  """Evaluates the given policy for n episodes."""
  
  env_name = eval_env_module.get_env_name(task, shared_memory_eval, image_obs)
  logging.info(('Got env name:', env_name))
  env = eval_env_module.get_eval_env(
          env_name, sequence_length, goal_tolerance, num_envs)

        

  if model_path and static_policy:
    raise ValueError(
        'Only pass in either a `saved_model_path` or a `static_policy`.')

  if model_path:
    if not checkpoint:
      raise ValueError('Must provide a `checkpoint` with a model.')
    saved_model_path = str(model_path + "/policies/greedy_policy")
    checkpoint_path = str(model_path + "/policies/checkpoints/policy_checkpoint_"+ checkpoint.zfill(10))

  if saved_model_path:
    if not checkpoint_path:
      raise ValueError('Must provide a `checkpoint_path` with a saved_model.')
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        saved_model_path, env.time_step_spec(), env.action_spec(),load_specs_from_pbtxt=False)
    policy.update_from_checkpoint(checkpoint_path)
  else:
    if static_policy == 'random':
      policy = random_py_policy.RandomPyPolicy(env.time_step_spec(),
                                               env.action_spec())

  proto_path = os.path.join(model_path,"Trajectories")
  #os.makedirs(proto_path)

  if not strategy:
      strategy = tf.distribute.get_strategy()
  with strategy.scope():

    train_step = train_utils.create_train_step()
    env_name_clean = env_name.replace('/', '_')
    eval_actor_class = EvalActor()
    eval_actor, success_metric = eval_actor_class.get_checkpoint_eval_actor(
                policy,
                env_name,
                env,
                train_step,
                eval_episodes,
                model_path,
                viz_img,
                num_envs,
                strategy,
                summary_dir_suffix=env_name_clean)

    metrics = evaluation_step(
              eval_episodes,
              env,
              eval_actor,
              train_step,
              name_scope_suffix=f'_{env_name}')
    run = 2
    proto_name = os.path.join(proto_path,f"algo=ibc,checkpoint={checkpoint},run={run}")
    eval_actor_class.write_to_protobuf(proto_name)


def evaluation_step(eval_episodes, eval_env, eval_actor, train_step,name_scope_suffix=''):
  """Evaluates the agent in the environment."""
  logging.info('Evaluating policy.')
  with tf.name_scope('eval' + name_scope_suffix):
    # This will eval on seeds:
    # [0, 1, ..., eval_episodes-1]
    export_dir = "/home/docker/irl_control_container/data/ibc_eval/quad_insert2_v11/2023-06-29_08:18:06/gif"
    export_prefix = 'quad_insert2_v11_150'
    for eval_seed in range(eval_episodes):
      try:
        with timeout(seconds=MAX_RUNTIME_SECONDS):
          eval_env.seed(eval_seed)
          print("seed : ",eval_seed) ## DO NOT REMOVE THIS PRINT
          eval_actor.reset()  # With the new seed, the env actually needs reset.
          eval_env.set_gif_recording(export_dir,export_prefix,str(eval_seed))
          eval_actor.run()
      except TimeoutError as e:
        print(f"Eval Failed with: {e}. Trying again...")
        eval_env.seed(eval_seed)
        eval_env.reset()
        print("seed : ",eval_seed) ## DO NOT REMOVE THIS PRINT
        eval_actor.reset()  # With the new seed, the env actually needs reset.
        eval_env.set_gif_recording(export_dir,export_prefix,str(eval_seed))
        eval_actor.run()
      eval_env.export_gif_recording()

    eval_actor.log_metrics()
    eval_actor.write_metric_summaries()
  return eval_actor.metrics


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.add_config_file_search_path(os.getcwd())
  gin.parse_config_files_and_bindings(flags.FLAGS.gin_file,
                                      flags.FLAGS.gin_bindings)
  evaluate(
    task=flags.FLAGS.task,
    use_image_obs=flags.FLAGS.use_image_obs,
    shared_memory=flags.FLAGS.shared_memory,
    flatten_env=flags.FLAGS.flatten_env,
    model_path=flags.FLAGS.model_path,
    saved_model_path=flags.FLAGS.saved_model_path,
    checkpoint_path=flags.FLAGS.checkpoint_path,
    checkpoint=flags.FLAGS.checkpoint,
    static_policy=flags.FLAGS.policy,
    dataset_path=flags.FLAGS.dataset_path,
    history_length=flags.FLAGS.history_length,
    video=flags.FLAGS.video,
    viz_img=flags.FLAGS.viz_img,
    output_path=flags.FLAGS.output_path,
    eval_episodes=flags.FLAGS.num_episodes
    )


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
