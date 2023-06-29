from ibc.ibc.train import get_data as data_module
import tensorflow as tf
from tf_agents.utils import example_encoding
from tf_agents.utils import example_encoding_dataset
from ibc.environments.d4rl import car_oracles
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.specs import tensor_spec
from utils.proto_tools.build.py.trajectory_pb2 import floatList, multiFloatList, trajectory
from utils.proto_tools.build.py import trajectory_pb2
import numpy as np

# dataset_path = '/home/docker/irl_control_container/libraries/algorithms/ibc/data/d4rl/car2/2d_oracle_car_3.tfrecord'
# spec_path = '/home/docker/irl_control_container/libraries/algorithms/ibc/data/d4rl/car2/2d_oracle_car_3.tfrecord.spec'
# sequence_length = 2
# replay_capacity = 1000
# batch_size = 512
# for_rnn = False
# dataset_eval_fraction = 0.0
# flatten_action = True
# _test_path = '/home/docker/irl_control_container/libraries/algorithms/ibc'
proto_file = '/home/docker/irl_control_container/data/expert_trajectories/quad_insert2_v11_test/quad_insert2_v11_test.proto'

# env_name = 'MountainCarContinuous-v0'
# env = suite_gym.load(env_name)
# policy = car_oracles.ParticleOracle(env)
# tensor_data_spec = policy.collect_data_spec
# _array_data_spec = tensor_spec.to_nest_array_spec(tensor_data_spec)
# _encoder = example_encoding.get_example_serializer(
#         _array_data_spec,
#         compress_image=True,
#         image_quality=95)

f = open(proto_file, "rb")
traj = trajectory()
traj.ParseFromString(f.read())
f.close()
# import pdb;pdb.set_trace()

# # num_trajs, max_len = len(traj.lengths), max(traj.lengths)
# # action_dim = len(traj.actions[0].sub_lists)
# # obs_dim = len(traj.observations[0].sub_lists)
# # actions_all = np.zeros((num_trajs, max_len, action_dim))
# # obs_all = np.zeros((num_trajs, max_len, obs_dim))
# # rewards_all = np.zeros((num_trajs, max_len))
# # for idx in range(len(traj.observations)):
# #         act = np.vstack([x.value for x in traj.actions[idx].sub_lists]).T
# #         actions_all[idx, :act.shape[0], :] = act
# #         obs = np.vstack([x.value for x in traj.observations[idx].sub_lists]).T
# #         obs_all[idx, :obs.shape[0], :] = obs
# #         rewards = traj.rewards[idx].value
# #         rewards_all[idx, :len(rewards)] = rewards
# # lengths = np.array(traj.lengths)
# #import pdb;pdb.set_trace()


# # create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
# #       dataset_path,
# #       sequence_length,
# #       replay_capacity,
# #       batch_size,
# #       for_rnn,
# #       dataset_eval_fraction,
# #       flatten_action)

# raw_dataset = tf.data.TFRecordDataset(dataset_path)

# for raw_record in raw_dataset.take(2):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _double_float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(action,discount,next_step_type,observation,reward,step_type):
        Trajectory = {
                "action": _float_feature(action),
                "discount": _float_feature(0),
                "next_step_type": _float_feature(next_step_type),
                "observation": _float_feature(observation),
                "reward": _float_feature(reward),
                "step_type": _float_feature(step_type),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto

serialized_example = serialize_example(5,0, 1, 65, -1,0)
import pdb;pdb.set_trace()
