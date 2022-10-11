from ibc.ibc.train import get_data as data_module
import tensorflow as tf
from tf_agents.utils import example_encoding
from tf_agents.utils import example_encoding_dataset
from ibc.environments.d4rl import car_oracles
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.specs import tensor_spec

dataset_path = '/home/docker/irl_control_container/libraries/algorithms/ibc/data/d4rl/car2/2d_oracle_car_3.tfrecord'
spec_path = '/home/docker/irl_control_container/libraries/algorithms/ibc/data/d4rl/car2/2d_oracle_car_3.tfrecord.spec'
sequence_length = 2
replay_capacity = 1000
batch_size = 512
for_rnn = False
dataset_eval_fraction = 0.0
flatten_action = True
_test_path = '/home/docker/irl_control_container/libraries/algorithms/ibc'

env_name = 'MountainCarContinuous-v0'
env = suite_gym.load(env_name)
policy = car_oracles.ParticleOracle(env)
tensor_data_spec = policy.collect_data_spec
_array_data_spec = tensor_spec.to_nest_array_spec(tensor_data_spec)
_encoder = example_encoding.get_example_serializer(
        _array_data_spec,
        compress_image=True,
        image_quality=95)
import pdb;pdb.set_trace()


# create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
#       dataset_path,
#       sequence_length,
#       replay_capacity,
#       batch_size,
#       for_rnn,
#       dataset_eval_fraction,
#       flatten_action)

raw_dataset = tf.data.TFRecordDataset(dataset_path)

for raw_record in raw_dataset.take(-1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)