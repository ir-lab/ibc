import tensorflow as tf
from tf_agents.utils import example_encoding
from tf_agents.specs import tensor_spec
from utils.proto_tools.build.py.trajectory_pb2 import trajectory as proto_trajectory
from tf_agents.trajectories import Trajectory

from hashids import Hashids

import functools
import os
import shutil
import numpy as np
import calendar
import time
import random

from absl import app
from absl import flags
from absl import logging

# raw_dataset = tf.data.TFRecordDataset(dataset_path).cache().repeat()
# import pdb; pdb.set_trace()
# for raw_record in raw_dataset.take(2):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)

def encode_spec_to_file(output_path, tensor_data_spec):
  """Save a tensor data spec to a tfrecord file.
  Args:
    output_path: The path to the TFRecord file which will contain the spec.
    tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
      shape of the non-batched Tensors.
  """
  spec_proto = tensor_spec.to_proto(tensor_data_spec)
  with tf.io.TFRecordWriter(output_path) as writer:
    writer.write(spec_proto.SerializeToString())

class TFRecorder(object):
  """Observer for writing experience to TFRecord file.
  To use this observer, create an instance using a trajectory spec object
  and a dataset path:
  trajectory_spec = agent.collect_data_spec
  dataset_path = '/tmp/my_example_dataset'
  tfrecord_observer = TFRecordObserver(dataset_path, trajectory_spec)
  Then add it to the observers kwarg for the driver:
  collect_op = MyDriver(
    ...
    observers=[..., tfrecord_observer],
    num_steps=collect_steps_per_iteration).run()
  *Note*: Depending on your driver you may have to do
    `common.function(tfrecord_observer)` to handle the use of a callable with no
    return within a `tf.group` operation.
  """

  def __init__(self,
               output_path,
               tensor_data_spec,
               py_mode=False,
               compress_image=False,
               image_quality=95):
    """Creates observer object.
    Args:
      output_path: The path to the TFRecords file.
      tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
        shape of the non-batched Tensors.
      py_mode: Whether the observer is being used in a py_driver.
      compress_image: Whether to compress image. It is assumed that any uint8
        tensor of rank 3 with shape (w,h,c) is an image.
      image_quality: An optional int. Defaults to 95. Quality of the compression
        from 0 to 100 (higher is better and slower).
    Raises:
      ValueError: if the tensors and specs have incompatible dimensions or
      shapes.
    """
    _SPEC_FILE_EXTENSION = '.spec'
    self._py_mode = py_mode
    self._array_data_spec = tensor_spec.to_nest_array_spec(tensor_data_spec)
    self._encoder = example_encoding.get_example_serializer(
        self._array_data_spec,
        compress_image=compress_image,
        image_quality=image_quality)
    # Two output files: a tfrecord file and a file with the serialized spec
    self.output_path = output_path
    tf.io.gfile.makedirs(os.path.dirname(self.output_path))
    self._writer = tf.io.TFRecordWriter(self.output_path)
    logging.info('Writing dataset to TFRecord at %s', self.output_path)
    # Save the tensor spec used to write the dataset to file
    spec_output_path = self.output_path + _SPEC_FILE_EXTENSION
    encode_spec_to_file(spec_output_path, tensor_data_spec)

  def write(self, *data):
    """Encodes and writes (to file) a batch of data.
    Args:
      *data: (unpacked) list/tuple of batched np.arrays.
    """
    dataspec = tf.data.DatasetSpec(data, dataset_shape=())
    if self._py_mode:
      structured_data = data
    else:
      data = nest_utils.unbatch_nested_array(data)
      structured_data = tf.nest.pack_sequence_as(self._array_data_spec, data)
    self._writer.write(self._encoder(structured_data))

  def flush(self):
    """Manually flush TFRecord writer."""
    self._writer.flush()

  def close(self):
    """Close the TFRecord writer."""
    self._writer.close()
    logging.info('Closing TFRecord file at %s', self.output_path)

  def __call__(self, data):
    """If not in py_mode Wraps write() into a TF op for eager execution."""
    if self._py_mode:
      self.write(data)
    else:
      flat_data = tf.nest.flatten(data)
      tf.numpy_function(self.write, flat_data, [], name='encoder_observer')

def export_to_tfrecord(proto_file):
  print("Exporting : ",proto_file)
  f = open(proto_file, "rb")
  traj = proto_trajectory()
  traj.ParseFromString(f.read())
  f.close()
  # import pdb; pdb.set_trace()
  #hash = hashids.encode(calendar.timegm(time.gmtime())+random.randint(0,1000))
  file_path = tfrecord_path + f".tfrecord"
  recorder = TFRecorder(
            file_path,
            dataspec,
            py_mode=True,
            compress_image=True)
  for idx in range(len(traj.observations)):
    obs = np.vstack([x.value for x in traj.observations[idx].sub_lists]).T
    act = np.vstack([x.value for x in traj.actions[idx].sub_lists]).T
    # import pdb;pdb.set_trace()
    rewards = traj.rewards[idx].value

    for index in range(len(obs)):
      if index == 0:
        step_type = 0
        next_step_type = 1
      elif index == len(obs)-2:
        step_type = 1
        next_step_type = 2
      elif index == len(obs)-1:
        step_type = 2
        next_step_type = 0
      else:
        step_type = 1
        next_step_type = 1
      tensor_traj = Trajectory(step_type=np.array(step_type,dtype=np.int32),
                            observation=np.array(obs[index],dtype=np.float32),
                            action=np.array(act[index],dtype=np.float32),
                            policy_info=(),
                            next_step_type=np.array(next_step_type,dtype=np.int32),
                            reward=np.array(rewards[index],dtype=np.float32),
                            discount=np.array(1,dtype=np.float32))
      recorder(tensor_traj)

tfrecord_path = "/home/docker/irl_control_container/libraries/algorithms/ibc/data/dual_insert_v3/dual_insert_v3_quat"

spec_path= "/home/docker/irl_control_container/libraries/algorithms/ibc/data/dual_insert_v3/bimanual_dual_insert_v3_quat.pbtxt"
dataset_path = "/home/docker/irl_control_container/data/expert_trajectories/dual_insert_v3/dual_insert_v3_reduced_quat.proto"
dataspec = tensor_spec.from_pbtxt_file(spec_path)

proto_files = tf.io.gfile.glob(dataset_path)
hashids = Hashids()

for proto_file in proto_files:
  export_to_tfrecord(proto_file)
