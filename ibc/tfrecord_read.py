import tensorflow as tf
from tf_agents.utils import example_encoding
from tf_agents.utils import example_encoding_dataset
from gail.policyopt import Trajectory, TrajBatch
from utils.proto_tools import proto_logger
import numpy as np

filename = "./ibc/data/d4rl/path_follow_v1/pathfollow_0A0Bk8G.tfrecord"
spec_path = "./ibc/data/d4rl/path_follow_v1/pathfollow_0A0Bk8G.tfrecord.spec"

EXPERT_TRAJ_DIR = "./"


spec = example_encoding_dataset.parse_encoded_spec_from_file(
        spec_path)
decoder = example_encoding.get_example_decoder(spec, batched=False,
                                                 compress_image=True)

raw_dataset = tf.data.TFRecordDataset(filename)

dataset = raw_dataset.map(decoder)
obs = []
act = []
rew = []
for data in dataset.take(-1):
    #data[1]o data[2]a data[4]r
    obs.append(data[1].numpy())
    act.append(data[2].numpy())
    rew.append(data[4].numpy())

obs = np.asarray(obs)  #(len,3)
obsfeat_T_Df = np.ones((obs.shape[0], 1))*np.nan # assert obsfeat_T_Df.shape[0] == len(obs)
adist_T_Pa = np.ones((obs.shape[0], 1))*np.nan
act = np.asarray(act)  #(len,2)
rew = np.asarray(rew)
tr = Trajectory(obs, obsfeat_T_Df, adist_T_Pa, act, rew)
tb = TrajBatch.FromTrajs([tr])

fname = f"{EXPERT_TRAJ_DIR}/path_follow_v1.proto"
print(f"Total Reward: {sum(rew)}")
proto_logger.export_samples_from_expert(tb, [obs.shape[0]], fname)

# feature_description = {
#     'step_type': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'observation': tf.io.FixedLenFeature([], tf.float32),
#     'action': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'reward': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#     'discount': tf.io.FixedLenFeature([],tf.int32),
#     'next_step_type': tf.io.FixedLenFeature([],tf.int32),
# }


import pdb;pdb.set_trace()