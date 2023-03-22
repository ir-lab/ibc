import tensorflow as tf
from tf_agents.utils import example_encoding
from tf_agents.utils import example_encoding_dataset
from tf_agents.specs.tensor_spec import to_pbtxt_file
from gail.policyopt import Trajectory, TrajBatch
from utils.proto_tools import proto_logger
import numpy as np
import os

filename = "./ibc/data/bimanual_single_euler/2d_oracle_particle_0.tfrecord"
spec_path = "./ibc/data/bimanual_single_euler/2d_oracle_particle_0.tfrecord.spec"

# filename = "./ibc/data/block_push_states_location_target/oracle_push_0.tfrecord"
# spec_path = "./ibc/data/block_push_states_location_target/oracle_push_0.tfrecord.spec"

EXPERT_TRAJ_DIR = "./"

root_dir = '/home/docker/irl_control_container/libraries/algorithms/ibc/data/particle_3d'

spec = example_encoding_dataset.parse_encoded_spec_from_file(
        spec_path)

output_path = './ibc/data/bimanual_single_quat/bimanual_single_quat.pbtxt'
to_pbtxt_file(output_path, spec)



decoder = example_encoding.get_example_decoder(spec, batched=False,
                                                 compress_image=True)

raw_dataset = tf.data.TFRecordDataset(filename)

dataset = raw_dataset.map(decoder)


# epi = []
# obs = []
# act = []
# rew = []
# episode = 0
# for data in dataset.take(-1):
#     if data[0].numpy() == 0:
#         obs = []
#         act = []
#         rew = []
#     obs.append([data[1]['pos_agent'].numpy(),data[1]['vel_agent'].numpy()])
#     # ,data[1]['pos_first_goal'].numpy(),data[1]['pos_second_goal'].numpy()
#     # obs.append([data[1]['effector_translation'].numpy(),data[1]['block_translation'].numpy()])#,data[1]['target_translation'].numpy()])
#     act.append([data[2].numpy()])
#     rew.append([data[5].numpy()])
#     if data[0].numpy() == 2:
#         epi.append([obs,act,rew])
#     #data[1]o data[2]a data[4]r
#     # obs.append(data[1].numpy())
#     # act.append(data[2].numpy())
#     # rew.append(data[4].numpy())
# epi = np.asarray(epi)
# npy_name = os.path.join(root_dir,f"particle_tri_dataset") #irl_control_container/libraries/algorithms/ibc/data/particle_vel
# np.save(npy_name,epi)


