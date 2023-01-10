import pandas as pd
from proto_tools.proto_logger import extract_samples_from_expert
import numpy as np
import matplotlib.pyplot as plt

expert_proto = "/home/docker/irl_control_container/data/expert_trajectories/bimanual_object_t_dpos/dual_insert_3YyV8gp.proto"
ibc_proto =  "/home/docker/irl_control_container/data/ibc_eval/bimanual_object_t_dpos/2023-01-05_16:15:48/Trajectories/algo=ibc,run=0.proto"

expert_obs,expert_act,_,_ = extract_samples_from_expert(expert_proto)
ibc_obs,ibc_act,_,_ = extract_samples_from_expert(ibc_proto)

import pdb;pdb.set_trace()
expert_obs = expert_obs[0]
ibc_obs = ibc_obs[0]
expert_act = expert_act[0]
ibc_act = ibc_act[0]

ur5left_expert = expert_act[:,:3]
ur5right_expert = expert_act[:,3:6]

ur5left_ibc = ibc_act[:,:3]
ur5right_ibc = ibc_act[:,3:6]

expert_grommet = expert_obs[:,0:3]
ibc_grommet = ibc_obs[:,0:3]

export_grommet = np.array([expert_grommet, ibc_grommet]) #,
np.save('./ibc/ibc/utils/t_dpos_tes_grommet',export_grommet)

export_array = np.array([ur5left_expert,ur5right_expert,ur5left_ibc,ur5right_ibc]) #
np.save('./ibc/ibc/utils/t_dpos_tes_object_act',export_array)
#import pdb;pdb.set_trace()


# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot3D(ur5left_expert[0],ur5left_expert[1],ur5left_expert[2])

#import pdb;pdb.set_trace()
