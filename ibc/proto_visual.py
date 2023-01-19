from proto_tools import proto_logger
import matplotlib.pyplot as plt

path = "/home/docker/irl_control_container/data/expert_trajectories/path_demo_storage/path_demo_0A0ywP5.proto"

obs,act,_,_ = proto_logger.extract_samples_from_expert(path)

obs = obs[0]
act = act[0]

plt.plot(obs[0],obs[1])
plt.show()

import pdb;pdb.set_trace()