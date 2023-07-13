import numpy as np
import matplotlib.pyplot as plt
from utils.proto_tools import proto_logger
from collections import OrderedDict as Dict
from gail.policyopt import TrajBatch
import seaborn as sns
import matplotlib.animation as animation

proto_fname = "/home/docker/irl_control_container/data/ibc_eval/quad_insert2_v11/2023-07-11_09:44:16/Trajectories/algo=ibc,checkpoint=12000,run=2.proto"
traj_list = proto_logger.extract_to_trajs(proto_fname)
tb = TrajBatch.FromTrajs(traj_list)

success_count = 0
for episode in range(len(tb.r)):
    if len(tb.r[episode]) < 1300:
        success_count += 1
    print("Episode : ",episode)
    print("Length of episode: ",len(tb.r[episode]))
    print("Reward : ",sum(tb.r[episode]))
    print("#####")

success_ratio = success_count/len(tb.r)
print("#####")
print("success_ratio : ",success_ratio)
