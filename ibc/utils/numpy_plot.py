import numpy as np
import matplotlib.pyplot as plt
from utils.proto_tools import proto_logger
from collections import OrderedDict as Dict
from gail.policyopt import TrajBatch
import seaborn as sns
import matplotlib.animation as animation

obs = 0
act = 1
sns.set()
#act = np.load('./t_dpos_5_object_act.npy',allow_pickle=True)
#
#obs = np.load('./t_dpos_5_grommet.npy',allow_pickle=True)
#x = np.arange(100,60001,100)
#                 /home/docker/irl_control_container/libraries/algorithms/ibc/data/particle/particel_dataset.npy
#expert = np.load('/home/docker/irl_control_container/libraries/algorithms/ibc/data/particle_3d/particle_tri_dataset.npy',allow_pickle=True)

expert_fname = '/home/docker/irl_control_container/data/expert_trajectories/bimanual_v1/bimanual_v1_b0.proto'
expert_list = proto_logger.extract_to_trajs(expert_fname)
expert = TrajBatch.FromTrajs(expert_list)
proto_fname = "/home/docker/irl_control_container/data/ibc_eval/bimanual_v1/2023-03-03_00:02:22/Trajectories/algo=ibc,train_step=1000,run=0.proto"
traj_list = proto_logger.extract_to_trajs(proto_fname)
tb = TrajBatch.FromTrajs(traj_list)
#import pdb;pdb.set_trace()

fig1,ax = plt.subplots(3,2)

fig2 = plt.figure()
ax5 = plt.axes(projection='3d')

fig1.suptitle("IBC policy Observations")
fig2.suptitle("Expert policy Trajectory")

outline = []  # [5,7,12,16,17] #3d [14,6,7,17,12,13] #[5,7,12,16,17] # 
ibc_x = []
ibc_y = []
ibc_z = []
ibc_r = []
ibc_p = []
ibc_a = []
for episode in range(len(tb.obs)):
    if episode in outline: continue
    ee_x = []
    ee_y = []
    ee_z = []
    radius = []
    polar = []
    azi = []
    for step in range(len(tb.obs[episode])):
        #import pdb;pdb.set_trace()
        ee_x.append(tb.obs[episode][step][0])
        ee_y.append(tb.obs[episode][step][1])
        ee_z.append(tb.obs[episode][step][2])
        radius.append(tb.obs[episode][step][3])
        polar.append(tb.obs[episode][step][4])
        azi.append(tb.obs[episode][step][5])
    ax5.plot3D(ee_x,ee_y,ee_z)
    
    ibc_x.append(ee_x)
    ibc_y.append(ee_y)
    ibc_z.append(ee_z)
    ibc_r.append(radius)
    ibc_p.append(polar)
    ibc_a.append(azi)
    step = np.arange(0,len(ee_x))
    # #import pdb;pdb.set_trace()

    ax[0,0].plot(step,ee_x)
    ax[0,0].set_title("x position")
    ax[0,0].set(ylabel='x')

    ax[1,0].plot(step,ee_y)
    ax[1,0].set_title("y position")
    ax[1,0].set(ylabel='y')

    ax[2,0].plot(step,ee_z)
    ax[2,0].set_title("z position")
    ax[2,0].set(xlabel='step',ylabel='z')

    ax[0,1].plot(step,radius)
    ax[0,1].set_title("Radius")
    ax[0,1].set(ylabel='r')

    ax[1,1].plot(step,polar)
    ax[1,1].set_title("polar")
    ax[1,1].set(ylabel='angle')
    
    ax[2,1].plot(step,azi)
    ax[2,1].set_title("azimuth")
    ax[2,1].set(xlabel='step',ylabel='angle')

# rewards = []
# for episode in range(len(tb.obs)):
#     total_r = 0
#     for step in range(len(tb.obs[episode])):
#         total_r += tb.r[episode][step]
#     rewards.append(total_r)

# ex_rewards = []
# for episode in range(len(expert.obs)):
#     total_r = 0
#     for step in range(len(expert.obs[episode])):
#         total_r += expert.r[episode][step]
#     ex_rewards.append(total_r)

# mean_r = np.mean(rewards)
# mean_ex_r = np.mean(ex_rewards)

# std_r = np.std(rewards)
# std_ex_r = np.std(ex_rewards)

# x_pos = np.arange(2)
# r = [mean_ex_r, mean_r]
# error = [std_ex_r,std_r]

# labels = ['Expert', 'IBC']
# fig,ax = plt.subplots()
# ax.bar(x_pos, r, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.set_ylabel('Reward')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(labels)
# ax.yaxis.grid(True)

# plt.tight_layout()





# ibc_xmean = np.mean(ibc_x,axis=0)
# ibc_ymean = np.mean(ibc_y,axis=0)
# ibc_zmean = np.mean(ibc_z,axis=0)

# ibc_xvar = np.var(ibc_x,axis=0)
# ibc_yvar = np.var(ibc_y,axis=0)
# ibc_zvar = np.var(ibc_z,axis=0)

# expert_xmean = np.mean(expert_x,axis=0)
# expert_ymean = np.mean(expert_y,axis=0)
# expert_zmean = np.mean(expert_z,axis=0)

# expert_xvar = np.var(expert_x,axis=0)
# expert_yvar = np.var(expert_y,axis=0)
# expert_zvar = np.var(expert_z,axis=0)

# x_step = np.arange(0,len(ibc_xmean))

# fig1, (ax1,ax2,ax3) = plt.subplots(3)
# fig1.suptitle("3D Observations with outliers")

# ax1.plot(x_step,ibc_xmean,'b--',label='IBC')
# ax1.fill_between(x_step, ibc_xmean - ibc_xvar, ibc_xmean + ibc_xvar, color='b',alpha=0.5)
# ax1.plot(x_step,expert_xmean,'r',label='Expert')
# ax1.fill_between(x_step, expert_xmean - expert_xvar, expert_xmean + expert_xvar, color='r',alpha=0.5)
# ax1.set(xlabel='step',ylabel='x position')
# ax1.legend()

# ax2.plot(x_step,ibc_ymean,'b--',label="IBC")
# ax2.fill_between(x_step, ibc_ymean - ibc_yvar, ibc_ymean + ibc_yvar, color='b',alpha=0.5)
# ax2.plot(x_step,expert_ymean,'r',label="Expert")
# ax2.fill_between(x_step, expert_ymean - expert_yvar, expert_ymean + expert_yvar, color='r',alpha=0.5)
# ax2.set(xlabel='step',ylabel='y position')
# ax2.legend()

# ax3.plot(x_step,ibc_zmean,'b--',label="IBC")
# ax3.fill_between(x_step, ibc_zmean - ibc_zvar, ibc_zmean + ibc_zvar, color='b',alpha=0.5)
# ax3.plot(x_step,expert_zmean,'r',label="Expert")
# ax3.fill_between(x_step, expert_zmean - expert_zvar, expert_zmean + expert_zvar, color='r',alpha=0.5)
# ax3.set(xlabel='step',ylabel='z position')
# ax3.legend()

plt.show()


