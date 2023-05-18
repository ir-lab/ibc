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

expert_fname = '/home/docker/irl_control_container/data/expert_trajectories/quad_insert2_v9/quad_insert2_v9.proto'
expert_list = proto_logger.extract_to_trajs(expert_fname)
expert = TrajBatch.FromTrajs(expert_list)
proto_fname = "/home/docker/irl_control_container/data/ibc_eval/quad_insert2_v9/2023-05-17_10:12:28/Trajectories/algo=ibc,train_step=12000,run=0.proto"
traj_list = proto_logger.extract_to_trajs(proto_fname)
tb = TrajBatch.FromTrajs(traj_list)

# fig1,ax = plt.subplots(4)
# fig2,ax2 = plt.subplots(4)

fig3,ax3 = plt.subplots(5,4)
# fig4,ax4 = plt.subplots(5,4)

# fig5 = plt.figure()
# ax5 = plt.axes(projection='3d')

# fig1.suptitle("Expert policy left arm observations")
# fig2.suptitle("Expert policy right arm observations")

fig3.suptitle("IBC policy left arm observations")
# # fig4.suptitle("IBC policy right arm observations")

# fig5.suptitle("IBC policy Trajectory")

outline = [] #np.arange(25,50)  # [5,7,12,16,17] #3d [14,6,7,17,12,13] #[5,7,12,16,17] #  35-45

for episode in range(len(tb.obs)):
    if episode in outline: continue
    ee_x_l = []
    ee_y_l = []
    ee_z_l = []
    ee_x_r = []
    ee_y_r = []
    ee_z_r = []
    obs_val = [0]*28
    for i in range(28):
        obs_val[i] = []
    for step in range(len(tb.obs[episode])):
        # import pdb;pdb.set_trace()
        ee_x_l.append(tb.obs[episode][step][0])
        # ee_y_l.append(tb.obs[episode][step][14])
        # ee_z_l.append(tb.obs[episode][step][15])

        # ee_x_r.append(tb.obs[episode][step][35])
        # ee_y_r.append(tb.obs[episode][step][36])
        # ee_z_r.append(tb.obs[episode][step][37])

        for i in range(28):
            obs_val[i].append(tb.obs[episode][step][i])

    # ax5.plot3D(ee_x_l,ee_y_l,ee_z_l)
    # ax5.plot3D(ee_x_r,ee_y_r,ee_z_r)
    
    step = np.arange(0,len(ee_x_l)) #len(ee_x)

    ax3[0,0].plot(step,obs_val[0])
    ax3[0,1].plot(step,obs_val[1])
    ax3[0,2].plot(step,obs_val[2])
    ax3[0,3].plot(step,obs_val[3])

    ax3[1,0].plot(step,obs_val[4])
    ax3[1,1].plot(step,obs_val[5])
    ax3[1,2].plot(step,obs_val[6])
    ax3[1,3].plot(step,obs_val[7])

    ax3[2,0].plot(step,obs_val[8])
    ax3[2,1].plot(step,obs_val[9])
    ax3[2,2].plot(step,obs_val[10])
    ax3[2,3].plot(step,obs_val[11])

    ax3[3,0].plot(step,obs_val[12])
    ax3[3,1].plot(step,obs_val[13])
    ax3[3,2].plot(step,obs_val[14])
    ax3[3,3].plot(step,obs_val[15])

    ax3[4,0].plot(step,obs_val[16])
    ax3[4,1].plot(step,obs_val[17])
    ax3[4,2].plot(step,obs_val[18])
    # ax3[4,3].plot(step,obs_val[19])




# fig1,ax = plt.subplots(3)
# fig2,ax2 = plt.subplots(3)

# # fig3,ax3 = plt.subplots(3,2)
# # fig4,ax4 = plt.subplots(3,2)

# fig1.suptitle("IBC policy left arm actions")
# fig2.suptitle("IBC policy right arm actions")

# # fig3.suptitle("Expert policy left arm actions")
# # fig4.suptitle("Expert policy right arm actions")
# outline = [6]

# for episode in range(len(tb.a)):
#     if episode in outline: continue
#     l_pos_x = []
#     l_pos_y = []
#     l_pos_z = []
#     a_l = []
#     b_l = []
#     c_l = []
#     d_l = []
#     e_l = []
#     f_l = []
#     r_pos_x = []
#     r_pos_y = []
#     r_pos_z = []
#     a_r = []
#     b_r = []
#     c_r = []
#     d_r = []
#     e_r = []
#     f_r = []
#     for step in range(len(tb.a[episode])):
#         # import pdb;pdb.set_trace()
#         l_pos_x.append(tb.a[episode][step][0])
#         l_pos_y.append(tb.a[episode][step][1])
#         l_pos_z.append(tb.a[episode][step][2])
#         # a_l.append(tb.a[episode][step][3])
#         # b_l.append(tb.a[episode][step][4])
#         # c_l.append(tb.a[episode][step][5])
#         # d_l.append(tb.a[episode][step][6])
#         # e_l.append(tb.a[episode][step][7])
#         # f_l.append(tb.a[episode][step][8])

#         r_pos_x.append(tb.a[episode][step][3])
#         r_pos_y.append(tb.a[episode][step][4])
#         r_pos_z.append(tb.a[episode][step][5])
#         # a_r.append(tb.a[episode][step][12])
#         # b_r.append(tb.a[episode][step][13])
#         # c_r.append(tb.a[episode][step][14])
#         # d_r.append(tb.a[episode][step][15])
#         # e_r.append(tb.a[episode][step][16])
#         # f_r.append(tb.a[episode][step][17])
#         # import pdb;pdb.set_trace()
#     step = np.arange(0,len(l_pos_x))
#     # print("Episode length : ",len(step))
#     ax[0].plot(step,l_pos_x)
#     ax[0].set(ylabel='del x')

#     ax[1].plot(step,l_pos_y)
#     ax[1].set(ylabel='del y')

#     ax[2].plot(step,l_pos_z)
#     ax[2].set(xlabel='step',ylabel='del z')

#     ax2[0].plot(step,r_pos_x)
#     ax2[0].set(ylabel='del x')

#     ax2[1].plot(step,r_pos_y)
#     ax2[1].set(ylabel='del y')

#     ax2[2].plot(step,r_pos_z)
#     ax2[2].set(xlabel='step',ylabel='del z')

#     ax3[0,0].plot(step,a_l)
#     ax3[0,0].set(ylabel='del M1')

#     ax3[0,1].plot(step,b_l)
#     ax3[0,1].set(ylabel='del M2')

#     ax3[1,0].plot(step,c_l)
#     ax3[1,0].set(ylabel='del M3')

#     ax3[1,1].plot(step,d_l)
#     ax3[1,1].set(ylabel='del M4')

#     ax3[2,0].plot(step,e_l)
#     ax3[2,0].set(xlabel='step',ylabel='del M5')

#     ax3[2,1].plot(step,f_l)
#     ax3[2,1].set(xlabel='step',ylabel='del M6')

#     ax4[0,0].plot(step,a_r)
#     ax4[0,0].set(ylabel='del M1')

#     ax4[0,1].plot(step,b_r)
#     ax4[0,1].set(ylabel='del M2')

#     ax4[1,0].plot(step,c_r)
#     ax4[1,0].set(ylabel='del M3')

#     ax4[1,1].plot(step,d_r)
#     ax4[1,1].set(ylabel='del M4')

#     ax4[2,0].plot(step,e_r)
#     ax4[2,0].set(xlabel='step',ylabel='del M5')

#     ax4[2,1].plot(step,f_r)
#     ax4[2,1].set(xlabel='step',ylabel='del M6')

outline = []
rewards = []
for episode in range(len(tb.obs)):
    if episode in outline: continue
    total_r = 0
    for step in range(len(tb.obs[episode])):
        total_r += tb.r[episode][step]
    print("reward : ",total_r)
    rewards.append(total_r)

ex_rewards = []
for episode in range(len(expert.obs)):
    total_r = 0
    for step in range(len(expert.obs[episode])):
        total_r += expert.r[episode][step]
    ex_rewards.append(total_r)

mean_r = np.mean(rewards)
mean_ex_r = np.mean(ex_rewards)

std_r = np.std(rewards)
std_ex_r = np.std(ex_rewards)
# import pdb;pdb.set_trace()
x_pos = np.arange(2)
r = [mean_ex_r, mean_r]
error = [std_ex_r,std_r]

labels = ['Expert', 'IBC']
fig,ax = plt.subplots()
ax.bar(x_pos, r, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Reward')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.yaxis.grid(True)

plt.tight_layout()





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
