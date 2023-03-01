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
expert = np.load('/home/docker/irl_control_container/libraries/algorithms/ibc/data/particle_3d/particle_tri_dataset.npy',allow_pickle=True)
proto_fname = "/home/docker/irl_control_container/data/ibc_eval/particle_3d/2023-02-16_15:08:10/Trajectories/algo=ibc,train_step=20000,run=0.proto"
traj_list = proto_logger.extract_to_trajs(proto_fname)
tb = TrajBatch.FromTrajs(traj_list)
#import pdb;pdb.set_trace()

# fig1,(ax1,ax2) = plt.subplots(2)

# fig2 = plt.figure()
# ax5 = plt.axes()

# fig1.suptitle("Learned policy Actions")
# fig2.suptitle("Learned policy Trajectory with outliers")

outline = [] #[5,7,12,16,17]#[14,6,7,17,12,13] # [5,7,12,16,17] #3d [14,6,7,17,12,13] #[5,7,12,16,17] # 
# ibc_x = []
# ibc_y = []
# ibc_z = []
# for episode in range(len(tb.obs)):
#     if episode in outline: continue
#     ee_x = []
#     ee_y = []
#     ee_z = []
#     x_vel = []
#     y_vel = []
#     z_vel = []
#     for step in range(len(tb.obs[episode])):
#         #import pdb;pdb.set_trace()
#         ee_x.append(tb.obs[episode][step][0])
#         ee_y.append(tb.obs[episode][step][1])
#         #ee_z.append(tb.obs[episode][step][2])
#         x_vel.append(tb.a[episode][step][0])
#         y_vel.append(tb.a[episode][step][1])
#         #z_vel.append(tb.a[episode][step][1])
#     ibc_x.append(ee_x)
#     ibc_y.append(ee_y)
    #ibc_z.append(ee_z)
    # step = np.arange(0,len(x_vel))
    # #import pdb;pdb.set_trace()

    # ax1.plot(step,x_vel)
    # ax2.plot(step,y_vel)

    # ax1.set(xlabel='step',ylabel='x velocity')
    # ax2.set(xlabel='step',ylabel='y velocity')


    # ax5.plot(ee_x,ee_y)
    # ax5.set(xlabel='x',ylabel='y')
    # #ax2.set(xlim=(0, 1), ylim=(0, 1))
    # ax5.plot(1,0.2,'x')

episode = 4 #5,7,12,16



ee_x = []
ee_y = []
ee_z = []
x_vel = []
y_vel = []
z_vel = []
for step in range(len(tb.obs[episode])):
    #import pdb;pdb.set_trace()
    ee_x.append(tb.obs[episode][step][0])
    ee_y.append(tb.obs[episode][step][1])
    ee_z.append(tb.obs[episode][step][2])

pos_x = []
pos_y = []
pos_z = []
epi = 0
for step in range(len(expert[epi][obs])):
    pos_x.append(expert[epi][obs][step][0][0])
    pos_y.append(expert[epi][obs][step][0][1])
    pos_z.append(expert[epi][obs][step][0][2])
# ax.plot(ee_x,ee_y,'b',label="IBC")


for n in range(len(ee_x)):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    line, = ax.plot3D(ee_x, ee_y, ee_z,label="IBC")
    ax.plot3D(pos_x,pos_y,pos_z,'r',label='Expert')
    ax.plot3D(ee_x[:n], ee_y[:n],ee_z[:n],'b',label='IBC')
    ax.legend()
    fig.canvas.draw()
    fig.savefig('Frame%03d.png' %n)
    #fig.savefig('Frame%03d.png' %n)

from PIL import image
images = [Image.open(f"{n}.png") for n in range(len(ee_x))]

images[0].save('ball.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
# x = np.linspace(0,1,76)
# mean1 = np.mean(y_pos,axis=0)
# std1 = np.var(y_pos,axis=0)
# plt.plot(x,mean1,'b-')
# plt.fill_between(x, mean1 - std1, mean1 + std1, color='b', alpha=0.2)

#fig,ax = plt.subplots()

# fig1, (ax1,ax2,ax3) = plt.subplots(3)
# fig2 =  plt.figure()
# ax5 = plt.axes(projection ='3d')
# fig1.suptitle("Expert policy Actions")
# fig2.suptitle("Expert policy Trajectory")
# expert_x = []
# expert_y = []
# expert_z = []
# for episode in range(len(expert)):
#     pos_x = []
#     pos_y = []
#     pos_z = []
#     # vel_x = []
#     # vel_y = []
#     # vel_z = []
#     for step in range(len(expert[episode][obs])):
#         pos_x.append(expert[episode][obs][step][0][0])
#         pos_y.append(expert[episode][obs][step][0][1])
#         pos_z.append(expert[episode][obs][step][0][2])
#         # vel_x.append(expert[episode][act][step][0][0])
#         # vel_y.append(expert[episode][act][step][0][1])
#         # vel_z.append(expert[episode][act][step][0][2])
#     expert_x.append(pos_x)
#     expert_y.append(pos_y)
#     expert_z.append(pos_z)
#     # step = np.arange(0,len(vel_x))
#     # #import pdb;pdb.set_trace()
#     # ax1.plot(step,vel_y)
#     # ax2.plot(step,vel_y)
#     # ax3.plot(step,vel_z)

#     # ax1.set(xlabel='step',ylabel='x velocity')
#     # ax2.set(xlabel='step',ylabel='y velocity')
#     # ax3.set(xlabel='step',ylabel='z velocity')

#     # ax5.plot3D(pos_x,pos_y,pos_z)
#     # ax5.set(xlabel='x',ylabel='y',zlabel='z')

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
# fig1.suptitle("3D Observations without outliers")

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


