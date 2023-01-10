import numpy as np
import matplotlib.pyplot as plt

act = np.load('./t_dpos_5_object_act.npy',allow_pickle=True)

obs = np.load('./t_dpos_5_grommet.npy',allow_pickle=True)


import pdb;pdb.set_trace()
# ur5left_expert = act[0]
# ur5right_expert = act[1]
# ur5left_ibc = act[2]
# ur5right_ibc = act[3]

# #import pdb;pdb.set_trace()
# for i in range(1,len(ur5left_expert)):
#     ur5left_expert[i] += ur5left_expert[i-1]
#     ur5right_expert[i] += ur5right_expert[i-1]

# for i in range(1,len(ur5left_ibc)):
#     ur5left_ibc[i] += ur5left_ibc[i-1]
#     ur5right_ibc[i] += ur5right_ibc[i-1]

import pdb;pdb.set_trace()

expert_grommet = obs[0]
ibc_grommet = obs[1]

# import pdb;pdb.set_trace()
fig1 = plt.figure()
ax1 = plt.axes(projection='3d')

# ax1.set_title("End effedtor position for object obs space and del pos action")
# ax1.plot3D(ur5right_expert[:,0],ur5right_expert[:,1],ur5right_expert[:,2])
# ax1.plot3D(ur5right_ibc[:,0],ur5right_ibc[:,1],ur5right_ibc[:,2])
# plt.legend(['ur5right_expert','ur5right_ibc'])

# fig2 = plt.figure()
# ax2 = plt.axes(projection='3d')
# ax2.set_title("End effedtor position for object obs space and del pos action")
# ax2.plot3D(ur5left_expert[:,0],ur5left_expert[:,1],ur5left_expert[:,2])
# ax2.plot3D(ur5left_ibc[:,0],ur5left_ibc[:,1],ur5left_ibc[:,2])
# plt.legend(['ur5left_expert','ur5left_ibc'])
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")
# ax2.set_zlabel("Z")

ax1.set_title("Female peg trajectory (from observations)") ,"ur5left_expert","ur5left_ibc"
ax1.plot3D(expert_grommet[:,0],expert_grommet[:,1],expert_grommet[:,2])
ax1.plot3D(ibc_grommet[:,0],ibc_grommet[:,1],ibc_grommet[:,2])
plt.legend(['expert_trajectory','ibc_trajectory'])

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")


plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# plt.show()


# import pdb;pdb.set_trace()

