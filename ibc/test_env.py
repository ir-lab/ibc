import gym
from tf_agents.environments import suite_gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time


# gym.envs.register(
#         id='pathfollow-v1',
#         entry_point='imitation.environments.path_follow_v1.path_follow_v1:PathFollowV1',
#         max_episode_steps=20000,
#     )
start_time = time.time()
env = gym.make('quad_insert2_v11')
export_dir = "/home/docker/irl_control_container/libraries/algorithms/ibc/gif"
export_prefix = 'test_run'
export_suffix = 'init'
env.set_gif_recording(export_dir,export_prefix,export_suffix)

# video_path = "/home/docker/irl_control_container/data/video"
# video_recorder = VideoRecorder(env,video_path,)

obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs,reward,done,_ = env.step(action)
    if i == 200 or i == 300:
        env.export_gif_recording()
        obs = env.reset()
        export_suffix = str(i)
        env.set_gif_recording(export_dir,export_prefix,export_suffix)
    #import pdb;pdb.set_trace()  
    # print("Action : ",action)
    # print("Observation : ",obs)
    # video_recorder.capture_frame()
env.export_gif_recording()

print("Run time: ",time.time()-start_time)