import gym
from tf_agents.environments import suite_gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time


gym.envs.register(
        id='bimanual-v0',
        entry_point='learning.dual_insertion:DualInsertion',
        max_episode_steps=20000,
    )
start_time = time.time()
env = gym.make('bimanual-v0')
#env = gym.make('Ant-v4')
# video_path = "/home/docker/irl_control_container/data/video"
# video_recorder = VideoRecorder(env,video_path,)

obs = env.reset()
for i in range(2000):
    action = env.action_space.sample()
    obs,reward,done,_ = env.step(action)
    #print("Observation : ",len(obs))
    # video_recorder.capture_frame()

print("Run time: ",time.time()-start_time)

# obs = env.reset()
# for i in range(2000):
#     action = env.action_space.sample()
#     obs,reward,done,_ = env.step(action)
#     # video_recorder.capture_frame()

#[0.10007154 0.6003884  0.02499061]
#pegs :  [ 1.00015635e-01  6.00220269e-01 -2.93255082e-04]

# video_recorder.close()

# obs = env.reset()
# for i in range(2000):
#     action = env.action_space.sample()
#     env.step(action)

