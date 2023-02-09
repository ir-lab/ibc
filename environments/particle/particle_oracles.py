# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oracles (experts) for particle tasks."""

import random

from ibc.environments.particle import particle
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
import numpy as np


class ParticleOracle(py_policy.PyPolicy):
  """Oracle moving between two different goals."""

  def __init__(self,
               env,
               wait_at_first_goal = 1,
               multimodal = False,
               goal_threshold = 0.02):
    """Create oracle.

    Args:
      env: Environment.
      wait_at_first_goal: How long to wait at first goal, once you get there.
                          Encourages memory.
      multimodal: If true, go to one or other goal.
      goal_threshold: How close is considered good enough.
    """
    super(ParticleOracle, self).__init__(env.time_step_spec(),
                                         env.action_spec())
    self._env = env
    self._np_random_state = np.random.RandomState(0)

    assert wait_at_first_goal > 0
    self.wait_at_first_goal = wait_at_first_goal
    assert goal_threshold > 0.
    self.goal_threshold = goal_threshold
    self.multimodal = multimodal
    self.scale = 5
    self.step = 0
    freq = 2
    w = 2 * np.pi * freq
    #self.x_wave = np.linspace(0.5,1,20)
    #self.y_wave = (0.1*np.sin(w*self.x_wave)) + 0.5
    self.x_wave = np.array([0.5,0.8,1])
    self.y_wave = np.array([0.5,0.6,0.5])
    self.wave_fn = np.dstack((self.x_wave,self.y_wave))[0]
    self.reset()

  def reset(self):
    self.steps_at_first_goal = 0
    self.fn_indx = 0
    self.goal_order = ['pos_first_goal', 'pos_second_goal']
    if self.multimodal:
      # Choose a random goal order.
      random.shuffle(self.goal_order)

  def _action(self, time_step,
              policy_state):

    if time_step.is_first():
      self.reset()
    self.step += 1
    first_goal_key = self.goal_order[0]
    second_goal_key = self.goal_order[1]
    obs = time_step.observation
    #print("Pos Observation : ",obs['pos_agent'])
    #gt_goals = self._env.obs_log[0]

    # dist = np.linalg.norm(obs['pos_agent'] - np.array([0.5,0.2],dtype = np.float32))  #gt_goals[first_goal_key]
    # if dist < self.goal_threshold:
    #   self.steps_at_first_goal += 1

    # if self.steps_at_first_goal < self.wait_at_first_goal:
    #   act = (np.array([0.5,0.2],dtype = np.float32) - obs['pos_agent']) * self.scale   #gt_goals[first_goal_key]
    #   # print("Goal 1 :",gt_goals[first_goal_key])
    #   #act = np.copy(gt_goals[first_goal_key])
    # else:
    #   act = (np.array([0.9,0.2],dtype = np.float32) - obs['pos_agent']) * self.scale
    #   # print("Goal 2 :",gt_goals[second_goal_key])
    #   #act = np.copy(gt_goals[second_goal_key])
    dist = np.linalg.norm(obs['pos_agent'] - self.wave_fn[self.fn_indx])
    final_dist = np.linalg.norm(obs['pos_agent'] - self.wave_fn[-1])
    if final_dist < self.goal_threshold:
      return policy_step.PolicyStep(action=np.array([0,0],dtype=np.float32))
  
    if dist < self.goal_threshold:
      self.fn_indx += 1
    act = np.array((self.wave_fn[self.fn_indx] - obs['pos_agent']) * self.scale, dtype=np.float32)

    return policy_step.PolicyStep(action=act)
