# coding=utf-8
# Copyright 2024 The Reach ML Authors.
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

"""Multimodal block environments for the XArm.

Vendored from google-research/ibc and adapted to:
  - relative imports (no `ibc.environments.*`)
  - the stubbed `gin` injected by `simulations.ibc_block_pushing.__init__`
  - the registry-compat pattern used by `block_pushing_discontinuous.py`
  - no `block_pushing_metrics` (tf-agents) dependency; success is exposed
    via the `succeeded` property
"""
import collections
import math
from typing import Dict, List

import gin  # stubbed by simulations.ibc_block_pushing.__init__
from gym import spaces
from gym.envs import registration
from . import block_pushing
from .utils import utils_pybullet
from .utils.pose3d import Pose3d
from .utils.utils_pybullet import ObjState
from .utils.utils_pybullet import XarmState
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

# pytype: skip-file
BLOCK2_URDF_PATH = 'third_party/py/ibc/environments/assets/block2.urdf'
ZONE2_URDF_PATH = 'third_party/py/ibc/environments/assets/zone2.urdf'

# When resetting multiple targets, they should all be this far apart.
MIN_BLOCK_DIST = 0.1
MIN_TARGET_DIST = 0.12
NUM_RESET_ATTEMPTS = 1000


@gin.configurable
def build_env_name(task, shared_memory, use_image_obs):
  """Construct the env name from parameters."""
  del task
  env_name = 'BlockPushMultimodal'

  if use_image_obs:
    env_name = env_name + 'Rgb'

  if shared_memory:
    env_name = 'Shared' + env_name

  env_name = env_name + '-v0'

  return env_name


@gin.configurable
class BlockPushMultimodal(block_pushing.BlockPush):
  """2 blocks, 2 targets."""

  def __init__(self,
               control_frequency=10.0,
               task=block_pushing.BlockTaskVariant.PUSH,
               image_size=None,
               shared_memory=False,
               seed=None,
               goal_dist_tolerance=0.04):
    self._target_ids = None
    self._target_poses = None
    super(BlockPushMultimodal, self).__init__(
        control_frequency=control_frequency,
        task=task,
        image_size=image_size,
        shared_memory=shared_memory,
        seed=seed,
        goal_dist_tolerance=goal_dist_tolerance)

  @property
  def target_poses(self):
    return self._target_poses

  def get_goal_translation(self):
    """Return the translation component of the goal (2D)."""
    if self._target_poses:
      return [i.translation for i in self._target_poses]
    else:
      return None

  def _setup_pybullet_scene(self):
    self._pybullet_client = bullet_client.BulletClient(self._connection_mode)

    # Temporarily disable rendering to speed up loading URDFs.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self._setup_workspace_and_robot()

    self._target_ids = [
        utils_pybullet.load_urdf(self._pybullet_client, i, useFixedBase=True)
        for i in [block_pushing.ZONE_URDF_PATH, ZONE2_URDF_PATH]]
    self._block_ids = []
    for i in [block_pushing.BLOCK_URDF_PATH, BLOCK2_URDF_PATH]:
      self._block_ids.append(utils_pybullet.load_urdf(
          self._pybullet_client, i, useFixedBase=False))

    # Re-enable rendering.
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    self.step_simulation_to_stabilize()

  def _reset_block_poses(self, workspace_center_x):
    """Resets block poses."""

    def _reset_block_pose(idx, avoid=None):
      def _get_random_translation():
        block_x = workspace_center_x + self._rng.uniform(low=-0.1, high=0.1)
        block_y = -0.2 + self._rng.uniform(low=-0.15, high=0.15)
        block_translation = np.array([block_x, block_y, 0])
        return block_translation

      if avoid is None:
        block_translation = _get_random_translation()
      else:
        for _ in range(NUM_RESET_ATTEMPTS):
          block_translation = _get_random_translation()
          dist = np.linalg.norm(block_translation[0] - avoid[0])
          if dist > MIN_BLOCK_DIST:
            break
      block_sampled_angle = self._rng.uniform(math.pi)
      block_rotation = transform.Rotation.from_rotvec(
          [0, 0, block_sampled_angle])
      self._pybullet_client.resetBasePositionAndOrientation(
          self._block_ids[idx], block_translation.tolist(),
          block_rotation.as_quat().tolist())
      return block_translation

    for _ in range(NUM_RESET_ATTEMPTS):
      b0_translation = _reset_block_pose(0)
      b1_translation = _reset_block_pose(1, avoid=b0_translation)
      dist = np.linalg.norm(b0_translation[0] - b1_translation[0])
      if dist > MIN_BLOCK_DIST:
        break
    else:
      raise ValueError('could not find matching block')
    assert dist > MIN_BLOCK_DIST

  def _reset_target_poses(self, workspace_center_x):
    """Resets target poses."""
    def _reset_target_pose(idx, avoid=None):
      def _get_random_translation():
        target_x = workspace_center_x + self._rng.uniform(low=-0.10, high=0.10)
        target_y = 0.2 + self._rng.uniform(low=-0.15, high=0.15)
        target_translation = np.array([target_x, target_y, 0.020])
        return target_translation

      if avoid is None:
        target_translation = _get_random_translation()
      else:
        for _ in range(NUM_RESET_ATTEMPTS):
          target_translation = _get_random_translation()
          dist = np.linalg.norm(target_translation[0] - avoid[0])
          if dist > MIN_TARGET_DIST:
            break
      target_sampled_angle = math.pi + self._rng.uniform(
          low=-math.pi / 6, high=math.pi / 6)
      target_rotation = transform.Rotation.from_rotvec(
          [0, 0, target_sampled_angle])
      self._pybullet_client.resetBasePositionAndOrientation(
          self._target_ids[idx], target_translation.tolist(),
          target_rotation.as_quat().tolist())
      self._target_poses[idx] = Pose3d(rotation=target_rotation,
                                       translation=target_translation)
    if self._target_poses is None:
      self._target_poses = [None for _ in range(len(self._target_ids))]

    for _ in range(NUM_RESET_ATTEMPTS):
      _reset_target_pose(0)
      _reset_target_pose(1, avoid=self._target_poses[0].translation)
      dist = np.linalg.norm(self._target_poses[0].translation[0] -
                            self._target_poses[1].translation[0])
      if dist > MIN_TARGET_DIST:
        break
    else:
      raise ValueError('could not find matching target')
    assert dist > MIN_TARGET_DIST

  def reset(self, reset_poses=True):
    workspace_center_x = 0.4

    if reset_poses:
      self._pybullet_client.restoreState(self._saved_state)

      rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
      translation = np.array([0.3, -0.4, block_pushing.EFFECTOR_HEIGHT])
      starting_pose = Pose3d(rotation=rotation, translation=translation)
      self._set_robot_target_effector_pose(starting_pose)

      self._reset_block_poses(workspace_center_x)
      self._reset_target_poses(workspace_center_x)
    else:
      self._target_poses = [
          self._get_target_pose(idx) for idx in self._target_ids]

    if reset_poses:
      self.step_simulation_to_stabilize()

    state = self._compute_state()
    self._previous_state = state
    return state

  def _get_target_pose(self, idx):
    target_translation, target_orientation_quat = (
        self._pybullet_client.getBasePositionAndOrientation(idx))
    target_rotation = transform.Rotation.from_quat(target_orientation_quat)
    target_translation = np.array(target_translation)
    return Pose3d(rotation=target_rotation, translation=target_translation)

  def _compute_reach_target(self, state):
    xy_block = state['block_translation']
    xy_target = state['target_translation']

    xy_block_to_target = xy_target - xy_block
    xy_dir_block_to_target = (
        xy_block_to_target) / np.linalg.norm(xy_block_to_target)
    self.reach_target_translation = (xy_block + -1
                                     * xy_dir_block_to_target * 0.05)

  def _compute_state(self):
    effector_pose = self._robot.forward_kinematics()

    def _get_block_pose(idx):
      block_position_and_orientation = self._pybullet_client.getBasePositionAndOrientation(
          self._block_ids[idx])
      block_pose = Pose3d(
          rotation=transform.Rotation.from_quat(
              block_position_and_orientation[1]),
          translation=block_position_and_orientation[0])
      return block_pose

    block_poses = [_get_block_pose(i) for i in range(len(self._block_ids))]

    def _yaw_from_pose(pose):
      return np.array([pose.rotation.as_euler('xyz', degrees=False)[-1]])

    obs = collections.OrderedDict(
        block_translation=block_poses[0].translation[0:2],
        block_orientation=_yaw_from_pose(block_poses[0]),
        block2_translation=block_poses[1].translation[0:2],
        block2_orientation=_yaw_from_pose(block_poses[1]),
        effector_translation=effector_pose.translation[0:2],
        effector_target_translation=self._target_effector_pose.translation[0:2],
        target_translation=self._target_poses[0].translation[0:2],
        target_orientation=_yaw_from_pose(self._target_poses[0]),
        target2_translation=self._target_poses[1].translation[0:2],
        target2_orientation=_yaw_from_pose(self._target_poses[1]))
    if self._image_size is not None:
      obs['rgb'] = self._render_camera(self._image_size)
    return obs

  def step(self, action):
    self._step_robot_and_sim(action)

    state = self._compute_state()
    done = False
    reward = self._get_reward(state)
    if reward > 0.:
      done = True
    return state, reward, done, {}

  def _get_reward(self, state):
    # Reward is 1. if both blocks are inside targets, but not the same target.
    targets = ['target', 'target2']

    def _block_target_dist(block, target):
      return np.linalg.norm(state['%s_translation' % block]
                            - state['%s_translation' % target])

    def _closest_target(block):
      dists = [_block_target_dist(block, t) for t in targets]
      closest_target = targets[np.argmin(dists)]
      closest_dist = np.min(dists)
      in_target = closest_dist < self.goal_dist_tolerance
      return closest_target, in_target

    b0_closest_target, b0_in_target = _closest_target('block')
    b1_closest_target, b1_in_target = _closest_target('block2')
    reward = 0.
    if b0_in_target and b1_in_target and (
        b0_closest_target != b1_closest_target):
      reward = 1.
    return reward

  def _compute_goal_distance(self, state):
    blocks = ['block', 'block2']

    def _target_block_dist(target, block):
      return np.linalg.norm(state['%s_translation' % block]
                            - state['%s_translation' % target])

    def _closest_block_dist(target):
      dists = [_target_block_dist(target, b) for b in blocks]
      closest_dist = np.min(dists)
      return closest_dist

    t0_closest_dist = _closest_block_dist('target')
    t1_closest_dist = _closest_block_dist('target2')
    return np.mean([t0_closest_dist, t1_closest_dist])

  @property
  def succeeded(self):
    state = self._compute_state()
    reward = self._get_reward(state)
    if reward > 0:
      return True
    return False

  def _create_observation_space(self, image_size):
    pi2 = math.pi * 2

    obs_dict = collections.OrderedDict(
        block_translation=spaces.Box(low=-5, high=5, shape=(2,)),
        block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),
        block2_translation=spaces.Box(low=-5, high=5, shape=(2,)),
        block2_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),
        effector_translation=spaces.Box(
            low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
            high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
        ),
        effector_target_translation=spaces.Box(
            low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
            high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
        ),
        target_translation=spaces.Box(low=-5, high=5, shape=(2,)),
        target_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),
        target2_translation=spaces.Box(low=-5, high=5, shape=(2,)),
        target2_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),
    )
    if image_size is not None:
      obs_dict['rgb'] = spaces.Box(
          low=0,
          high=255,
          shape=(image_size[0], image_size[1], 3),
          dtype=np.uint8)
    return spaces.Dict(obs_dict)

  def get_pybullet_state(self):
    """Save pybullet state of the scene."""
    state: Dict[str, List[ObjState]] = {}

    state['robots'] = [
        XarmState.get_bullet_state(
            self._pybullet_client, self.robot.xarm,
            target_effector_pose=self._target_effector_pose,
            goal_translation=None)]

    state['robot_end_effectors'] = []
    if self.robot.end_effector:
      state['robot_end_effectors'].append(
          ObjState.get_bullet_state(
              self._pybullet_client, self.robot.end_effector))

    state['targets'] = []
    if self._target_ids:
      for target_id in self._target_ids:
        state['targets'].append(ObjState.get_bullet_state(
            self._pybullet_client, target_id))

    state['objects'] = []
    for obj_id in self.get_obj_ids():
      state['objects'].append(ObjState.get_bullet_state(
          self._pybullet_client, obj_id))

    return state

  def set_pybullet_state(self, state):
    """Restore pyullet state."""
    assert isinstance(state['robots'][0], XarmState)
    xarm_state: XarmState = state['robots'][0]
    xarm_state.set_bullet_state(self._pybullet_client, self.robot.xarm)
    self._set_robot_target_effector_pose(xarm_state.target_effector_pose)

    def _set_state_safe(obj_state, obj_id):
      if obj_state is not None:
        assert obj_id is not None, 'Cannot set state for missing object.'
        obj_state.set_bullet_state(self._pybullet_client, obj_id)
      else:
        assert obj_id is None, f'No state found for obj_id {obj_id}'

    robot_end_effectors = state['robot_end_effectors']
    _set_state_safe(
        None if not robot_end_effectors else robot_end_effectors[0],
        self.robot.end_effector)

    for target_state, target_id in zip(state['targets'], self._target_ids):
      _set_state_safe(target_state, target_id)

    obj_ids = self.get_obj_ids()
    assert len(state['objects']) == len(obj_ids), 'State length mismatch'
    for obj_state, obj_id in zip(state['objects'], obj_ids):
      _set_state_safe(obj_state, obj_id)

    self.reset(reset_poses=False)


if 'BlockPushMultimodal-v0' in registration.registry:
  del registration.registry['BlockPushMultimodal-v0']
registration.register(
    id='BlockPushMultimodal-v0',
    entry_point=BlockPushMultimodal,
    max_episode_steps=200)

registration.register(
    id='BlockPushMultimodalRgb-v0',
    entry_point=BlockPushMultimodal,
    max_episode_steps=200,
    kwargs=dict(
        image_size=(block_pushing.IMAGE_HEIGHT, block_pushing.IMAGE_WIDTH)))
