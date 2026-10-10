# Copyright 2024 The Brax Authors.
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

"""Dodge environment.

The agent must stay healthy while keeping every body part's hitbox from
intersecting a hazard plane. The plane is defined by a point and a normal (its
orientation; neither side is special) and is tilted randomly every episode.
It can also slide horizontally through the platform at a constant speed.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import jax
import mujoco
import numpy as np
from jax import numpy as jp
from etils import epath

from crax import actuator
from crax import base
from crax import math as brax_math
from crax.envs.base import PipelineEnv, State
from crax.io import mjcf


class SafeDodge(PipelineEnv, ABC):
    """Abstract base class for hazard-plane environments.

    The agent must stay healthy on a round platform while keeping its body-part
    hitboxes out of a hazard plane. The cost is how deep they cut into it.

    Subclasses must implement agent-specific properties for:
    - XML file path
    - Body configuration
    """

    @property
    @abstractmethod
    def agent_xml_path(self) -> str:
        """Return the path to the agent's XML file."""
        pass

    @property
    @abstractmethod
    def default_spawn_height(self) -> float:
        """Return the default z-position for spawning the agent."""
        pass

    @property
    @abstractmethod
    def default_spawn_rotation(self) -> Tuple[float, float, float, float]:
        """Return the default quaternion [w, x, y, z] for spawning."""
        pass

    @property
    @abstractmethod
    def default_healthy_z_range(self) -> Tuple[float, float]:
        """Return the default (min, max) z-range for healthy state."""
        pass

    @abstractmethod
    def _get_gear_for_backend(self, backend: str) -> jp.ndarray:
        """Return actuator gear values for the given backend."""
        pass

    def __init__(
            self,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            height_reward_weight: float = 5.0,
            upright_reward_weight: float = 0.0,
            still_cost_weight: float = 1.0,
            terminate_when_unhealthy=True,
            healthy_z_range: Optional[Tuple[float, float]] = None,
            reset_noise_scale=1e-2,
            plane_point: Tuple[float, float, float] = (0.0, 0.0, 1.3),
            plane_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            plane_cost_weight: float = 1.25,
            fall_cost: float = 25.0,
            plane_tilt_range: float = 90.0,
            plane_enabled: bool = True,
            plane_slide_speed: float = 0.0,
            plane_slide_distance: float = 1.5,
            plane_offset_range: float = 0.0,
            plane_height_range: Optional[Tuple[float, float]] = None,
            plane_half_size: Optional[float] = None,
            plane_repeat: bool = False,
            platform_radius: float = 0.75,
            episode_length: int = 1000,
            backend: str = 'generalized',
            **kwargs,
    ):
        """Initialize the hazard-plane environment.

        Args:
            ctrl_cost_weight: Weight for control cost penalty.
            healthy_reward: Reward for staying healthy (alive).
            height_reward_weight: Reward for standing upright, 0 at the fall height.
            upright_reward_weight: Reward for keeping the torso vertical.
            still_cost_weight: Penalty per m/s of horizontal centre-of-mass speed.
            terminate_when_unhealthy: Whether to terminate episode when unhealthy.
            healthy_z_range: (min, max) z-range for healthy state.
            reset_noise_scale: Scale of noise added to initial state.
            plane_point: Any point [x, y, z] on the hazard plane.
            plane_normal: Normal of the untilted hazard plane.
            plane_cost_weight: Cost per metre of hitbox overlap with the plane.
            fall_cost: Cost added when the agent falls.
            plane_tilt_range: Max random tilt of the plane per axis, in degrees.
            plane_enabled: False hides the plane and makes its cost 0.
            plane_slide_speed: Horizontal speed of the plane in m/s (0 = still).
            plane_slide_distance: Distance from plane_point where the sliding plane starts.
            plane_offset_range: Max random horizontal offset of the plane from plane_point.
            plane_height_range: (low, high) random plane height; None keeps plane_point's.
            plane_half_size: Half the side of the square plane; None covers the platform.
            plane_repeat: Start a new pass each time the sliding plane has gone past.
            platform_radius: Radius of the platform; a part on the floor past it is a fall.
            episode_length: Maximum number of steps per episode.
            backend: Physics backend ('generalized', 'spring', 'positional', 'mjx').
        """
        plane_point = np.asarray(plane_point, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        normal_norm = np.linalg.norm(plane_normal)
        if normal_norm == 0.0:
            raise ValueError('plane_normal must be a non-zero vector.')
        plane_normal = plane_normal / normal_norm
        self._plane_cost_weight = plane_cost_weight if plane_enabled else 0.0
        self._fall_cost = fall_cost
        if not 0.0 <= plane_tilt_range <= 90.0:
            raise ValueError('plane_tilt_range must be in [0, 90] degrees.')
        self._plane_tilt_range = jp.deg2rad(plane_tilt_range)
        if plane_slide_speed < 0.0 or plane_slide_distance < 0.0:
            raise ValueError('plane_slide_speed and plane_slide_distance must be >= 0.')
        self._plane_slide_speed = plane_slide_speed
        self._plane_slide_distance = plane_slide_distance if plane_slide_speed > 0.0 else 0.0
        self._plane_offset_range = plane_offset_range
        if plane_height_range is None:
            plane_height_range = (plane_point[2], plane_point[2])
        if not 0.0 < plane_height_range[0] <= plane_height_range[1]:
            raise ValueError('plane_height_range must be (low, high) with 0 < low <= high.')
        self._plane_height_range = (float(plane_height_range[0]), float(plane_height_range[1]))
        self._plane_base_height = float(plane_point[2])
        if plane_repeat and plane_slide_speed <= 0.0:
            raise ValueError('plane_repeat needs a sliding plane (plane_slide_speed > 0).')
        self._plane_repeat = plane_repeat
        if platform_radius <= 0.0:
            raise ValueError('platform_radius must be positive.')
        self._platform_radius = platform_radius

        # Use default healthy z range if not provided
        if healthy_z_range is None:
            healthy_z_range = self.default_healthy_z_range
        if healthy_z_range[0] >= self.default_spawn_height:
            raise ValueError('healthy_z_range min must be below the spawn height.')

        # Load XML and place the hazard plane and platform
        path = epath.resource_path('crax') / self.agent_xml_path
        xml_string = path.read_text()
        xml_string = xml_string.replace('PLANE_POS', ' '.join(str(v) for v in plane_point))
        xml_string = xml_string.replace('PLANE_NORMAL', ' '.join(str(v) for v in plane_normal))
        xml_string = xml_string.replace('PLATFORM_RADIUS', str(float(platform_radius)))
        # By default the plane reaches the platform's edge at any tilt
        if plane_half_size is None:
            plane_half_size = np.hypot(
                platform_radius + np.linalg.norm(plane_point[:2]) + plane_offset_range,
                max(plane_point[2], self._plane_height_range[1]))
        self._plane_half_size = float(plane_half_size)
        xml_string = xml_string.replace('PLANE_HALF_SIZE', str(self._plane_half_size))
        # Group 3 is hidden in renders
        xml_string = xml_string.replace('PLANE_GROUP', '2' if plane_enabled else '3')

        # Parse the modified XML
        sys = mjcf.loads(xml_string)

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.0015})
            n_frames = 10
            gear = self._get_gear_for_backend(backend)
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            })

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self.episode_length = episode_length
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._height_reward_weight = height_reward_weight
        self._upright_reward_weight = upright_reward_weight
        self._still_cost_weight = still_cost_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        # Plane joints and link (brax link i = MuJoCo body i + 1)
        mj_model = self.sys.mj_model
        body_id = lambda name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = lambda name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        plane_bodies = (body_id('plane_slider'), body_id('hazard_plane'))
        self._plane_link = plane_bodies[1] - 1
        pos_joints = [joint_id('plane_slide_x'), joint_id('plane_slide_y'), joint_id('plane_slide_z')]
        tilt_joints = [joint_id('plane_tilt_x'), joint_id('plane_tilt_y')]
        self._plane_pos_q = jp.array(mj_model.jnt_qposadr[pos_joints])
        self._plane_pos_qd = jp.array(mj_model.jnt_dofadr[pos_joints])
        self._plane_tilt_q = jp.array(mj_model.jnt_qposadr[tilt_joints])
        self._plane_tilt_qd = jp.array(mj_model.jnt_dofadr[tilt_joints])

        # Agent's part of the state (everything but the plane)
        plane_joints = pos_joints + tilt_joints
        self._agent_q = jp.array(np.setdiff1d(np.arange(self.sys.q_size()), mj_model.jnt_qposadr[plane_joints]))
        self._agent_qd = jp.array(np.setdiff1d(np.arange(self.sys.qd_size()), mj_model.jnt_dofadr[plane_joints]))
        self._agent_links = jp.array([i for i in range(self.sys.num_links()) if i + 1 not in plane_bodies])

        # Hitboxes: every agent geom
        part_ids = np.array([g for g in range(mj_model.ngeom)
                             if mj_model.geom_bodyid[g] not in (0, *plane_bodies)])
        part_types = mj_model.geom_type[part_ids]
        sphere, capsule, box = (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                int(mujoco.mjtGeom.mjGEOM_BOX))
        if not np.all(np.isin(part_types, (sphere, capsule, box))):
            raise ValueError('SafeDodge agents may only use sphere, capsule and box geoms.')
        self._part_link = jp.array(mj_model.geom_bodyid[part_ids] - 1)
        self._part_pos = jp.array(mj_model.geom_pos[part_ids])
        # Each hitbox is a box grown by a radius (sphere: point, capsule: segment, box: radius 0)
        size = mj_model.geom_size[part_ids]
        half_size = np.zeros_like(size)
        half_size[part_types == capsule, 2] = size[part_types == capsule, 1]
        half_size[part_types == box] = size[part_types == box]
        self._part_half_size = jp.array(half_size)
        self._part_radius = jp.array(np.where(part_types == box, 0.0, size[:, 0]))
        self._part_axes = jax.vmap(jax.vmap(brax_math.rotate, in_axes=(0, None)), in_axes=(None, 0))(
            jp.eye(3), jp.array(mj_model.geom_quat[part_ids]))

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics with the plane constraint."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Store previous state for velocity calculation
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        new_info = dict(state.info)
        if self._plane_repeat:
            pipeline_state, new_info["plane_rng"] = self._next_pass(pipeline_state, state.info["plane_rng"])

        # Healthy reward
        min_z, _ = self._healthy_z_range
        is_healthy = self._is_healthy(pipeline_state)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        # Height reward
        standing = (pipeline_state.x.pos[0, 2] - min_z) / (self.default_spawn_height - min_z)
        height_reward = self._height_reward_weight * jp.clip(standing, 0.0, 1.0)

        # Upright reward: how vertical the torso's z axis is
        torso_up = brax_math.rotate(jp.array([0.0, 0.0, 1.0]), pipeline_state.x.rot[0])[2]
        upright_reward = self._upright_reward_weight * jp.clip(torso_up, 0.0, 1.0)

        # Stillness cost: horizontal centre-of-mass speed
        _, velocity = self._com_velocity(pipeline_state0, pipeline_state)
        still_cost = self._still_cost_weight * jp.linalg.norm(velocity[:2])

        # Control cost
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state, action)

        # Reward structure
        reward = healthy_reward + height_reward + upright_reward - still_cost - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Update metrics
        metrics = self._metrics(pipeline_state0, pipeline_state, {
            'reward_alive': healthy_reward,
            'reward_height': height_reward,
            'reward_upright': upright_reward,
            'reward_still': -still_cost,
            'reward_quadctrl': -ctrl_cost,
        })
        state.metrics.update(metrics)

        # Update info dictionary with cost
        new_info.update({k: metrics[k] for k in self._INFO_KEYS})
        new_info["step_count"] = state.info.get("step_count", 0) + 1

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
        )

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )

        # Set spawn position and rotation from subclass defaults
        qpos = qpos.at[2].set(self.default_spawn_height)
        w, x, y, z = self.default_spawn_rotation
        qpos = qpos.at[3].set(w)
        qpos = qpos.at[4].set(x)
        qpos = qpos.at[5].set(y)
        qpos = qpos.at[6].set(z)

        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        # Random plane for this episode
        qpos, qvel = self._set_plane(qpos, qvel, rng3)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        rewards = {k: zero for k in ('reward_alive', 'reward_height', 'reward_upright', 'reward_still',
                                     'reward_quadctrl')}
        metrics = jax.tree_util.tree_map(
            jp.zeros_like, self._metrics(pipeline_state, pipeline_state, rewards))
        info = {k: metrics[k] for k in self._INFO_KEYS}
        info["step_count"] = 0
        info["plane_rng"] = rng4

        return State(pipeline_state, obs, reward, done, metrics, info)

    def _set_plane(self, qpos: jax.Array, qvel: jax.Array, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Puts the plane at the start of a new pass: random tilt, height, position and slide."""
        rng_tilt, rng_slide, rng_pos = jax.random.split(rng, 3)
        qpos = qpos.at[self._plane_tilt_q].set(self._sample_plane_tilt(rng_tilt))
        qvel = qvel.at[self._plane_tilt_qd].set(0.0)
        slide_offset, slide_velocity = self._sample_plane_slide(rng_slide)
        qpos = qpos.at[self._plane_pos_q].set(self._sample_plane_position(rng_pos) + jp.append(slide_offset, 0.0))
        qvel = qvel.at[self._plane_pos_qd].set(jp.append(slide_velocity, 0.0))
        return qpos, qvel

    def _next_pass(self, pipeline_state: base.State, rng: jax.Array) -> Tuple[base.State, jax.Array]:
        """Starts a new pass once the sliding plane has gone past plane_point."""
        rng, pass_rng = jax.random.split(rng)
        slide_q = pipeline_state.q[self._plane_pos_q[:2]]
        slide_qd = pipeline_state.qd[self._plane_pos_qd[:2]]
        direction = slide_qd / jp.maximum(jp.linalg.norm(slide_qd), 1e-6)
        passed = slide_q @ direction >= self._plane_slide_distance + self._plane_offset_range
        qpos, qvel = self._set_plane(pipeline_state.q, pipeline_state.qd, pass_rng)
        restarted = self.pipeline_init(qpos, qvel)
        pipeline_state = jax.tree_util.tree_map(
            lambda new, old: jp.where(passed, new, old), restarted, pipeline_state)
        return pipeline_state, rng

    def _is_healthy(self, pipeline_state: base.State) -> jax.Array:
        """1.0 unless the torso is out of healthy_z_range or the agent is off the platform."""
        min_z, max_z = self._healthy_z_range
        torso_z = pipeline_state.x.pos[0, 2]
        is_healthy = jp.where((torso_z < min_z) | (torso_z > max_z), 0.0, 1.0)
        return jp.where(self._part_off_platform(pipeline_state), 0.0, is_healthy)

    # A body part whose hitbox bottom is within this many metres of the floor rests on it
    _GROUND_CONTACT_MARGIN = 0.02

    def _part_off_platform(self, pipeline_state: base.State) -> jax.Array:
        """True when any body part rests on the floor past the platform's edge."""
        center, axes = self._part_frames(pipeline_state)
        bottom = center[:, 2] - self._part_reach(axes, jp.array([0.0, 0.0, 1.0]))
        on_floor = bottom < self._GROUND_CONTACT_MARGIN
        past_edge = jp.linalg.norm(center[:, :2], axis=-1) > self._platform_radius
        return jp.any(on_floor & past_edge)

    def _part_frames(self, pipeline_state: base.State) -> Tuple[jax.Array, jax.Array]:
        """Hitbox centres (N, 3) and local axes (N, 3, 3) in world coordinates."""
        link_rot = pipeline_state.x.rot[self._part_link]
        center = pipeline_state.x.pos[self._part_link] + jax.vmap(brax_math.rotate)(self._part_pos, link_rot)
        axes = jax.vmap(jax.vmap(brax_math.rotate, in_axes=(0, None)))(self._part_axes, link_rot)
        return center, axes

    def _part_reach(self, axes: jax.Array, direction: jax.Array) -> jax.Array:
        """How far each hitbox extends from its centre along a unit direction."""
        return jp.sum(self._part_half_size * jp.abs(axes @ direction), axis=-1) + self._part_radius

    def _sample_plane_tilt(self, rng: jax.Array) -> jax.Array:
        """Random plane tilt [about x, about y] in radians."""
        r = self._plane_tilt_range
        return jax.random.uniform(rng, (2,), minval=-r, maxval=r)

    def _sample_plane_position(self, rng: jax.Array) -> jax.Array:
        """Random plane offset [x, y, z] from plane_point."""
        rng_r, rng_a, rng_h = jax.random.split(rng, 3)
        radius = self._plane_offset_range * jp.sqrt(jax.random.uniform(rng_r, ()))
        angle = jax.random.uniform(rng_a, (), minval=0.0, maxval=2 * jp.pi)
        low, high = self._plane_height_range
        height = jax.random.uniform(rng_h, (), minval=low, maxval=high)
        return jp.array([radius * jp.cos(angle), radius * jp.sin(angle), height - self._plane_base_height])

    def _sample_plane_slide(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Random plane start offset [x, y] and velocity [x, y]."""
        angle = jax.random.uniform(rng, (), minval=0.0, maxval=2 * jp.pi)
        direction = jp.array([jp.cos(angle), jp.sin(angle)])
        return -self._plane_slide_distance * direction, self._plane_slide_speed * direction

    # Spinning "bullet time" camera
    ORBIT_CAMERA = 'orbit'
    _ORBIT_DISTANCE = 4.0
    _ORBIT_HEIGHT = 0.9
    _ORBIT_ELEVATION = 0.0
    _ORBIT_SPEED = 30.0  # degrees per second

    def render(
            self,
            trajectory: Union[List[base.State], base.State],
            height: int = 240,
            width: int = 320,
            camera: Optional[str] = None,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """Renders a trajectory; camera='orbit' circles the agent like bullet time."""
        if camera != self.ORBIT_CAMERA:
            return super().render(trajectory, height=height, width=width, camera=camera)

        mj_model = self.sys.mj_model
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        d = mujoco.MjData(mj_model)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = self._ORBIT_DISTANCE
        cam.elevation = self._ORBIT_ELEVATION
        cam.lookat[:] = [0.0, 0.0, self._ORBIT_HEIGHT]

        def get_image(i: int, state: base.State) -> np.ndarray:
            d.qpos, d.qvel = state.q, state.qd
            mujoco.mj_forward(mj_model, d)
            cam.azimuth = self._ORBIT_SPEED * i * self.dt
            renderer.update_scene(d, camera=cam)
            return renderer.render()

        if isinstance(trajectory, list):
            frames = [get_image(i, s) for i, s in enumerate(trajectory)]
        else:
            frames = get_image(0, trajectory)
        renderer.close()
        return frames

    def _plane_frame(self, pipeline_state: base.State) -> Tuple[jax.Array, jax.Array]:
        """Current plane centre and unit normal in world coordinates."""
        plane_pos = pipeline_state.x.pos[self._plane_link]
        normal = brax_math.rotate(jp.array([0.0, 0.0, 1.0]), pipeline_state.x.rot[self._plane_link])
        return plane_pos, normal

    def _part_overlaps(self, pipeline_state: base.State) -> jax.Array:
        """How deep each hitbox cuts into the square plane in metres (<= 0: gap, no contact)."""
        plane_pos, normal = self._plane_frame(pipeline_state)
        plane_rot = pipeline_state.x.rot[self._plane_link]
        side_u = brax_math.rotate(jp.array([1.0, 0.0, 0.0]), plane_rot)
        side_v = brax_math.rotate(jp.array([0.0, 1.0, 0.0]), plane_rot)
        center, axes = self._part_frames(pipeline_state)
        offset = center - plane_pos
        reach = lambda direction: self._part_reach(axes, direction)

        depth = reach(normal) - jp.abs(offset @ normal)
        # > 0: the hitbox is past the square's edge
        past_edge = jp.maximum(jp.abs(offset @ side_u) - reach(side_u),
                               jp.abs(offset @ side_v) - reach(side_v)) - self._plane_half_size
        return jp.where(past_edge > 0.0, jp.minimum(depth, -past_edge), depth)

    def _constraint_metrics(self, pipeline_state: base.State) -> dict:
        """Safety constraint: summed hitbox overlap with the hazard plane, plus falling."""
        part_overlaps = self._part_overlaps(pipeline_state)
        plane_violation = jp.sum(jp.maximum(0.0, part_overlaps))
        plane_cost = self._plane_cost_weight * plane_violation
        fall_cost = self._fall_cost * (1.0 - self._is_healthy(pipeline_state))
        return {
            'plane_distance': jp.max(part_overlaps),
            'plane_violation': plane_violation,
            'plane_parts_touching': jp.sum(part_overlaps > 0.0).astype(jp.float32),
            'plane_cost': plane_cost,
            'fall_cost': fall_cost,
            'cost': plane_cost + fall_cost,
        }

    # Metrics that are also copied into state.info each step.
    _INFO_KEYS = ('cost', 'plane_distance', 'plane_violation', 'fall_cost')

    def _metrics(
            self, pipeline_state0: base.State, pipeline_state: base.State, rewards: dict,
    ) -> dict:
        """All per-step metrics."""
        com_after, velocity = self._com_velocity(pipeline_state0, pipeline_state)
        return {
            **rewards,
            'x_position': com_after[0],
            'y_position': com_after[1],
            'distance_from_origin': jp.linalg.norm(com_after),
            'x_velocity': velocity[0],
            'y_velocity': velocity[1],
            **self._constraint_metrics(pipeline_state),
        }

    def _com_velocity(
            self, pipeline_state0: base.State, pipeline_state: base.State
    ) -> Tuple[jax.Array, jax.Array]:
        """The agent's centre of mass after the step and its velocity over the step."""
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        return com_after, (com_after - com_before) / self.dt

    def _get_obs(
            self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes the agent's body, the plane, and the platform."""
        position = pipeline_state.q[self._agent_q[2:]]
        velocity = pipeline_state.qd[self._agent_qd]

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        agent_x = pipeline_state.x.take(self._agent_links)
        agent_xd = pipeline_state.xd.take(self._agent_links)
        xd_i = (
            base.Transform.create(pos=x_i.pos - agent_x.pos)
            .vmap()
            .do(agent_xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )[self._agent_qd]

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
            self._plane_obs(pipeline_state),
            self._platform_obs(pipeline_state),
        ])

    def _plane_obs(self, pipeline_state: base.State) -> jax.Array:
        """Plane offset, normal (toward the torso), distance and velocities, relative to the torso."""
        plane_pos, normal = self._plane_frame(pipeline_state)
        offset = plane_pos - pipeline_state.x.pos[0]
        normal = jp.where(jp.dot(-offset, normal) >= 0.0, 1.0, -1.0) * normal
        plane_xd = pipeline_state.xd.take(self._plane_link)
        return jp.concatenate([
            offset, normal, jp.dot(-offset, normal)[None], plane_xd.vel, plane_xd.ang,
        ])

    def _platform_obs(self, pipeline_state: base.State) -> jax.Array:
        """Torso x, y, platform radius and torso distance to the edge."""
        torso_xy = pipeline_state.x.pos[0, :2]
        edge_distance = self._platform_radius - jp.linalg.norm(torso_xy)
        return jp.concatenate([torso_xy, jp.array([self._platform_radius, edge_distance])])

    def _com(
            self, pipeline_state: base.State
    ) -> Tuple[jax.Array, base.Inertia, jax.Array, base.Transform]:
        """Calculate the agent's center of mass (the plane is left out)."""
        inertia = jax.tree.map(lambda a: a[self._agent_links], self.sys.link.inertia)
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.take(self._agent_links).vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i


class SafeDodgeHumanoid(SafeDodge):
    """Humanoid environment with a hazard plane.

    The humanoid must stay healthy while keeping its body parts from
    intersecting the hazard plane, forcing it to learn to crouch or lean away.
    """

    @property
    def agent_xml_path(self) -> str:
        return 'envs/assets/safe/humanoid_dodge.xml'

    @property
    def default_spawn_height(self) -> float:
        return 1.29

    @property
    def default_spawn_rotation(self) -> Tuple[float, float, float, float]:
        return (1.0, 0.0, 0.0, 0.0)

    @property
    def default_healthy_z_range(self) -> Tuple[float, float]:
        return (0.6, 2.0)

    def _get_gear_for_backend(self, backend: str) -> jp.ndarray:
        return jp.array([
            350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
            350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ])
