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
intersecting a hazard plane: a square, horizontal until tilted randomly every
episode (neither side is special). It can also slide horizontally through the
platform at a constant speed.
"""

import dataclasses
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union
from xml.etree import ElementTree

import jax
import mujoco
import numpy as np
from jax import numpy as jp
from etils import epath
from flax import struct

from crax import actuator
from crax import base
from crax import math as brax_math
from crax.envs.base import PipelineEnv, State
from crax.generalized import base as generalized_base
from crax.io import mjcf


@struct.dataclass
class Plane:
    """A square hazard plane. It is not part of the physics: SafeDodge.step moves it."""
    pos: jax.Array  # (3,) centre
    rot: jax.Array  # (4,) quaternion; the plane's normal is its z axis
    vel: jax.Array  # (3,) linear velocity (the plane never rotates during a pass)
    stop_z: jax.Array  # () height where a dropping plane stops

    def axes(self) -> jax.Array:
        """(3, 3) rows: the square's two side directions and its unit normal, in world coordinates."""
        return jax.vmap(brax_math.rotate, in_axes=(0, None))(jp.eye(3), self.rot)


@struct.dataclass
class DodgeState(generalized_base.State):
    """The physics state plus the hazard plane.

    Keeping the plane in the pipeline state means every renderer gets it with the frames.
    """
    plane: Plane


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

    def __init__(
            self,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            height_reward_weight: float = 5.0,
            height_reward_full_z: Optional[float] = None,
            height_reward_full_frac: float = 1.0,
            upright_reward_weight: float = 0.0,
            upright_free_distance: float = 0.0,
            still_cost_weight: float = 1.0,
            terminate_when_unhealthy=True,
            healthy_z_range: Optional[Tuple[float, float]] = None,
            fall_on_floor_contact: bool = False,
            reset_noise_scale=1e-2,
            plane_cost_weight: float = 1.25,
            fall_cost: float = 25.0,
            plane_tilt_range: float = 90.0,
            plane_enabled: bool = True,
            plane_slide_speed: float = 0.0,
            plane_slide_distance: float = 1.5,
            plane_slide_fraction: float = 0.0,
            plane_slide_heading: float = 180.0,
            plane_slide_heading_range: float = 180.0,
            plane_resample_interval: int = 0,
            plane_drop_speed: float = 0.0,
            plane_drop_height: float = 2.3,
            plane_offset_range: float = 0.0,
            plane_height_range: Tuple[float, float] = (1.3, 1.3),
            plane_half_size: Optional[float] = None,
            platform_radius: float = 0.75,
            episode_length: int = 1000,
            solver_iterations: Optional[int] = None,
            backend: str = 'generalized',
            **kwargs,
    ):
        """Initialize the hazard-plane environment.

        Args:
            ctrl_cost_weight: Weight for control cost penalty.
            healthy_reward: Reward for staying healthy (alive).
            height_reward_weight: Reward for standing upright, 0 at the fall height.
            height_reward_full_z: Torso height (m) above which ducking costs little height reward:
                height_reward_full_frac of it here, all of it at the spawn height. None = spawn height.
            height_reward_full_frac: Fraction of height_reward_weight earned at height_reward_full_z.
            upright_reward_weight: Reward for keeping the torso vertical.
            upright_free_distance: While the plane is within this distance (m) of the torso, the upright
                reward is paid in full whatever the torso's tilt, so leaning to dodge loses no upright
                reward. The height reward still applies, so it must stand back up. 0 = never.
            still_cost_weight: Penalty per m/s of horizontal centre-of-mass speed.
            terminate_when_unhealthy: Whether to terminate episode when unhealthy.
            healthy_z_range: (min, max) z-range for healthy state.
            fall_on_floor_contact: Also a fall: any body part other than the feet resting on the floor.
            reset_noise_scale: Scale of noise added to initial state.
            plane_cost_weight: Cost per metre of hitbox overlap with the plane.
            fall_cost: Cost added when the agent falls.
            plane_tilt_range: Max random tilt of the plane per axis, in degrees.
            plane_enabled: False hides the plane and makes its cost 0.
            plane_slide_speed: Horizontal speed of the plane in m/s (0 = still). Without
                plane_resample_interval every plane slides, and a new pass starts each time it has gone past.
            plane_slide_distance: Distance from the platform centre where the sliding plane starts.
            plane_slide_fraction: With plane_resample_interval: the fraction of new planes that slide
                through at plane_slide_speed instead of dropping (or appearing).
            plane_slide_heading: Mean direction a sliding plane travels in, degrees from +x (the way the
                humanoid faces); 180 = it comes at the humanoid's front.
            plane_slide_heading_range: Max random deviation from plane_slide_heading in degrees
                (180 = any direction).
            plane_resample_interval: Steps between new planes (0 = one per episode).
            plane_drop_speed: Speed (m/s) at which each new plane comes down from plane_drop_height
                to its sampled height, then stops (0 = it appears there). Not for sliding planes.
            plane_drop_height: Height of the plane's centre when it starts dropping.
            plane_offset_range: Max random horizontal offset of the plane from the platform centre.
            plane_height_range: (low, high) random height of the plane's centre.
            plane_half_size: Half the side of the square plane; None covers the platform.            platform_radius: Radius of the platform; a part on the floor past it is a fall.
            episode_length: Maximum number of steps per episode.
            solver_iterations: Contact/limit solver iterations ('generalized'); None keeps the XML's.
            backend: Physics backend; only 'generalized' is supported.
        """
        if backend != 'generalized':
            raise ValueError(f"safe_dodge only supports the 'generalized' backend, not {backend!r}.")
        self._plane_cost_weight = plane_cost_weight if plane_enabled else 0.0
        self._fall_cost = fall_cost
        if not 0.0 <= plane_tilt_range <= 90.0:
            raise ValueError('plane_tilt_range must be in [0, 90] degrees.')
        self._plane_tilt_range = jp.deg2rad(plane_tilt_range)
        if plane_slide_speed < 0.0 or plane_slide_distance < 0.0:
            raise ValueError('plane_slide_speed and plane_slide_distance must be >= 0.')
        self._plane_slide_speed = plane_slide_speed
        self._plane_slide_distance = plane_slide_distance if plane_slide_speed > 0.0 else 0.0
        if plane_resample_interval < 0 or plane_drop_speed < 0.0:
            raise ValueError('plane_resample_interval and plane_drop_speed must be >= 0.')
        if not 0.0 <= plane_slide_fraction <= 1.0:
            raise ValueError('plane_slide_fraction must be in [0, 1].')
        if plane_resample_interval > 0:
            if (plane_slide_speed > 0.0) != (plane_slide_fraction > 0.0):
                raise ValueError('With plane_resample_interval, plane_slide_speed and plane_slide_fraction '
                                 'go together: set both (some planes slide) or neither.')
        elif plane_slide_speed > 0.0:
            if plane_drop_speed > 0.0:
                raise ValueError('Without plane_resample_interval every plane slides; it cannot also drop.')
            plane_slide_fraction = 1.0  # each pass starts the next one
        elif plane_slide_fraction > 0.0:
            raise ValueError('plane_slide_fraction needs plane_resample_interval and plane_slide_speed.')
        self._plane_slide_fraction = float(plane_slide_fraction)
        if not 0.0 <= plane_slide_heading_range <= 180.0:
            raise ValueError('plane_slide_heading_range must be in [0, 180] degrees.')
        # Python floats: the defaults give exactly [0, 2 pi), so the planes are drawn as before these settings
        self._plane_slide_heading_bounds = (math.radians(plane_slide_heading - plane_slide_heading_range),
                                            math.radians(plane_slide_heading + plane_slide_heading_range))
        self._plane_resample_interval = int(plane_resample_interval)
        self._plane_drop_speed = plane_drop_speed
        self._plane_drop_height = plane_drop_height
        self._plane_offset_range = plane_offset_range
        if not 0.0 < plane_height_range[0] <= plane_height_range[1]:
            raise ValueError('plane_height_range must be (low, high) with 0 < low <= high.')
        self._plane_height_range = (float(plane_height_range[0]), float(plane_height_range[1]))
        if platform_radius <= 0.0:
            raise ValueError('platform_radius must be positive.')
        self._platform_radius = platform_radius

        # Use default healthy z range if not provided
        if healthy_z_range is None:
            healthy_z_range = self.default_healthy_z_range
        if healthy_z_range[0] >= self.default_spawn_height:
            raise ValueError('healthy_z_range min must be below the spawn height.')
        if height_reward_full_z is None:
            height_reward_full_z = self.default_spawn_height
        if not healthy_z_range[0] < height_reward_full_z <= self.default_spawn_height:
            raise ValueError('height_reward_full_z must be above the fall height and at most the spawn height.')
        if not 0.0 <= height_reward_full_frac <= 1.0:
            raise ValueError('height_reward_full_frac must be in [0, 1].')
        self._height_reward_full_z = float(height_reward_full_z)
        self._height_reward_full_frac = float(height_reward_full_frac)

        # Load XML and size the hazard plane and platform
        path = epath.resource_path('crax') / self.agent_xml_path
        xml_string = path.read_text()
        xml_string = xml_string.replace('PLATFORM_RADIUS', str(float(platform_radius)))
        # By default the plane reaches the platform's edge at any tilt
        if plane_half_size is None:
            plane_half_size = np.hypot(platform_radius + plane_offset_range, self._plane_height_range[1])
        self._plane_half_size = float(plane_half_size)
        xml_string = xml_string.replace('PLANE_HALF_SIZE', str(self._plane_half_size))
        # Group 3 is hidden in renders
        xml_string = xml_string.replace('PLANE_GROUP', '2' if plane_enabled else '3')

        # Renders use the full XML (the plane is a mocap body there); the physics leaves the plane out
        self._render_model = mujoco.MjModel.from_xml_string(self._render_xml(xml_string))
        self._plane_mocap = int(self._render_model.body_mocapid[
            mujoco.mj_name2id(self._render_model, mujoco.mjtObj.mjOBJ_BODY, 'hazard_plane')])
        xml = ElementTree.fromstring(xml_string)
        worldbody = xml.find('worldbody')
        worldbody.remove(worldbody.find("body[@name='hazard_plane']"))
        sys = mjcf.loads(ElementTree.tostring(xml, encoding='unicode'))
        if solver_iterations is not None:
            sys = sys.replace(solver_iterations=int(solver_iterations))

        kwargs['n_frames'] = kwargs.get('n_frames', 5)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self.episode_length = episode_length
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._height_reward_weight = height_reward_weight
        self._upright_reward_weight = upright_reward_weight
        if upright_free_distance < 0.0:
            raise ValueError('upright_free_distance must be >= 0.')
        self._upright_free_distance = float(upright_free_distance)
        self._still_cost_weight = still_cost_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._fall_on_floor_contact = fall_on_floor_contact
        self._reset_noise_scale = reset_noise_scale

        # Hitboxes: every agent geom (brax link i = MuJoCo body i + 1)
        mj_model = self.sys.mj_model
        part_ids = np.array([g for g in range(mj_model.ngeom) if mj_model.geom_bodyid[g] != 0])
        part_types = mj_model.geom_type[part_ids]
        sphere, capsule, box = (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                int(mujoco.mjtGeom.mjGEOM_BOX))
        if not np.all(np.isin(part_types, (sphere, capsule, box))):
            raise ValueError('SafeDodge agents may only use sphere, capsule and box geoms.')
        self._part_names = tuple(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in part_ids)
        self._part_is_foot = jp.array(['foot' in name for name in self._part_names])
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
        pipeline_state, new_info = self._step_hazard(
            pipeline_state0, self.pipeline_step(pipeline_state0, action), state.info)

        # Healthy reward
        min_z, _ = self._healthy_z_range
        is_healthy = self._is_healthy(pipeline_state)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        # Height reward: steep from the fall height up to full_z, gentle from full_z to the spawn height
        torso_z = pipeline_state.x.pos[0, 2]
        full_z, full_frac = self._height_reward_full_z, self._height_reward_full_frac
        low_part = jp.clip((torso_z - min_z) / (full_z - min_z), 0.0, 1.0)
        high_part = jp.clip((torso_z - full_z) / max(self.default_spawn_height - full_z, 1e-6), 0.0, 1.0)
        standing = full_frac * low_part + (1.0 - full_frac) * high_part
        height_reward = self._height_reward_weight * standing

        # Upright reward: how vertical the torso's z axis is
        torso_up = brax_math.rotate(jp.array([0.0, 0.0, 1.0]), pipeline_state.x.rot[0])[2]
        upright = jp.clip(torso_up, 0.0, 1.0)
        if self._upright_free_distance > 0.0:
            upright = jp.where(self._hazard_torso_distance(pipeline_state) < self._upright_free_distance,
                               1.0, upright)
        upright_reward = self._upright_reward_weight * upright

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

        new_info["cost"] = metrics["cost"]

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
        qpos = qpos.at[3:7].set(jp.array(self.default_spawn_rotation))

        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        physics = self.pipeline_init(qpos, qvel)
        pipeline_state, hazard_info = self._init_hazard(physics, rng3, rng4)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        rewards = {k: zero for k in ('reward_alive', 'reward_height', 'reward_upright', 'reward_still',
                                     'reward_quadctrl')}
        metrics = jax.tree_util.tree_map(
            jp.zeros_like, self._metrics(pipeline_state, pipeline_state, rewards))
        info = {"cost": metrics["cost"], **hazard_info}

        return State(pipeline_state, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------------
    # Hazard hooks: what a subclass with another hazard (e.g. SafeBulletDodge) replaces
    # ------------------------------------------------------------------------

    def _render_xml(self, xml_string: str) -> str:
        """The render model's XML: the physics XML plus render-only bodies. The plane is already in it."""
        return xml_string

    def _init_hazard(self, physics: base.State, rng: jax.Array, stream_rng: jax.Array) -> Tuple[base.State, dict]:
        """The pipeline state with the episode's first hazard, and the info entries the hazard keeps.

        rng draws the first hazard; stream_rng is kept in the info for the later ones.
        """
        pipeline_state = DodgeState(**{f.name: getattr(physics, f.name) for f in dataclasses.fields(physics)},
                                    plane=self._sample_plane(rng))
        return pipeline_state, {"plane_rng": stream_rng, "plane_timer": jp.zeros((), jp.int32)}

    def _step_hazard(
            self, pipeline_state0: DodgeState, pipeline_state: DodgeState, info: dict,
    ) -> Tuple[DodgeState, dict]:
        """Moves the hazard one step. pipeline_state is the physics after the step, still with the old hazard;
        returns it with the new hazard, and the new info."""
        plane = pipeline_state0.plane
        pos = plane.pos + plane.vel * self.dt
        # A dropping plane stops at its height
        arrived = pos[2] <= plane.stop_z
        plane = plane.replace(pos=pos.at[2].max(plane.stop_z),
                              vel=jp.where(arrived, plane.vel.at[2].set(0.0), plane.vel))
        new_info = dict(info)
        if self._plane_slide_speed > 0.0 and self._plane_resample_interval == 0:
            plane, new_info["plane_rng"] = self._next_pass(plane, info["plane_rng"])
        if self._plane_resample_interval > 0:
            plane, new_info["plane_rng"], new_info["plane_timer"] = self._next_plane(
                plane, info["plane_rng"], info["plane_timer"])
        return pipeline_state.replace(plane=plane), new_info

    def _set_render_hazard(self, data: mujoco.MjData, pipeline_state: DodgeState) -> None:
        """Moves the render model's hazard bodies (mocap) to where the hazard is."""
        data.mocap_pos[self._plane_mocap] = np.asarray(pipeline_state.plane.pos)
        data.mocap_quat[self._plane_mocap] = np.asarray(pipeline_state.plane.rot)

    def _sample_plane(self, rng: jax.Array) -> Plane:
        """The plane at the start of a new pass: random tilt, height, position and slide."""
        rng_tilt, rng_slide, rng_pos = jax.random.split(rng, 3)
        tilt_x, tilt_y = self._sample_plane_tilt(rng_tilt)
        rot = brax_math.quat_mul(brax_math.quat_rot_axis(jp.array([1.0, 0.0, 0.0]), tilt_x),
                                 brax_math.quat_rot_axis(jp.array([0.0, 1.0, 0.0]), tilt_y))
        slide_offset, slide_velocity = self._sample_plane_slide(rng_slide)
        position = self._sample_plane_position(rng_pos)
        # fold_in keeps the split above, so planes that always slide or never slide are drawn as before
        slides = jax.random.uniform(jax.random.fold_in(rng, 1)) < self._plane_slide_fraction
        pos = position + jp.append(jp.where(slides, slide_offset, 0.0), 0.0)
        vel = jp.append(jp.where(slides, slide_velocity, 0.0), 0.0)
        if self._plane_drop_speed > 0.0:
            pos = pos.at[2].set(jp.where(slides, position[2], jp.maximum(self._plane_drop_height, position[2])))
            vel = vel.at[2].set(jp.where(slides, 0.0, -self._plane_drop_speed))
        return Plane(pos=pos, rot=rot, vel=vel, stop_z=position[2])

    def _next_plane(self, plane: Plane, rng: jax.Array, timer: jax.Array) -> Tuple[Plane, jax.Array, jax.Array]:
        """A new plane every plane_resample_interval steps."""
        rng, plane_rng = jax.random.split(rng)
        timer = timer + 1
        new = timer >= self._plane_resample_interval
        new_plane = self._sample_plane(plane_rng)
        plane = jax.tree.map(lambda a, b: jp.where(new, a, b), new_plane, plane)
        return plane, rng, jp.where(new, 0, timer)

    def _next_pass(self, plane: Plane, rng: jax.Array) -> Tuple[Plane, jax.Array]:
        """A new plane once the sliding one has gone past the platform centre."""
        rng, pass_rng = jax.random.split(rng)
        direction = plane.vel[:2] / jp.maximum(jp.linalg.norm(plane.vel[:2]), 1e-6)
        passed = plane.pos[:2] @ direction >= self._plane_slide_distance + self._plane_offset_range
        new_plane = self._sample_plane(pass_rng)
        return jax.tree.map(lambda new, old: jp.where(passed, new, old), new_plane, plane), rng

    def _is_healthy(self, pipeline_state: base.State) -> jax.Array:
        """1.0 unless the torso is out of healthy_z_range, the agent is off the platform, or (with
        fall_on_floor_contact) a body part other than the feet rests on the floor."""
        min_z, max_z = self._healthy_z_range
        torso_z = pipeline_state.x.pos[0, 2]
        is_healthy = jp.where((torso_z < min_z) | (torso_z > max_z), 0.0, 1.0)
        center, on_floor = self._parts_on_floor(pipeline_state)
        past_edge = jp.linalg.norm(center[:, :2], axis=-1) > self._platform_radius
        fallen = jp.any(on_floor & past_edge)
        if self._fall_on_floor_contact:
            fallen = fallen | jp.any(on_floor & ~self._part_is_foot)
        return jp.where(fallen, 0.0, is_healthy)

    # A body part whose hitbox bottom is within this many metres of the floor rests on it
    _GROUND_CONTACT_MARGIN = 0.02

    def _parts_on_floor(self, pipeline_state: base.State) -> Tuple[jax.Array, jax.Array]:
        """Hitbox centres (N, 3) and whether each rests on the floor (N,)."""
        center, axes = self._part_frames(pipeline_state)
        bottom = center[:, 2] - self._part_reach(axes, jp.array([0.0, 0.0, 1.0]))
        return center, bottom < self._GROUND_CONTACT_MARGIN

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
        """Random plane centre [x, y, z]: within plane_offset_range of the platform centre, at a random height."""
        rng_r, rng_a, rng_h = jax.random.split(rng, 3)
        radius = self._plane_offset_range * jp.sqrt(jax.random.uniform(rng_r, ()))
        angle = jax.random.uniform(rng_a, (), minval=0.0, maxval=2 * jp.pi)
        low, high = self._plane_height_range
        height = jax.random.uniform(rng_h, (), minval=low, maxval=high)
        return jp.array([radius * jp.cos(angle), radius * jp.sin(angle), height])

    def _sample_plane_slide(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Random plane start offset [x, y] and velocity [x, y]."""
        low, high = self._plane_slide_heading_bounds
        angle = jax.random.uniform(rng, (), minval=low, maxval=high)
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
        """Renders a trajectory with the plane; camera='orbit' circles the agent like bullet time."""
        mj_model = self._render_model
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        d = mujoco.MjData(mj_model)
        orbit = camera == self.ORBIT_CAMERA
        if orbit:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = self._ORBIT_DISTANCE
            cam.elevation = self._ORBIT_ELEVATION
            cam.lookat[:] = [0.0, 0.0, self._ORBIT_HEIGHT]
        else:
            cam = camera or -1

        def get_image(i: int, state: base.State) -> np.ndarray:
            d.qpos, d.qvel = state.q, state.qd
            self._set_render_hazard(d, state)
            mujoco.mj_forward(mj_model, d)
            if orbit:
                cam.azimuth = self._ORBIT_SPEED * i * self.dt
            renderer.update_scene(d, camera=cam)
            return renderer.render()

        if isinstance(trajectory, list):
            frames = [get_image(i, s) for i, s in enumerate(trajectory)]
        else:
            frames = get_image(0, trajectory)
        renderer.close()
        return frames

    def _part_overlaps(self, pipeline_state: DodgeState) -> jax.Array:
        """How deep each hitbox cuts into the square plane in metres (<= 0: gap, no contact)."""
        plane = pipeline_state.plane
        side_u, side_v, normal = plane.axes()
        center, axes = self._part_frames(pipeline_state)
        offset = center - plane.pos
        reach = lambda direction: self._part_reach(axes, direction)

        depth = reach(normal) - jp.abs(offset @ normal)
        # > 0: the hitbox is past the square's edge
        past_edge = jp.maximum(jp.abs(offset @ side_u) - reach(side_u),
                               jp.abs(offset @ side_v) - reach(side_v)) - self._plane_half_size
        return jp.where(past_edge > 0.0, jp.minimum(depth, -past_edge), depth)

    def _hazard_torso_distance(self, pipeline_state: DodgeState) -> jax.Array:
        """Distance (m) from the torso's centre to the nearest point of the plane."""
        plane = pipeline_state.plane
        side_u, side_v, normal = plane.axes()
        offset = pipeline_state.x.pos[0] - plane.pos
        gap_u = jp.maximum(jp.abs(offset @ side_u) - self._plane_half_size, 0.0)
        gap_v = jp.maximum(jp.abs(offset @ side_v) - self._plane_half_size, 0.0)
        return jp.sqrt(gap_u ** 2 + gap_v ** 2 + (offset @ normal) ** 2)

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
        position = pipeline_state.q[2:]
        velocity = pipeline_state.qd

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
            self._hazard_obs(pipeline_state),
            self._platform_obs(pipeline_state),
        ])

    def _hazard_obs(self, pipeline_state: DodgeState) -> jax.Array:
        """Plane offset, normal (toward the torso), distance and velocity, relative to the torso."""
        plane = pipeline_state.plane
        normal = plane.axes()[2]
        offset = plane.pos - pipeline_state.x.pos[0]
        normal = jp.where(jp.dot(-offset, normal) >= 0.0, 1.0, -1.0) * normal
        return jp.concatenate([offset, normal, jp.dot(-offset, normal)[None], plane.vel])

    def _platform_obs(self, pipeline_state: base.State) -> jax.Array:
        """Torso x, y, platform radius and torso distance to the edge."""
        torso_xy = pipeline_state.x.pos[0, :2]
        edge_distance = self._platform_radius - jp.linalg.norm(torso_xy)
        return jp.concatenate([torso_xy, jp.array([self._platform_radius, edge_distance])])

    def _com(
            self, pipeline_state: base.State
    ) -> Tuple[jax.Array, base.Inertia, jax.Array, base.Transform]:
        """Calculate the agent's center of mass."""
        inertia = self.sys.link.inertia
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
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
