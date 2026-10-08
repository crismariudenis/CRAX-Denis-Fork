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

"""SafeBullets environment.

The agent must stay healthy while keeping every body part's hitbox from
intersecting multiple bullet spheres. Bullets spawn in front of the humanoid
and travel toward it with constant velocity. The cost is the sum of penetration
depths of body parts into bullet spheres.

No vision - only proprioception + bullet relative positions/velocities.
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


class SafeBullets(PipelineEnv, ABC):
    """Abstract base class for bullet-dodging environments.

    The agent must stay healthy on a round platform while keeping its body-part
    hitboxes out of multiple bullet spheres. The cost is how deep they cut into
    the bullets, summed over all body parts and all bullets.

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

    def _get_bullet_pos_vel(self, pipeline_state: base.State) -> Tuple[jax.Array, jax.Array]:
        """Get bullet positions and velocities as 2D arrays (num_bullets, 3).

        Uses jp.take with explicit axis to ensure 2D output shape.
        """
        bullet_links_array = jp.array(self._bullet_links_list, dtype=jp.int32)
        # Get the full position/velocity arrays first
        all_pos = pipeline_state.x.pos
        all_vel = pipeline_state.xd.vel
        # Use jp.take with explicit axis and mode to ensure 2D output
        bullet_pos = jp.take(all_pos, jp.array(self._bullet_links_list, dtype=jp.int32), axis=0, mode='clip')
        bullet_vel = jp.take(all_vel, jp.array(self._bullet_links_list, dtype=jp.int32), axis=0, mode='clip')
        # Ensure 2D shape (num_bullets, 3) - explicitly reshape
        bullet_pos = jp.reshape(bullet_pos, (self._num_bullets, 3))
        bullet_vel = jp.reshape(bullet_vel, (self._num_bullets, 3))
        return bullet_pos, bullet_vel

    def __init__(
            self,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            height_reward_weight: float = 5.0,
            terminate_when_unhealthy=True,
            healthy_z_range: Optional[Tuple[float, float]] = None,
            reset_noise_scale=1e-2,
            num_bullets: int = 4,
            bullet_speed_range: Tuple[float, float] = (3.0, 6.0),
            bullet_radius_range: Tuple[float, float] = (0.04, 0.08),
            bullet_spawn_x_range: Tuple[float, float] = (2.0, 5.0),
            bullet_spawn_y_range: Tuple[float, float] = (-1.5, 1.5),
            bullet_spawn_z_range: Tuple[float, float] = (0.2, 1.8),
            bullet_vel_y_range: Tuple[float, float] = (-1.0, 1.0),
            bullet_vel_z_range: Tuple[float, float] = (-1.0, 1.0),
            bullet_reset_x: float = -1.0,
            bullet_cost_weight: float = 1.25,
            fall_cost: float = 25.0,
            platform_radius: float = 0.5,
            episode_length: int = 1000,
            backend: str = 'generalized',
            **kwargs,
    ):
        """Initialize the bullet-dodging environment.

        Args:
            ctrl_cost_weight: Weight for control cost penalty.
            healthy_reward: Reward for staying healthy (alive).
            height_reward_weight: Reward per step for standing fully upright. Scales
                linearly from 0 at the fall height (healthy_z_range min) to this at
                the spawn height, so ducking under bullets costs reward.
            terminate_when_unhealthy: Whether to terminate episode when unhealthy.
            healthy_z_range: (min, max) z-range for healthy state.
            reset_noise_scale: Scale of noise added to initial state.
            num_bullets: Number of bullets to spawn (max 6, limited by XML).
            bullet_speed_range: (min, max) speed in m/s for bullet x-velocity (negative).
            bullet_radius_range: (min, max) radius in metres for bullet spheres.
            bullet_spawn_x_range: (min, max) x-position for bullet spawn (front of humanoid).
            bullet_spawn_y_range: (min, max) y-position for bullet spawn.
            bullet_spawn_z_range: (min, max) z-position for bullet spawn.
            bullet_vel_y_range: (min, max) y-velocity component for bullets.
            bullet_vel_z_range: (min, max) z-velocity component for bullets.
            bullet_reset_x: x-position at which bullets reset (behind humanoid).
            bullet_cost_weight: Cost per metre of hitbox overlap with bullets,
                summed over all body parts and bullets.
            fall_cost: Cost added on every step the agent is fallen (torso out of
                healthy_z_range or off the platform).
            platform_radius: Radius in metres of the round platform the agent
                stands on, centred on the origin. The torso leaving it counts as a fall.
            episode_length: Maximum number of steps per episode.
            backend: Physics backend ('generalized', 'spring', 'positional', 'mjx').
        """
        if num_bullets < 1 or num_bullets > 6:
            raise ValueError('num_bullets must be between 1 and 6 (limited by XML).')
        if bullet_speed_range[0] <= 0 or bullet_speed_range[1] <= 0:
            raise ValueError('bullet_speed_range must be positive (speed magnitude).')
        if bullet_radius_range[0] <= 0 or bullet_radius_range[1] <= 0:
            raise ValueError('bullet_radius_range must be positive.')
        if bullet_spawn_x_range[0] <= 0:
            raise ValueError('bullet_spawn_x_range min must be positive (in front of humanoid).')
        if platform_radius <= 0.0:
            raise ValueError('platform_radius must be positive.')

        self._num_bullets = num_bullets
        self._bullet_speed_range = bullet_speed_range
        self._bullet_radius_range = bullet_radius_range
        self._bullet_spawn_x_range = bullet_spawn_x_range
        self._bullet_spawn_y_range = bullet_spawn_y_range
        self._bullet_spawn_z_range = bullet_spawn_z_range
        self._bullet_vel_y_range = bullet_vel_y_range
        self._bullet_vel_z_range = bullet_vel_z_range
        self._bullet_reset_x = bullet_reset_x
        self._bullet_cost_weight = bullet_cost_weight
        self._fall_cost = fall_cost
        self._platform_radius = platform_radius

        # Load XML and place the platform
        path = epath.resource_path('crax') / self.agent_xml_path
        xml_string = path.read_text()
        xml_string = xml_string.replace('PLATFORM_RADIUS', str(float(platform_radius)))

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
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._reset_noise_scale = reset_noise_scale
        self._healthy_z_range = (
            healthy_z_range if healthy_z_range is not None else self.default_healthy_z_range)

        # Find bullet body/mocap indices from the system
        self._find_bullet_indices()

        # Agent's own slice of the physics state: everything but the bullets
        mj_model = self.sys.mj_model
        body_id = lambda name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        bullet_bodies = [body_id(f'bullet{i}') for i in range(self._num_bullets)]
        self._agent_q = jp.array(np.setdiff1d(np.arange(self.sys.q_size()), mj_model.jnt_qposadr[bullet_bodies]))
        self._agent_qd = jp.array(np.setdiff1d(np.arange(self.sys.qd_size()), mj_model.jnt_dofadr[bullet_bodies]))
        self._agent_links = jp.array([i for i in range(self.sys.num_links()) if i + 1 not in bullet_bodies])

        # Every body part of the agent (each geom, a sphere or capsule) counts for the bullet cost
        part_ids = np.array([g for g in range(mj_model.ngeom)
                             if mj_model.geom_bodyid[g] not in (0, *bullet_bodies)])
        part_types = mj_model.geom_type[part_ids]
        sphere, capsule = int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        if not np.all((part_types == sphere) | (part_types == capsule)):
            raise ValueError('SafeBullets agents may only use sphere and capsule geoms.')
        self._part_link = jp.array(mj_model.geom_bodyid[part_ids] - 1)
        self._part_pos = jp.array(mj_model.geom_pos[part_ids])
        # A capsule's axis is its geom's local z; a sphere gets half-length 0
        self._part_axis = jax.vmap(brax_math.rotate, in_axes=(None, 0))(
            jp.array([0.0, 0.0, 1.0]), jp.array(mj_model.geom_quat[part_ids]))
        self._part_radius = jp.array(mj_model.geom_size[part_ids, 0])
        self._part_half_length = jp.array(
            np.where(part_types == capsule, mj_model.geom_size[part_ids, 1], 0.0))

    def _find_bullet_indices(self):
        """Find bullet body/joint IDs from the MuJoCo model."""
        mj_model = self.sys.mj_model
        body_id = lambda name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = lambda name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        
        # Bullet slider bodies (parents with slide joints)
        self._bullet_slider_body_ids = [body_id(f'bullet{i}_slider') for i in range(self._num_bullets)]
        
        # Bullet mocap bodies (children with mocap=true) - these are the ones we track for positions
        self._bullet_body_ids = [body_id(f'bullet{i}') for i in range(self._num_bullets)]
        
        # Bullet slide joints (x, y, z) - these are the qpos/qvel addresses for bullet positions
        self._bullet_slide_x_q = jp.array([mj_model.jnt_qposadr[joint_id(f'bullet{i}_slide_x')] for i in range(self._num_bullets)])
        self._bullet_slide_y_q = jp.array([mj_model.jnt_qposadr[joint_id(f'bullet{i}_slide_y')] for i in range(self._num_bullets)])
        self._bullet_slide_z_q = jp.array([mj_model.jnt_qposadr[joint_id(f'bullet{i}_slide_z')] for i in range(self._num_bullets)])
        self._bullet_slide_x_qd = jp.array([mj_model.jnt_dofadr[joint_id(f'bullet{i}_slide_x')] for i in range(self._num_bullets)])
        self._bullet_slide_y_qd = jp.array([mj_model.jnt_dofadr[joint_id(f'bullet{i}_slide_y')] for i in range(self._num_bullets)])
        self._bullet_slide_z_qd = jp.array([mj_model.jnt_dofadr[joint_id(f'bullet{i}_slide_z')] for i in range(self._num_bullets)])
        
        # Bullet links in Brax (link i = MuJoCo body i + 1) - use the mocap child bodies
        self._bullet_links = jp.array([bid + 1 for bid in self._bullet_body_ids])
        # Ensure it's a 1D array for proper indexing
        self._bullet_links = jp.reshape(self._bullet_links, (-1,))
        # Also store as Python list for indexing
        self._bullet_links_list = [int(link) for link in self._bullet_links]

    def _sample_bullet_initial_conditions(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Sample random spawn position, velocity, and radius for each bullet.

        Returns:
            positions: (num_bullets, 3) array of spawn positions
            velocities: (num_bullets, 3) array of velocities
            radii: (num_bullets,) array of radii
        """
        rng_pos_x, rng_pos_y, rng_pos_z, rng_vel_x, rng_vel_y, rng_vel_z, rng_rad = jax.random.split(rng, 7)

        # Spawn positions: x in front, y/z spread around humanoid
        pos_x = jax.random.uniform(rng_pos_x, (self._num_bullets,),
                                   minval=self._bullet_spawn_x_range[0],
                                   maxval=self._bullet_spawn_x_range[1])
        pos_y = jax.random.uniform(rng_pos_y, (self._num_bullets,),
                                   minval=self._bullet_spawn_y_range[0],
                                   maxval=self._bullet_spawn_y_range[1])
        pos_z = jax.random.uniform(rng_pos_z, (self._num_bullets,),
                                   minval=self._bullet_spawn_z_range[0],
                                   maxval=self._bullet_spawn_z_range[1])
        positions = jp.stack([pos_x, pos_y, pos_z], axis=1)

        # Velocities: negative x (toward humanoid) + random y/z
        speed = jax.random.uniform(rng_vel_x, (self._num_bullets,),
                                   minval=self._bullet_speed_range[0],
                                   maxval=self._bullet_speed_range[1])
        vel_x = -speed  # Negative x = toward humanoid (which faces +x)
        vel_y = jax.random.uniform(rng_vel_y, (self._num_bullets,),
                                   minval=self._bullet_vel_y_range[0],
                                   maxval=self._bullet_vel_y_range[1])
        vel_z = jax.random.uniform(rng_vel_z, (self._num_bullets,),
                                   minval=self._bullet_vel_z_range[0],
                                   maxval=self._bullet_vel_z_range[1])
        velocities = jp.stack([vel_x, vel_y, vel_z], axis=1)

        # Radii
        radii = jax.random.uniform(rng_rad, (self._num_bullets,),
                                   minval=self._bullet_radius_range[0],
                                   maxval=self._bullet_radius_range[1])

        return positions, velocities, radii

    def _set_bullet_qpos_qvel(self, qpos: jax.Array, qvel: jax.Array, positions: jax.Array, velocities: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Set bullet positions and velocities in qpos/qvel arrays."""
        for i in range(self._num_bullets):
            qpos = qpos.at[self._bullet_slide_x_q[i]].set(positions[i, 0])
            qpos = qpos.at[self._bullet_slide_y_q[i]].set(positions[i, 1])
            qpos = qpos.at[self._bullet_slide_z_q[i]].set(positions[i, 2])
            qvel = qvel.at[self._bullet_slide_x_qd[i]].set(velocities[i, 0])
            qvel = qvel.at[self._bullet_slide_y_qd[i]].set(velocities[i, 1])
            qvel = qvel.at[self._bullet_slide_z_qd[i]].set(velocities[i, 2])
        return qpos, qvel

    def _update_bullets(self, pipeline_state: base.State, step_count: jax.Array) -> base.State:
        """Update bullet positions with constant velocity, reset when past humanoid.

        Args:
            pipeline_state: Current physics state
            step_count: Current step number (for time-based updates)

        Returns:
            Updated pipeline state with new bullet positions
        """
        dt = self.dt
        # Current bullet positions from mocap - use helper method for proper JAX indexing
        bullet_pos, bullet_vel = self._get_bullet_pos_vel(pipeline_state)

        # Update positions: constant velocity
        new_pos = bullet_pos + bullet_vel * dt

        # Check which bullets need reset (passed behind humanoid)
        needs_reset = new_pos[:, 0] < self._bullet_reset_x

        # For bullets that need reset, sample new initial conditions
        # We use step_count as a seed for deterministic but varied resets
        rng = jax.random.fold_in(jax.random.PRNGKey(0), step_count)
        rng = jax.random.fold_in(rng, jp.sum(new_pos[:, 0] < self._bullet_reset_x).astype(jp.int32))
        new_positions, new_velocities, new_radii = self._sample_bullet_initial_conditions(rng)

        # Where reset is needed, use new positions/velocities; otherwise keep current
        new_pos = jp.where(needs_reset[:, None], new_positions, new_pos)
        new_vel = jp.where(needs_reset[:, None], new_velocities, bullet_vel)

        # Update the pipeline state
        # Bullet positions are in qpos at slide joint addresses
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        for i in range(self._num_bullets):
            qpos = qpos.at[self._bullet_slide_x_q[i]].set(new_pos[i, 0])
            qpos = qpos.at[self._bullet_slide_y_q[i]].set(new_pos[i, 1])
            qpos = qpos.at[self._bullet_slide_z_q[i]].set(new_pos[i, 2])
            qvel = qvel.at[self._bullet_slide_x_qd[i]].set(new_vel[i, 0])
            qvel = qvel.at[self._bullet_slide_y_qd[i]].set(new_vel[i, 1])
            qvel = qvel.at[self._bullet_slide_z_qd[i]].set(new_vel[i, 2])

        # Also update xpos/xvel in the pipeline state
        bullet_links_array = jp.array(self._bullet_links_list, dtype=jp.int32)
        x_pos = pipeline_state.x.pos.at[jp.array(self._bullet_links_list, dtype=jp.int32)].set(new_pos)
        x_vel = pipeline_state.xd.vel.at[jp.array(self._bullet_links_list, dtype=jp.int32)].set(new_vel)

        return pipeline_state.replace(q=qpos, qd=qvel, x=pipeline_state.x.replace(pos=x_pos),
                                       xd=pipeline_state.xd.replace(vel=x_vel))

    def _bullet_overlaps(self, pipeline_state: base.State) -> jax.Array:
        """Compute penetration depth of each body part into each bullet sphere.

        Returns:
            (num_parts, num_bullets) array where > 0 means penetration depth,
            <= 0 means gap (negative = distance to surface).
        """
        # Bullet positions and radii - use helper method for proper JAX indexing
        bullet_pos, _ = self._get_bullet_pos_vel(pipeline_state)
        # Bullet radii are stored in geom_size of bullet geoms
        mj_model = self.sys.mj_model
        geom_id = lambda name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        bullet_geom_ids = [geom_id(f'bullet{i}') for i in range(self._num_bullets)]
        bullet_radii = jp.array([mj_model.geom_size[gid, 0] for gid in bullet_geom_ids])
        # Ensure bullet_radii is 1D array
        bullet_radii = jp.reshape(bullet_radii, (-1,))

        # Body part positions, axes, radii, half-lengths
        rotate = jax.vmap(brax_math.rotate)
        link_rot = pipeline_state.x.rot[self._part_link]
        part_center = pipeline_state.x.pos[self._part_link] + rotate(self._part_pos, link_rot)
        part_axis = rotate(self._part_axis, link_rot)

        # For each part and each bullet, compute penetration.
        # Part is a sphere (half_length=0) or capsule (half_length>0); bullet is always a sphere.
        # Everything is already batched over parts, so plain broadcasting is used (no vmap).
        center = part_center[:, None, :]          # (num_parts, 1, 3)
        axis = part_axis[:, None, :]              # (num_parts, 1, 3)
        half_len = self._part_half_length[:, None]  # (num_parts, 1)
        radius = self._part_radius[:, None]         # (num_parts, 1)
        bpos = bullet_pos[None, :, :]             # (1, num_bullets, 3)

        # Vector from part center to bullet center
        offset = bpos - center                    # (num_parts, num_bullets, 3)
        dist_sphere = jp.linalg.norm(offset, axis=-1)

        # Capsule: closest point on the segment center +/- axis * half_length to the bullet center
        t = jp.sum(offset * axis, axis=-1)        # (num_parts, num_bullets)
        t = jp.clip(t, -half_len, half_len)
        closest = center + axis * t[..., None]
        dist_capsule = jp.linalg.norm(bpos - closest, axis=-1)

        dist = jp.where(half_len > 0, dist_capsule, dist_sphere)

        # Penetration = (part_radius + bullet_radius) - distance; > 0 means overlap
        penetrations = (radius + bullet_radii[None, :]) - dist  # (num_parts, num_bullets)

        return penetrations

    def _bullet_obs(self, pipeline_state: base.State) -> jax.Array:
        """Bullet observations: relative position and velocity from torso.

        Returns:
            (num_bullets * 6,) array: [rel_pos_x, rel_pos_y, rel_pos_z, rel_vel_x, rel_vel_y, rel_vel_z] per bullet
        """
        torso_pos = pipeline_state.x.pos[0]  # Torso is link 0
        torso_vel = pipeline_state.xd.vel[0]

        # Use helper method for proper JAX indexing
        bullet_pos, bullet_vel = self._get_bullet_pos_vel(pipeline_state)

        rel_pos = bullet_pos - torso_pos  # (num_bullets, 3)
        rel_vel = bullet_vel - torso_vel  # (num_bullets, 3)

        return jp.concatenate([rel_pos, rel_vel], axis=1).ravel()

    def _platform_obs(self, pipeline_state: base.State) -> jax.Array:
        """Where the torso is on the platform (4 values).

        [torso x, y relative to the platform centre (2), platform radius (1),
        distance from the torso to the edge (1)].
        """
        torso_xy = pipeline_state.x.pos[0, :2]
        edge_distance = self._platform_radius - jp.linalg.norm(torso_xy)
        return jp.concatenate([torso_xy, jp.array([self._platform_radius, edge_distance])])

    def _constraint_metrics(self, pipeline_state: base.State) -> dict:
        """Safety constraint: body-part hitboxes intersecting bullet spheres, plus falling.

        The bullet violation sums the overlap of every part with every bullet,
        so the cost grows with how many parts touch and how deep.
        A fall adds fall_cost once, on the step it happens.
        """
        penetrations = self._bullet_overlaps(pipeline_state)  # (num_parts, num_bullets)
        bullet_violation = jp.sum(jp.maximum(0.0, penetrations))
        bullet_cost = self._bullet_cost_weight * bullet_violation
        fall_cost = self._fall_cost * (1.0 - self._is_healthy(pipeline_state))

        # Deepest overlap of any part with any bullet; negative = gap
        max_penetration = jp.max(penetrations)
        num_touching = jp.sum(penetrations > 0.0).astype(jp.float32)

        return {
            'bullet_distance': max_penetration,
            'bullet_violation': bullet_violation,
            'bullet_parts_touching': num_touching,
            'bullet_cost': bullet_cost,
            'fall_cost': fall_cost,
            'cost': bullet_cost + fall_cost,
        }

    def _is_healthy(self, pipeline_state: base.State) -> jax.Array:
        """1.0 while standing, 0.0 once fallen: torso out of healthy_z_range or off the platform."""
        min_z, max_z = self._healthy_z_range
        torso_z = pipeline_state.x.pos[0, 2]
        is_healthy = jp.where((torso_z < min_z) | (torso_z > max_z), 0.0, 1.0)
        # Torso off the edge of the round platform counts as a fall
        torso_radius = jp.linalg.norm(pipeline_state.x.pos[0, :2])
        return jp.where(torso_radius > self._platform_radius, 0.0, is_healthy)

    # Metrics that are also copied into state.info each step.
    _INFO_KEYS = ('cost', 'bullet_distance', 'bullet_violation', 'fall_cost')

    def _metrics(
            self, pipeline_state0: base.State, pipeline_state: base.State,
            healthy_reward: jax.Array, height_reward: jax.Array, ctrl_cost: jax.Array,
    ) -> dict:
        """All per-step metrics. The single source of the metric keys for step() and reset()."""
        # Center-of-mass position and velocity (logged only, not rewarded)
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        return {
            'reward_quadctrl': -ctrl_cost,
            'reward_alive': healthy_reward,
            'reward_height': height_reward,
            'x_position': com_after[0],
            'y_position': com_after[1],
            'distance_from_origin': jp.linalg.norm(com_after),
            'x_velocity': velocity[0],
            'y_velocity': velocity[1],
            **self._constraint_metrics(pipeline_state),
        }

    def _get_obs(
            self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes the agent's body and the bullets (no vision)."""
        # Body: the agent's own joints and links only (torso x, y are in the platform block)
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

        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
            self._bullet_obs(pipeline_state),
            self._platform_obs(pipeline_state),
        ])

    def _com(
            self, pipeline_state: base.State
    ) -> Tuple[jax.Array, base.Inertia, jax.Array, base.Transform]:
        """Calculate the agent's center of mass (bullets are left out)."""
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

        # Sample initial bullet conditions
        bullet_positions, bullet_velocities, bullet_radii = self._sample_bullet_initial_conditions(rng3)

        # Set bullet positions and velocities in qpos/qvel using slide joints
        qpos, qvel = self._set_bullet_qpos_qvel(qpos, qvel, bullet_positions, bullet_velocities)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        # Same keys as step() produces, all zero
        metrics = jax.tree_util.tree_map(
            jp.zeros_like, self._metrics(pipeline_state, pipeline_state, zero, zero, zero))
        info = {k: metrics[k] for k in self._INFO_KEYS}
        info["step_count"] = 0
        info["bullet_rng"] = rng4  # for bullet resets during episode

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics with the bullet constraints."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Store previous state for velocity calculation
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Update bullets: constant velocity, reset when past humanoid
        step_count = state.info.get("step_count", 0)
        pipeline_state = self._update_bullets(pipeline_state, step_count)

        new_info = dict(state.info)

        # Healthy reward
        min_z, _ = self._healthy_z_range
        is_healthy = self._is_healthy(pipeline_state)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        # Height reward, pulls against ducking: 0 at the fall height,
        # full weight when the torso is as high as at spawn (standing upright)
        standing = (pipeline_state.x.pos[0, 2] - min_z) / (self.default_spawn_height - min_z)
        height_reward = self._height_reward_weight * jp.clip(standing, 0.0, 1.0)

        # Control cost
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state, action)

        # Reward structure
        reward = healthy_reward + height_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Update metrics
        metrics = self._metrics(
            pipeline_state0, pipeline_state, healthy_reward, height_reward, ctrl_cost)
        state.metrics.update(metrics)

        # Update info dictionary with cost
        new_info.update({k: metrics[k] for k in self._INFO_KEYS})
        new_info["step_count"] = step_count + 1

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
        )


class SafeBulletsHumanoid(SafeBullets):
    """Humanoid environment with bullet dodging.

    The humanoid must stay healthy while keeping its body parts from
    intersecting bullet spheres that fly toward it from the front.
    """

    @property
    def agent_xml_path(self) -> str:
        return 'envs/assets/safe/humanoid_bullets.xml'

    @property
    def default_spawn_height(self) -> float:
        return 1.25

    @property
    def default_spawn_rotation(self) -> Tuple[float, float, float, float]:
        # Rotate 90 degrees around y-axis to stand upright: quaternion [w, x, y, z]
        return (0.707107, 0.0, 0.707107, 0.0)

    @property
    def default_healthy_z_range(self) -> Tuple[float, float]:
        return (0.6, 2.0)

    def _get_gear_for_backend(self, backend: str) -> jp.ndarray:
        return jp.array([
            350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
            350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0
        ])