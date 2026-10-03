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
Moving planes will be added on top of this.
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
            terminate_when_unhealthy=True,
            healthy_z_range: Optional[Tuple[float, float]] = None,
            reset_noise_scale=1e-2,
            plane_point: Tuple[float, float, float] = (0.0, 0.0, 1.2),
            plane_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            plane_cost_weight: float = 1.25,
            plane_tilt_range: float = 90.0,
            platform_radius: float = 0.75,
            episode_length: int = 1000,
            backend: str = 'generalized',
            **kwargs,
    ):
        """Initialize the hazard-plane environment.

        Args:
            ctrl_cost_weight: Weight for control cost penalty.
            healthy_reward: Reward for staying healthy (alive).
            height_reward_weight: Reward per step for standing fully upright. Scales
                linearly from 0 at the fall height (healthy_z_range min) to this at
                the spawn height, so ducking under the plane costs reward.
            terminate_when_unhealthy: Whether to terminate episode when unhealthy.
            healthy_z_range: (min, max) z-range for healthy state.
            reset_noise_scale: Scale of noise added to initial state.
            plane_point: Any point [x, y, z] on the hazard plane.
            plane_normal: Normal [x, y, z] of the untilted hazard plane; sets its
                orientation only. Normalized internally. [0, 0, 1] is horizontal.
            plane_cost_weight: Cost per metre of hitbox overlap with the plane,
                summed over all body parts.
            plane_tilt_range: Max random tilt of the plane in degrees, per axis,
                drawn at every reset (see _sample_plane_tilt). 0 keeps it fixed,
                90 allows a vertical wall.
            platform_radius: Radius in metres of the round platform the agent
                stands on, centred on the origin. The torso leaving it counts as a fall.
            episode_length: Maximum number of steps per episode.
            backend: Physics backend ('generalized', 'spring', 'positional', 'mjx').
        """
        plane_point = np.asarray(plane_point, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        normal_norm = np.linalg.norm(plane_normal)
        if normal_norm == 0.0:
            raise ValueError('plane_normal must be a non-zero vector.')
        plane_normal = plane_normal / normal_norm
        self._plane_cost_weight = plane_cost_weight
        # Neither side is special, so tilting past 90 only repeats smaller tilts
        if not 0.0 <= plane_tilt_range <= 90.0:
            raise ValueError('plane_tilt_range must be in [0, 90] degrees.')
        self._plane_tilt_range = jp.deg2rad(plane_tilt_range)
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
        # Visual size only (the cost treats the plane as infinite): big enough that, at any
        # tilt, it reaches from its centre down to the platform's edge
        plane_half_size = np.hypot(platform_radius + np.linalg.norm(plane_point[:2]), plane_point[2])
        xml_string = xml_string.replace('PLANE_HALF_SIZE', str(plane_half_size))

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
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        # Where the plane's tilt lives in the physics state (brax link i = MuJoCo body i + 1)
        mj_model = self.sys.mj_model
        self._plane_link = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'hazard_plane') - 1
        tilt_joints = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                       for name in ('plane_tilt_x', 'plane_tilt_y')]
        self._plane_tilt_q = jp.array(mj_model.jnt_qposadr[tilt_joints])
        self._plane_tilt_qd = jp.array(mj_model.jnt_dofadr[tilt_joints])

        # The agent's own slice of the physics state: everything but the plane
        self._agent_q = jp.array(np.setdiff1d(np.arange(self.sys.q_size()), mj_model.jnt_qposadr[tilt_joints]))
        self._agent_qd = jp.array(np.setdiff1d(np.arange(self.sys.qd_size()), mj_model.jnt_dofadr[tilt_joints]))
        self._agent_links = jp.array([i for i in range(self.sys.num_links()) if i != self._plane_link])

        # Every body part of the agent (each geom, a sphere or capsule) counts for the plane cost
        plane_body = self._plane_link + 1
        part_ids = np.array([g for g in range(mj_model.ngeom)
                             if mj_model.geom_bodyid[g] not in (0, plane_body)])
        part_types = mj_model.geom_type[part_ids]
        sphere, capsule = int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        if not np.all((part_types == sphere) | (part_types == capsule)):
            raise ValueError('SafeDodge agents may only use sphere and capsule geoms.')
        self._part_link = jp.array(mj_model.geom_bodyid[part_ids] - 1)
        self._part_pos = jp.array(mj_model.geom_pos[part_ids])
        # A capsule's axis is its geom's local z; a sphere gets half-length 0
        self._part_axis = jax.vmap(brax_math.rotate, in_axes=(None, 0))(
            jp.array([0.0, 0.0, 1.0]), jp.array(mj_model.geom_quat[part_ids]))
        self._part_radius = jp.array(mj_model.geom_size[part_ids, 0])
        self._part_half_length = jp.array(
            np.where(part_types == capsule, mj_model.geom_size[part_ids, 1], 0.0))

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics with the plane constraint."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Store previous state for velocity calculation
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Healthy reward
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        # Torso off the edge of the round platform counts as a fall
        torso_radius = jp.linalg.norm(pipeline_state.x.pos[0, :2])
        is_healthy = jp.where(torso_radius > self._platform_radius, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        # Height reward, pulls against ducking under the plane: 0 at the fall height,
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
        new_info = dict(state.info)
        new_info.update({k: metrics[k] for k in self._INFO_KEYS})
        new_info["step_count"] = state.info.get("step_count", 0) + 1

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
        )

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng1, rng2, rng3 = jax.random.split(rng, 3)

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

        # New random plane tilt for this episode; the plane starts (and stays) still
        qpos = qpos.at[self._plane_tilt_q].set(self._sample_plane_tilt(rng3))
        qvel = qvel.at[self._plane_tilt_qd].set(0.0)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        # Same keys as step() produces, all zero
        metrics = jax.tree_util.tree_map(
            jp.zeros_like, self._metrics(pipeline_state, pipeline_state, zero, zero, zero))
        info = {k: metrics[k] for k in self._INFO_KEYS}
        info["step_count"] = 0

        return State(pipeline_state, obs, reward, done, metrics, info)

    def _sample_plane_tilt(self, rng: jax.Array) -> jax.Array:
        """Plane tilt [about x, about y] in radians for a new episode.

        The single place that decides how the plane is randomized; edit freely.
        """
        r = self._plane_tilt_range
        return jax.random.uniform(rng, (2,), minval=-r, maxval=r)

    # Spinning "bullet time" camera, made at render time (not an MJCF camera)
    ORBIT_CAMERA = 'orbit'
    _ORBIT_DISTANCE = 4.0  # metres from the spawn spot
    _ORBIT_HEIGHT = 0.9  # look-at height, about the humanoid's middle
    _ORBIT_ELEVATION = 0.0  # degrees; negative looks down
    _ORBIT_SPEED = 30.0  # degrees per simulated second (one lap = 12 s)

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
        # Circle a fixed point above the spawn spot. Following the torso made
        # the camera slide during falls and snap back on every reset.
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
        """Current plane centre and unit normal in world coordinates, tilt included."""
        # The plane body's z axis is the normal
        plane_pos = pipeline_state.x.pos[self._plane_link]
        normal = brax_math.rotate(jp.array([0.0, 0.0, 1.0]), pipeline_state.x.rot[self._plane_link])
        return plane_pos, normal

    def _part_overlaps(self, pipeline_state: base.State) -> jax.Array:
        """How deep each body part's hitbox (sphere or capsule) cuts into the plane, in metres.

        > 0: the hitbox intersects the plane by that much.
        <= 0: no contact; minus the gap to the plane. Neither side is special.
        """
        plane_pos, normal = self._plane_frame(pipeline_state)
        rotate = jax.vmap(brax_math.rotate)
        link_rot = pipeline_state.x.rot[self._part_link]
        center = pipeline_state.x.pos[self._part_link] + rotate(self._part_pos, link_rot)
        axis = rotate(self._part_axis, link_rot)
        # How far the hitbox extends along the normal from its centre, either way
        reach = self._part_half_length * jp.abs(axis @ normal) + self._part_radius
        return reach - jp.abs((center - plane_pos) @ normal)

    def _constraint_metrics(self, pipeline_state: base.State) -> dict:
        """Safety constraint: body-part hitboxes intersecting the hazard plane, and its cost.

        The violation sums the overlap of every part, so the cost grows with how
        many parts touch and how deep, and passing through costs for every part
        that crosses.
        """
        part_overlaps = self._part_overlaps(pipeline_state)
        plane_violation = jp.sum(jp.maximum(0.0, part_overlaps))
        plane_cost = self._plane_cost_weight * plane_violation
        return {
            # Deepest overlap of any part; negative = gap between the body and the plane
            'plane_distance': jp.max(part_overlaps),
            'plane_violation': plane_violation,
            'plane_parts_touching': jp.sum(part_overlaps > 0.0).astype(jp.float32),
            'plane_cost': plane_cost,
            'cost': plane_cost,
        }

    # Metrics that are also copied into state.info each step.
    _INFO_KEYS = ('cost', 'plane_distance', 'plane_violation')

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
        """Observes the agent's body, the plane, and the platform."""
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
        """Where the plane is and how it moves, as seen from the torso (13 values).

        [plane centre - torso (3), unit normal (3), torso distance to the plane (1),
        plane linear velocity (3), plane angular velocity (3)]. Neither side is
        special, so the normal is flipped to point from the plane toward the torso
        and the distance is >= 0. The velocities are 0 until the plane moves.
        """
        plane_pos, normal = self._plane_frame(pipeline_state)
        offset = plane_pos - pipeline_state.x.pos[0]
        normal = jp.where(jp.dot(-offset, normal) >= 0.0, 1.0, -1.0) * normal
        plane_xd = pipeline_state.xd.take(self._plane_link)
        return jp.concatenate([
            offset, normal, jp.dot(-offset, normal)[None], plane_xd.vel, plane_xd.ang,
        ])

    def _platform_obs(self, pipeline_state: base.State) -> jax.Array:
        """Where the torso is on the platform (4 values).

        [torso x, y relative to the platform centre (2), platform radius (1),
        distance from the torso to the edge (1)]. The edge distance is what
        matters most when the platform is small.
        """
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
