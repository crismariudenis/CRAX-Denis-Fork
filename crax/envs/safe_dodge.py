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

The agent must stay healthy while keeping its body out of a hazard plane's
forbidden side. The plane is defined by a point and a normal; everything on the
side the normal points to is forbidden. A horizontal plane with normal (0, 0, 1)
is a ceiling, like the height task (safe_height.py). Moving/tilted planes will be
added on top of this.
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

    The agent must stay healthy while keeping its body on the safe side of a
    hazard plane. The constraint violation (cost) is how far the agent reaches
    past the plane.

    Subclasses must implement agent-specific properties for:
    - XML file path
    - Body configuration
    - Head geometry
    """

    @property
    @abstractmethod
    def agent_xml_path(self) -> str:
        """Return the path to the agent's XML file."""
        pass

    @property
    @abstractmethod
    def head_local_offset(self) -> jp.ndarray:
        """Return the local offset of the head from the torso body."""
        pass

    @property
    @abstractmethod
    def head_radius(self) -> float:
        """Return the radius of the head geometry for the plane distance."""
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
            height_reward_weight: float = 1.0,
            terminate_when_unhealthy=True,
            healthy_z_range: Optional[Tuple[float, float]] = None,
            reset_noise_scale=1e-2,
            plane_point: Tuple[float, float, float] = (0.0, 0.0, 1.2),
            plane_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            plane_cost_weight: float = 1.25,
            platform_radius: float = 1.5,
            episode_length: int = 1000,
            backend: str = 'generalized',
            **kwargs,
    ):
        """Initialize the hazard-plane environment.

        Args:
            ctrl_cost_weight: Weight for control cost penalty.
            healthy_reward: Reward for staying healthy (alive).
            height_reward_weight: Reward per metre of torso height. Standing tall
                pays more, which conflicts with ducking under the plane.
            terminate_when_unhealthy: Whether to terminate episode when unhealthy.
            healthy_z_range: (min, max) z-range for healthy state.
            reset_noise_scale: Scale of noise added to initial state.
            plane_point: Any point [x, y, z] on the hazard plane.
            plane_normal: Normal [x, y, z] of the hazard plane, pointing to the
                forbidden side. Normalized internally. [0, 0, 1] is a ceiling.
            plane_cost_weight: Cost per metre the head reaches past the plane.
            platform_radius: Radius in metres of the round platform the agent
                stands on, centred on the origin. The torso leaving it counts as a fall.
            episode_length: Maximum number of steps per episode.
            backend: Physics backend ('generalized', 'spring', 'positional', 'mjx').
        """
        self._plane_point = jp.array(plane_point, dtype=jp.float32)
        normal = jp.array(plane_normal, dtype=jp.float32)
        normal_norm = float(jp.linalg.norm(normal))
        if normal_norm == 0.0:
            raise ValueError('plane_normal must be a non-zero vector.')
        self._plane_normal = normal / normal_norm
        self._plane_cost_weight = plane_cost_weight
        if platform_radius <= 0.0:
            raise ValueError('platform_radius must be positive.')
        self._platform_radius = platform_radius

        # Use default healthy z range if not provided
        if healthy_z_range is None:
            healthy_z_range = self.default_healthy_z_range

        # Load XML and place the hazard plane's visual
        path = epath.resource_path('crax') / self.agent_xml_path
        xml_string = path.read_text()
        xml_string = xml_string.replace('PLANE_POS', ' '.join(str(float(v)) for v in self._plane_point))
        xml_string = xml_string.replace('PLANE_NORMAL', ' '.join(str(float(v)) for v in self._plane_normal))
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
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

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

        # Height reward: pulls against the plane constraint
        height_reward = self._height_reward_weight * pipeline_state.x.pos[0, 2]

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
        rng1, rng2 = jax.random.split(rng)

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

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        # Same keys as step() produces, all zero
        metrics = jax.tree_util.tree_map(
            jp.zeros_like, self._metrics(pipeline_state, pipeline_state, zero, zero, zero))
        info = {k: metrics[k] for k in self._INFO_KEYS}
        info["step_count"] = 0

        return State(pipeline_state, obs, reward, done, metrics, info)

    # Spinning "bullet time" camera, made at render time (not an MJCF camera)
    ORBIT_CAMERA = 'orbit'
    _ORBIT_DISTANCE = 4.0  # metres from the spawn spot
    _ORBIT_HEIGHT = 0.9  # look-at height, below the plane at every level
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

    def _get_head_center(self, pipeline_state: base.State) -> jax.Array:
        """World position [x, y, z] of the centre of the agent's head."""
        torso_pos = pipeline_state.x.pos[0]
        torso_rot = pipeline_state.x.rot[0]
        head_offset_world = brax_math.rotate(self.head_local_offset, torso_rot)
        return torso_pos + head_offset_world

    def _plane_distance(self, point: jax.Array, radius: float = 0.0) -> jax.Array:
        """Signed distance of a sphere to the hazard plane.

        > 0: the sphere reaches that far into the forbidden side.
        <= 0: fully on the safe side.
        """
        return jp.dot(point - self._plane_point, self._plane_normal) + radius

    def _constraint_metrics(self, pipeline_state: base.State) -> dict:
        """Safety constraint: how far the head reaches past the hazard plane, and its cost."""
        head_center = self._get_head_center(pipeline_state)
        plane_distance = self._plane_distance(head_center, self.head_radius)
        plane_violation = jp.maximum(0.0, plane_distance)
        plane_cost = self._plane_cost_weight * plane_violation
        return {
            'head_height': head_center[2] + self.head_radius,
            'plane_distance': plane_distance,
            'plane_violation': plane_violation,
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
        """Observes body position, velocities, and angles."""
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
        ])

    def _com(
            self, pipeline_state: base.State
    ) -> Tuple[jax.Array, base.Inertia, jax.Array, base.Transform]:
        """Calculate center of mass."""
        inertia = self.sys.link.inertia
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i


class SafeDodgeHumanoid(SafeDodge):
    """Humanoid environment with a hazard plane.

    The humanoid must stay healthy while keeping its head on the safe side of
    the hazard plane, forcing it to learn to crouch or lean away.
    """

    @property
    def agent_xml_path(self) -> str:
        return 'envs/assets/safe/humanoid_dodge.xml'

    @property
    def head_local_offset(self) -> jp.ndarray:
        return jp.array([-0.15, 0.0, 0.0])

    @property
    def head_radius(self) -> float:
        return 0.09

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
