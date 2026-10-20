"""Bullet dodge environment.

The dodge humanoid on its platform, with bullets instead of the hazard plane:
small capsules that fly in straight lines past the torso and vanish when they
hit a body part. The cost is the number of hits, plus falls. Everything else
(the humanoid, physics, platform, falls, rewards, hitboxes and cameras) is
SafeDodge's.

Like the plane, the bullets are not part of the physics: step moves them, and
the render model shows them as mocap bodies with a fading trail.
"""

import dataclasses
import math
from typing import Optional, Tuple
from xml.etree import ElementTree

import jax
import mujoco
import numpy as np
from flax import struct
from jax import numpy as jp

from crax import base
from crax.envs.safe_dodge import SafeDodge, SafeDodgeHumanoid
from crax.generalized import base as generalized_base


@struct.dataclass
class Bullets:
    """Every bullet slot (max_bullets of them). Slots past num_bullets never fire."""
    pos: jax.Array  # (B, 3) centre
    vel: jax.Array  # (B, 3)
    active: jax.Array  # (B,) bool: in flight
    hit: jax.Array  # (B,) bool: hit a body part this step (and vanished)
    fired: jax.Array  # (B,) bool: fired this step
    wait: jax.Array  # (B,) int: steps until an idle slot fires again


@struct.dataclass
class BulletDodgeState(generalized_base.State):
    """The physics state plus the bullets, so every renderer gets them with the frames."""
    bullets: Bullets


def _quat_z_to(direction: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] that turns the z axis to a direction."""
    d = direction / max(np.linalg.norm(direction), 1e-9)
    q = np.array([1.0 + d[2], -d[1], d[0], 0.0])  # [1 + z.d, z x d], halfway rotation
    if q[0] < 1e-6:
        return np.array([0.0, 1.0, 0.0, 0.0])  # straight down
    return q / np.linalg.norm(q)


class SafeBulletDodge(SafeDodge):
    """Abstract base class for bullet-dodging environments (see the module docstring).

    Each bullet slot fires, flies straight and level until it hits a body part or
    has passed the platform, then waits a random interval and fires again. It
    appears bullet_warning_range seconds before it reaches the torso, aimed where
    the torso will be by then if it keeps its velocity (bullet_lead), so moving
    all the time doesn't dodge anything; only reacting to a bullet does. Its
    height and sideways miss are normal around the body or, with
    bullet_aim_parts, around a random body part where it is at the shot, so no
    posture held in advance (bowing, kneeling) dodges anything either.
    """

    # Where idle bullets wait in renders: below the floor
    _PARKED = (0.0, 0.0, -10.0)
    _TRAIL_SEGMENTS = 5
    # A bullet this far (m) past the platform's edge is gone
    _PASSED_MARGIN = 0.5
    # Wait of a volley slot that has fired this volley (it won't fire again until the next one)
    _VOLLEY_FIRED = 1_000_000
    # Observed time until a bullet is closest to the torso is clipped to this (s)
    _MAX_CLOSEST_IN = 2.0
    # Ternary-search steps for the closest point of a bullet's path to a hitbox ((2/3)^20: < 0.1 mm)
    _HIT_SEARCH_STEPS = 20

    def __init__(
            self,
            num_bullets: int = 3,
            max_bullets: int = 6,
            bullet_enabled: bool = True,
            bullet_radius: float = 0.02,
            bullet_length: float = 0.06,
            bullet_speed_range: Tuple[float, float] = (1.5, 3.0),
            bullet_warning_range: Tuple[float, float] = (0.5, 1.0),
            bullet_heading: float = 180.0,
            bullet_heading_range: float = 0.0,
            bullet_height_mean: float = 1.0,
            bullet_height_std: float = 0.35,
            bullet_height_range: Tuple[float, float] = (0.3, 1.7),
            bullet_aim_std: float = 0.15,
            bullet_aim_offset: float = 0.3,
            bullet_aim_parts: Optional[Tuple[str, ...]] = None,
            bullet_lead: bool = True,
            bullet_interval_range: Tuple[int, int] = (20, 60),
            bullet_volley: bool = False,
            bullet_volley_spacing: int = 0,
            bullet_hit_cost: float = 1.0,
            bullet_trail_length: float = 0.4,
            **kwargs,
    ):
        """Initialize the bullet-dodge environment.

        Args:
            num_bullets: Bullet slots in use (at most this many bullets in flight).
            max_bullets: Bullet slots in the observation. Keep it the same on every level, so a policy
                carries over between levels.
            bullet_enabled: False hides the bullets and makes hits cost nothing (they fly through the body),
                but they are still observed.
            bullet_radius: Radius of the bullet capsule (m); it is also the hitbox.
            bullet_length: Length of the capsule's straight part (m), along the flight direction.
            bullet_speed_range: (low, high) random speed in m/s.
            bullet_warning_range: (low, high) random time (s) from a bullet appearing to it reaching the
                torso: it starts speed x warning before it. Just enough to move, so bullets don't fly for long
                while nothing is near (time the policy gets standing reward for doing nothing).
            bullet_heading: Mean direction a bullet travels in, degrees from +x (the way the humanoid faces);
                180 = it comes at the humanoid's front.
            bullet_heading_range: Max random deviation from bullet_heading in degrees (0 = every bullet straight
                from the front, as in The Matrix; 180 = any direction).
            bullet_height_mean: Mean height of the line of fire (m). Heights are normal around it, so most
                bullets cross the body rather than pass it.
            bullet_height_std: Std of the height (m).
            bullet_height_range: (low, high) limits the height is clipped to (m).
            bullet_aim_std: Std (m) of the sideways distance between the line of fire and the torso centre,
                normal around 0 (the humanoid's half-width is ~0.15-0.25 m with the arms).
            bullet_aim_offset: Max sideways distance (m) it is clipped to.
            bullet_aim_parts: Hitboxes (geom names, e.g. 'head', 'right_hand') a shot aims at, one picked at
                random per shot (name one twice to aim at it more often). The line of fire then goes through
                that part where it is when the shot is fired, missed by bullet_aim_std sideways and up or down
                (clipped to bullet_aim_offset), instead of at the torso at a height from bullet_height_mean /
                std. A posture taken before the shot (bowing, kneeling) moves the aim with it, so only moving
                the part after the shot dodges it. bullet_height_range still clips the height. None = off.
            bullet_lead: Aim at where the torso will be when the bullet arrives, from its horizontal velocity
                at the shot, instead of where it is. Without it, swaying all the time makes most bullets miss.
            bullet_interval_range: (low, high) random number of steps an idle slot waits before firing.
            bullet_volley: The slots fire together, from the same direction (a volley), and the next volley
                waits until all of them are gone. bullet_interval_range is then the wait between volleys.
            bullet_volley_spacing: Steps between consecutive shots of a volley (0 = all at once), so a volley
                comes as a burst.
            bullet_hit_cost: Cost per bullet that hits.
            bullet_trail_length: Length of the fading trail behind each bullet in renders (m).
            **kwargs: SafeDodge's settings (rewards, falls, platform, physics). The plane is always off.
        """
        if not 0 <= num_bullets <= max_bullets or max_bullets < 1:
            raise ValueError('Need 0 <= num_bullets <= max_bullets and max_bullets >= 1.')
        if bullet_radius <= 0.0 or bullet_length < 0.0 or bullet_trail_length < 0.0:
            raise ValueError('bullet_radius must be positive; bullet_length and bullet_trail_length >= 0.')
        if not 0.0 < bullet_speed_range[0] <= bullet_speed_range[1]:
            raise ValueError('bullet_speed_range must be (low, high) with 0 < low <= high.')
        if not 0.0 <= bullet_heading_range <= 180.0:
            raise ValueError('bullet_heading_range must be in [0, 180] degrees.')
        if not 0.0 <= bullet_height_range[0] <= bullet_height_range[1]:
            raise ValueError('bullet_height_range must be (low, high) with 0 <= low <= high.')
        if not 1 <= bullet_interval_range[0] <= bullet_interval_range[1]:
            raise ValueError('bullet_interval_range must be (low, high) steps with 1 <= low <= high.')
        if not 0.0 < bullet_warning_range[0] <= bullet_warning_range[1]:
            raise ValueError('bullet_warning_range must be (low, high) seconds with 0 < low <= high.')
        if bullet_aim_offset < 0.0:
            raise ValueError('bullet_aim_offset must be >= 0.')
        if bullet_height_std < 0.0 or bullet_aim_std < 0.0:
            raise ValueError('bullet_height_std and bullet_aim_std must be >= 0.')
        if bullet_volley_spacing < 0:
            raise ValueError('bullet_volley_spacing must be >= 0.')
        self._num_bullets = int(num_bullets)
        self._max_bullets = int(max_bullets)
        self._bullet_enabled = bool(bullet_enabled)
        self._bullet_radius = float(bullet_radius)
        self._bullet_length = float(bullet_length)
        self._bullet_speed_range = (float(bullet_speed_range[0]), float(bullet_speed_range[1]))
        self._bullet_warning_range = (float(bullet_warning_range[0]), float(bullet_warning_range[1]))
        self._bullet_heading_bounds = (math.radians(bullet_heading - bullet_heading_range),
                                       math.radians(bullet_heading + bullet_heading_range))
        self._bullet_height_mean = float(bullet_height_mean)
        self._bullet_height_std = float(bullet_height_std)
        self._bullet_height_range = (float(bullet_height_range[0]), float(bullet_height_range[1]))
        self._bullet_aim_std = float(bullet_aim_std)
        self._bullet_aim_offset = float(bullet_aim_offset)
        self._bullet_lead = bool(bullet_lead)
        self._bullet_interval_range = (int(bullet_interval_range[0]), int(bullet_interval_range[1]))
        self._bullet_volley = bool(bullet_volley)
        # Extra wait of each slot in a volley: slot i fires i x spacing steps after the first
        self._volley_stagger = jp.arange(self._max_bullets, dtype=jp.int32) * int(bullet_volley_spacing)
        self._bullet_hit_cost = float(bullet_hit_cost) if bullet_enabled else 0.0
        self._bullet_trail_length = float(bullet_trail_length)
        self._slot_used = jp.arange(self._max_bullets) < self._num_bullets

        kwargs['plane_enabled'] = False  # hides the plane and makes its cost 0
        super().__init__(**kwargs)

        # Hitbox indices the shots aim at (the hitboxes exist only after SafeDodge's init)
        self._aim_part_ids = None
        if bullet_aim_parts is not None:
            unknown = sorted(set(bullet_aim_parts) - set(self._part_names))
            if len(bullet_aim_parts) == 0 or unknown:
                raise ValueError(f'bullet_aim_parts must name hitboxes from {self._part_names}; '
                                 f'unknown: {unknown}.')
            self._aim_part_ids = jp.array([self._part_names.index(name) for name in bullet_aim_parts])

        model = self._render_model
        self._bullet_mocaps = [
            int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'bullet_{i}')])
            for i in range(self._max_bullets)]

    # ------------------------------------------------------------------------
    # Hazard hooks (see SafeDodge)
    # ------------------------------------------------------------------------

    def _render_xml(self, xml_string: str) -> str:
        """Adds one mocap body per bullet slot: a brass capsule along the body's z axis (the way it flies),
        with a trail of thinner, fainter capsules behind it."""
        xml = ElementTree.fromstring(xml_string)
        worldbody = xml.find('worldbody')
        group = '2' if self._bullet_enabled else '3'  # group 3 is hidden in renders
        radius, half = self._bullet_radius, self._bullet_length / 2
        segment = self._bullet_trail_length / self._TRAIL_SEGMENTS
        for i in range(self._max_bullets):
            body = ElementTree.SubElement(worldbody, 'body', name=f'bullet_{i}', mocap='true',
                                          pos=' '.join(map(str, self._PARKED)))
            ElementTree.SubElement(body, 'geom', name=f'bullet_{i}', type='capsule', size=str(radius),
                                   fromto=f'0 0 {-half} 0 0 {half}', rgba='0.78 0.58 0.2 1',
                                   contype='0', conaffinity='0', group=group)
            for k in range(self._TRAIL_SEGMENTS if segment > 0.0 else 0):
                fade = 1.0 - k / self._TRAIL_SEGMENTS
                start = -half - k * segment
                ElementTree.SubElement(body, 'geom', type='capsule', size=str(radius * 0.8 * fade),
                                       fromto=f'0 0 {start} 0 0 {start - segment}',
                                       rgba=f'0.92 0.92 0.95 {0.35 * fade:.3f}',
                                       contype='0', conaffinity='0', group=group)
        return ElementTree.tostring(xml, encoding='unicode')

    def _init_hazard(self, physics: base.State, rng: jax.Array, stream_rng: jax.Array) -> Tuple[base.State, dict]:
        """No bullets in flight; each slot's first shot comes after a random wait, so they don't all come
        at once (a volley's slots share theirs, one shot every bullet_volley_spacing steps)."""
        n = self._max_bullets
        parked = jp.tile(jp.array(self._PARKED), (n, 1))
        wait = jax.random.randint(rng, (n,), 0, self._bullet_interval_range[1] + 1)
        if self._bullet_volley:
            wait = wait[0] + self._volley_stagger
        bullets = Bullets(pos=parked, vel=jp.zeros((n, 3)), active=jp.zeros(n, bool),
                          hit=jp.zeros(n, bool), fired=jp.zeros(n, bool), wait=wait)
        pipeline_state = BulletDodgeState(
            **{f.name: getattr(physics, f.name) for f in dataclasses.fields(physics)}, bullets=bullets)
        return pipeline_state, {"bullet_rng": stream_rng}

    def _step_hazard(
            self, pipeline_state0: BulletDodgeState, pipeline_state: BulletDodgeState, info: dict,
    ) -> Tuple[BulletDodgeState, dict]:
        """Flies the bullets, removes the ones that hit or have gone past, and fires the slots whose wait
        is over."""
        b = pipeline_state0.bullets
        rng, fire_rng, wait_rng = jax.random.split(info["bullet_rng"], 3)
        pos = jp.where(b.active[:, None], b.pos + b.vel * self.dt, b.pos)
        # Checked along the whole path from the last position, so a fast bullet can't skip a limb
        if self._bullet_enabled:
            hit = b.active & jp.any(self._bullet_gaps(pipeline_state, b.pos, pos, b.vel) < 0.0, axis=-1)
        else:
            hit = jp.zeros_like(b.active)
        # Gone past: beyond the platform's far edge, where no body part can be
        heading = b.vel[:, :2] / jp.maximum(jp.linalg.norm(b.vel[:, :2], axis=-1, keepdims=True), 1e-6)
        passed = jp.sum(pos[:, :2] * heading, axis=-1) >= self._platform_radius + self._PASSED_MARGIN
        ended = b.active & (hit | passed)
        active = b.active & ~ended

        # Waiting: an idle slot counts down from a new random wait, then fires
        low, high = self._bullet_interval_range
        new_wait = jax.random.randint(wait_rng, active.shape, low, high + 1)
        if self._bullet_volley:
            # Every slot counts down until it has fired once (then it holds _VOLLEY_FIRED); once all have
            # fired and their bullets are gone, the next volley starts, one shot every bullet_volley_spacing
            fired = b.wait >= self._VOLLEY_FIRED
            over = ~jp.any(active) & jp.all(fired | ~self._slot_used)
            wait = jp.where(over, new_wait[0] + self._volley_stagger, jp.where(fired, b.wait, b.wait - 1))
            fire = self._slot_used & (wait <= 0)
            wait = jp.where(fire, self._VOLLEY_FIRED, wait)
        else:
            idle = ~active
            wait = jp.where(ended, new_wait, jp.where(idle, b.wait - 1, b.wait))
            fire = self._slot_used & idle & (wait <= 0)

        shot_pos, shot_vel = self._sample_shots(fire_rng, pipeline_state)
        parked = jp.array(self._PARKED)
        pos = jp.where(fire[:, None], shot_pos, jp.where(active[:, None], pos, parked))
        bullets = Bullets(pos=pos, vel=jp.where(fire[:, None], shot_vel, b.vel), active=active | fire,
                          hit=hit, fired=fire, wait=wait)
        return pipeline_state.replace(bullets=bullets), {**info, "bullet_rng": rng}

    def _set_render_hazard(self, data: mujoco.MjData, pipeline_state: BulletDodgeState) -> None:
        """Puts each bullet in flight where it is, pointing the way it flies; idle ones below the floor."""
        b = pipeline_state.bullets
        active, pos, vel = np.asarray(b.active), np.asarray(b.pos), np.asarray(b.vel)
        for i, mocap in enumerate(self._bullet_mocaps):
            if active[i]:
                data.mocap_pos[mocap] = pos[i]
                data.mocap_quat[mocap] = _quat_z_to(vel[i])
            else:
                data.mocap_pos[mocap] = self._PARKED

    def _hazard_torso_distance(self, pipeline_state: BulletDodgeState) -> jax.Array:
        """Distance (m) from the torso's centre to the nearest bullet in flight (1000 if none)."""
        b = pipeline_state.bullets
        distance = jp.linalg.norm(b.pos - pipeline_state.x.pos[0], axis=-1)
        return jp.min(jp.where(b.active, distance, 1000.0))

    def _hazard_obs(self, pipeline_state: BulletDodgeState) -> jax.Array:
        """Per slot (11): bullet offset from the torso (3) and velocity (3), 1 if in flight, the time (s) until
        it is closest to the torso, and where it passes then, relative to the torso (3). All 0 when idle.

        The last two follow from the first ones, but given directly they are easier to learn from.
        Slots are sorted by that time, soonest first, so the first slot is always the most urgent
        bullet, whatever the number of bullets.
        """
        b = pipeline_state.bullets
        offset = b.pos - pipeline_state.x.pos[0]
        speed_sq = jp.maximum(jp.sum(b.vel ** 2, axis=-1), 1e-6)
        closest_in = jp.clip(-jp.sum(offset * b.vel, axis=-1) / speed_sq, 0.0, self._MAX_CLOSEST_IN)
        passes_at = offset + closest_in[:, None] * b.vel
        order = jp.argsort(jp.where(b.active, closest_in, jp.inf))
        flying = b.active[:, None]
        slots = jp.concatenate([offset, b.vel, jp.ones_like(closest_in)[:, None], closest_in[:, None], passes_at],
                               axis=-1)
        return jp.where(flying, slots, 0.0)[order].ravel()

    def _constraint_metrics(self, pipeline_state: BulletDodgeState) -> dict:
        """Safety constraint: bullets that hit this step, plus falling."""
        b = pipeline_state.bullets
        hits = jp.sum(b.hit).astype(jp.float32)
        bullet_cost = self._bullet_hit_cost * hits
        fall_cost = self._fall_cost * (1.0 - self._is_healthy(pipeline_state))
        return {
            'bullet_hits': hits,
            # Summed over an episode: the shots, so hit rate = bullet_hits / bullets_fired
            'bullets_fired': jp.sum(b.fired).astype(jp.float32),
            'bullets_flying': jp.sum(b.active).astype(jp.float32),
            'bullet_cost': bullet_cost,
            'fall_cost': fall_cost,
            'cost': bullet_cost + fall_cost,
        }

    # ------------------------------------------------------------------------
    # Bullets
    # ------------------------------------------------------------------------

    def _sample_shots(self, rng: jax.Array, pipeline_state: base.State) -> Tuple[jax.Array, jax.Array]:
        """A new shot for every slot: start position (B, 3) and velocity (B, 3)."""
        n = self._max_bullets
        torso, torso_vel = pipeline_state.x.pos[0], pipeline_state.xd.vel[0]
        rng_heading, rng_height, rng_offset, rng_speed, rng_warning = jax.random.split(rng, 5)
        rng_part = jax.random.fold_in(rng, 1)  # apart from the split, so shots without aim parts don't change
        low, high = self._bullet_heading_bounds
        angle = jax.random.uniform(rng_heading, (n,), minval=low, maxval=high)
        if self._bullet_volley:
            angle = jp.full((n,), angle[0])  # one shooter
        heading = jp.stack([jp.cos(angle), jp.sin(angle)], axis=-1)
        side = jp.stack([-heading[:, 1], heading[:, 0]], axis=-1)
        # Normal around the body (as on the old bullets branch), so most bullets would hit a still humanoid
        miss = jp.clip(self._bullet_aim_std * jax.random.normal(rng_offset, (n, 1)),
                       -self._bullet_aim_offset, self._bullet_aim_offset)
        speed = jax.random.uniform(rng_speed, (n, 1), minval=self._bullet_speed_range[0],
                                   maxval=self._bullet_speed_range[1])
        warning = jax.random.uniform(rng_warning, (n, 1), minval=self._bullet_warning_range[0],
                                     maxval=self._bullet_warning_range[1])
        if self._aim_part_ids is None:
            aim = jp.broadcast_to(torso[:2], (n, 2))
            height = self._bullet_height_mean + self._bullet_height_std * jax.random.normal(rng_height, (n, 1))
        else:
            # A random body part, where it is now: its height (missed up or down like sideways) and position
            center, _ = self._part_frames(pipeline_state)
            part = self._aim_part_ids[jax.random.randint(rng_part, (n,), 0, self._aim_part_ids.shape[0])]
            aim = center[part, :2]
            height = center[part, 2:] + jp.clip(self._bullet_aim_std * jax.random.normal(rng_height, (n, 1)),
                                                 -self._bullet_aim_offset, self._bullet_aim_offset)
        height = jp.clip(height, *self._bullet_height_range)
        # Aim point: the target, or where it will be when the bullet arrives (from the torso's velocity)
        target = aim + (warning * torso_vel[:2] if self._bullet_lead else 0.0)
        start = target + miss * side - speed * warning * heading
        return (jp.concatenate([start, height], axis=-1),
                jp.concatenate([speed * heading, jp.zeros((n, 1))], axis=-1))

    def _bullet_gaps(self, pipeline_state: base.State, prev_pos: jax.Array, pos: jax.Array,
                     vel: jax.Array) -> jax.Array:
        """(B, N) closest gap (m) between each bullet and each hitbox on the bullet's path; < 0 is a hit.

        The path is the segment the capsule swept, from its tail at prev_pos to its tip at pos, so a fast
        bullet can't skip through a thin limb between two steps. It is checked against the body in
        pipeline_state (in step: after the step).
        """
        center, axes = self._part_frames(pipeline_state)
        direction = vel / jp.maximum(jp.linalg.norm(vel, axis=-1, keepdims=True), 1e-6)
        half = self._bullet_length / 2
        start = (prev_pos - half * direction)[:, None]  # (B, 1, 3)
        path = (pos + half * direction)[:, None] - start

        def gap(t):  # (B, N) distance from the point at t along each path to each hitbox, minus both radii
            local = jp.einsum('nij,bnj->bni', axes, start + t[..., None] * path - center)
            outside = jp.maximum(jp.abs(local) - self._part_half_size, 0.0)
            return jp.linalg.norm(outside, axis=-1) - self._part_radius - self._bullet_radius

        # The distance from a point on a segment to a box is convex along it: ternary search finds the minimum
        def narrow(_, bounds):
            low, high = bounds
            third = (high - low) / 3
            left = gap(low + third) < gap(high - third)
            return jp.where(left, low, low + third), jp.where(left, high - third, high)

        shape = (pos.shape[0], center.shape[0])
        low, high = jax.lax.fori_loop(0, self._HIT_SEARCH_STEPS, narrow, (jp.zeros(shape), jp.ones(shape)))
        return gap((low + high) / 2)


class SafeBulletDodgeHumanoid(SafeBulletDodge, SafeDodgeHumanoid):
    """The dodge humanoid (model, spawn, fall height) with bullets instead of the plane."""
