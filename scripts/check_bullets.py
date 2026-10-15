import jax, jax.numpy as jp
from crax import envs

env = envs.get_environment("safe_bullets_humanoid", level=1)
inner = env.unwrapped
idx = inner._bullet_links_list[0]          # bullet 0
reset, step = jax.jit(env.reset), jax.jit(env.step)

state = reset(jax.random.PRNGKey(0))
act = jp.zeros(env.action_size)
prev = None
for t in range(60):
    p = state.pipeline_state.x.pos[idx]
    v = state.pipeline_state.xd.vel[idx]
    dx = None if prev is None else float(p[0] - prev[0])
    print(f"t={t:2d} x={float(p[0]):.3f} y={float(p[1]):.3f} z={float(p[2]):.3f} "
          f"vx={float(v[0]):.3f} vz={float(v[2]):.4f} dx/step={dx}")
    prev = p
    state = step(state, act)
print("dt =", inner.dt)