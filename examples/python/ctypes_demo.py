#!/usr/bin/env python3
"""Drive the shared library from Python with ctypes.

`make lib` leaves the shared library in the repository root (`librl_lib.dylib`
or `librl_lib.so`), and `make install` puts it in `$PREFIX/lib`. This script
loads it and talks to the plain C ABI: it steps CartPole through the exported
`Env` vtable and multiplies two `Mat`s. No bindings, no generated glue - just
the struct layouts and the exported symbols, which is all a ctypes consumer
needs.

    python3 examples/python/ctypes_demo.py [path/to/librl_lib.so]

The library is built in single precision by default, so `real` is `c_float`
here; an `-DRL_LIB_USE_DOUBLE` build would use `c_double` instead.
"""

import ctypes
import pathlib
import sys

EPISODES = 100


def find_library() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    names = ["librl_lib.dylib", "librl_lib.so", "librl_lib.so.1", "rl_lib.dll"]
    for directory in (root, pathlib.Path("/usr/local/lib"),
                      pathlib.Path("/usr/lib")):
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    sys.exit("could not find the shared library; pass its path as an argument")


lib = ctypes.CDLL(find_library())
print(f"loaded: {lib._name}")


# ── The C ABI ────────────────────────────────────────────────────────
# Matches include/env.h (Env) and include/linalg.h (Mat). Function-pointer
# fields become CFUNCTYPE so calls go through the real vtable.
class Env(ctypes.Structure):
    pass


EnvPtr = ctypes.POINTER(Env)
ResetFn = ctypes.CFUNCTYPE(None, EnvPtr, ctypes.c_uint64)
StepFn = ctypes.CFUNCTYPE(ctypes.c_float, EnvPtr, ctypes.c_uint32,
                          ctypes.POINTER(ctypes.c_uint32),
                          ctypes.POINTER(ctypes.c_uint32))
ObserveFn = ctypes.CFUNCTYPE(None, EnvPtr, ctypes.POINTER(ctypes.c_float))

Env._fields_ = [
    ("obs_dim", ctypes.c_uint32),
    ("act_dim", ctypes.c_uint32),
    ("max_steps", ctypes.c_uint32),
    ("state", ctypes.c_void_p),
    ("continuous", ctypes.c_uint32),
    ("action_min", ctypes.POINTER(ctypes.c_float)),
    ("action_max", ctypes.POINTER(ctypes.c_float)),
    ("reset", ResetFn),
    ("step", StepFn),
    ("step_continuous", ctypes.c_void_p),
    ("observe", ObserveFn),
    ("clone", ctypes.c_void_p),
    ("destroy", ctypes.c_void_p),
]


class Mat(ctypes.Structure):
    _fields_ = [("rows", ctypes.c_uint32),
                ("cols", ctypes.c_uint32),
                ("data", ctypes.POINTER(ctypes.c_float))]


# ── Signatures ───────────────────────────────────────────────────────
lib.arena_new.restype = ctypes.c_void_p
lib.arena_new.argtypes = [ctypes.c_size_t]
lib.arena_dispose.restype = None
lib.arena_dispose.argtypes = [ctypes.c_void_p]

lib.rng_seed_g.restype = None
lib.rng_seed_g.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
lib.rng_u32_g.restype = ctypes.c_uint32

lib.env_cartpole_create.restype = EnvPtr
lib.env_cartpole_create.argtypes = [ctypes.c_void_p]

lib.mat_new.restype = ctypes.POINTER(Mat)
lib.mat_new.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
lib.mat_mm.restype = ctypes.c_uint32
lib.mat_mm.argtypes = [ctypes.POINTER(Mat), ctypes.POINTER(Mat),
                       ctypes.POINTER(Mat), ctypes.c_uint8, ctypes.c_uint8,
                       ctypes.c_uint8]


# ── CartPole through the Env vtable ──────────────────────────────────
arena = lib.arena_new(1 << 20)
env = lib.env_cartpole_create(arena)
e = env.contents
print(f"CartPole: obs_dim={e.obs_dim} act_dim={e.act_dim} "
      f"max_steps={e.max_steps}")

lib.rng_seed_g(42, 7)  # the global RNG stream, for the random actions
done = ctypes.c_uint32()
truncated = ctypes.c_uint32()

returns = []
total = 0.0
e.reset(env, 1234)
while len(returns) < EPISODES:
    action = lib.rng_u32_g() % e.act_dim
    reward = e.step(env, action, ctypes.byref(done), ctypes.byref(truncated))
    total += reward
    if done.value or truncated.value:
        returns.append(total)
        total = 0.0
        e.reset(env, 1234 + len(returns))

mean = sum(returns) / len(returns)
print(f"random policy over {EPISODES} episodes: mean return {mean:.2f} "
      f"(README baseline ~22)")

# ── A Mat multiply ───────────────────────────────────────────────────
rows_a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
rows_b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
expected = [[sum(a * b for a, b in zip(ra, cb)) for cb in zip(*rows_b)]
            for ra in rows_a]

a = lib.mat_new(arena, 2, 3)
b = lib.mat_new(arena, 3, 2)
c = lib.mat_new(arena, 2, 2)
for i, row in enumerate(rows_a):
    for j, value in enumerate(row):
        a.contents.data[i * 3 + j] = value
for i, row in enumerate(rows_b):
    for j, value in enumerate(row):
        b.contents.data[i * 2 + j] = value

lib.mat_mm(c, a, b, 1, 0, 0)  # zero_out=1, no transposes
got = [[c.contents.data[i * 2 + j] for j in range(2)] for i in range(2)]
print(f"mat_mm 2x3 @ 3x2 -> {got}")

if got != expected or not (15.0 < mean < 30.0):
    sys.exit("demo check failed")
print("ctypes demo OK")
