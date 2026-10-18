#include "vecenv.h"

#if defined(RL_LIB_USE_PTHREADS)
#include <pthread.h>
#elif defined(RL_LIB_USE_WIN32_THREADS)
// windows.h is what brings in CreateThread, CRITICAL_SECTION and
// CONDITION_VARIABLE. WIN32_LEAN_AND_MEAN keeps the winsock/OLE declarations out
// of the way; nothing here needs them.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// What a worker should do when it is released for a step.
typedef enum {
  VECENV_STEP_DISCRETE = 0,
  VECENV_STEP_CONTINUOUS = 1,
  VECENV_STEP_STOP = 2
} VecEnvStepMode;

// ── Auto-reset seeds ────────────────────────────────────────────────────────
//
// Each sub-environment owns its reset stream: vecenv_reset starts env i at
// seed + i, every auto-reset mixes that base with the number of resets env i has
// had, and nothing is shared between envs. One vecenv-wide PRNG would make each
// env's next seed depend on how many of its siblings reset first — harmless
// while stepping is serial, wrong the moment it is not.
static u64 vecenv_env_seed(VecEnv *v, u32 i) {
  v->reset_counts[i]++;
  Rng rng;
  rng_seed(&rng, v->reset_base[i], v->reset_counts[i]);
  const u64 hi = rng_u32(&rng);
  const u64 lo = rng_u32(&rng);
  return (hi << 32) ^ lo;
}

VecEnv *vecenv_new(Arena *arena, const Env *prototype, u32 num_envs) {
  RL_LIB_ASSERT(arena != NULL && prototype != NULL && num_envs > 0,
                "vecenv_new: bad argument");
  RL_LIB_ASSERT(prototype->clone != NULL,
                "vecenv_new: prototype env does not implement clone");

  VecEnv *v = ARENA_OBJ(arena, VecEnv);
  v->num_envs = num_envs;
  v->obs_dim = prototype->obs_dim;
  v->act_dim = prototype->act_dim;
  v->step_count = 0;

  const u64 obs_total = (u64)num_envs * v->obs_dim;
  v->envs = ARENA_ARR(arena, Env *, num_envs);
  v->obs = ARENA_ARR(arena, real, obs_total);
  v->final_obs = ARENA_ARR(arena, real, obs_total);
  v->rewards = ARENA_ARR(arena, real, num_envs);
  v->dones = ARENA_ARR(arena, b8, num_envs);
  v->terminals = ARENA_ARR(arena, b8, num_envs);
  v->truncateds = ARENA_ARR(arena, b8, num_envs);
  v->reset_base = ARENA_ARR(arena, u64, num_envs);
  v->reset_counts = ARENA_ARR(arena, u64, num_envs);  // zeroed by the arena
  v->pool = NULL;

  for (u32 i = 0; i < num_envs; i++) {
    v->envs[i] = prototype->clone(arena, prototype);
  }
  return v;
}

void vecenv_reset(VecEnv *v, u64 seed) {
  RL_LIB_ASSERT(v != NULL, "vecenv_reset: NULL vecenv");
  for (u32 i = 0; i < v->num_envs; i++) {
    v->reset_base[i] = seed + (u64)i;
    v->reset_counts[i] = 0;
    v->envs[i]->reset(v->envs[i], v->reset_base[i]);
    v->envs[i]->observe(v->envs[i], v->obs + (u64)i * v->obs_dim);
    v->rewards[i] = 0.0f;
    v->dones[i] = 0;
    v->terminals[i] = 0;
    v->truncateds[i] = 0;
  }
  v->step_count = 0;
}

void vecenv_reset_seeds(VecEnv *v, const u64 *seeds) {
  RL_LIB_ASSERT(v != NULL && seeds != NULL, "vecenv_reset_seeds: NULL argument");
  for (u32 i = 0; i < v->num_envs; i++) {
    v->reset_base[i] = seeds[i];
    v->reset_counts[i] = 0;
    v->envs[i]->reset(v->envs[i], v->reset_base[i]);
    v->envs[i]->observe(v->envs[i], v->obs + (u64)i * v->obs_dim);
  }
  v->step_count = 0;
}

// One sub-environment step plus the in-place auto-reset. Everything this touches
// belongs to env i alone, which is what lets workers run it concurrently.
static void vecenv_step_env(VecEnv *v, u32 i, const u32 *actions) {
  Env *e = v->envs[i];
  RL_LIB_ASSERT(e->step != NULL, "vecenv_step: env does not implement step");
  b32 done = false;
  b32 truncated = false;
  const real r = e->step(e, actions[i], &done, &truncated);
  v->rewards[i] = r;

  if (done || truncated) {
    // Capture the terminal observation, then auto-reset in place.
    e->observe(e, v->final_obs + (u64)i * v->obs_dim);
    e->reset(e, vecenv_env_seed(v, i));
  }
  e->observe(e, v->obs + (u64)i * v->obs_dim);

  v->dones[i] = (b8)(done || truncated);
  v->terminals[i] = (b8)done;
  v->truncateds[i] = (b8)truncated;
}

static void vecenv_step_env_continuous(VecEnv *v, u32 i, const real *actions) {
  Env *e = v->envs[i];
  RL_LIB_ASSERT(e->step_continuous != NULL,
                "vecenv_step_continuous: env does not implement "
                "step_continuous");
  b32 done = false;
  b32 truncated = false;
  const real r =
      e->step_continuous(e, actions + (u64)i * v->act_dim, &done, &truncated);
  v->rewards[i] = r;

  if (done || truncated) {
    // Capture the terminal observation, then auto-reset in place.
    e->observe(e, v->final_obs + (u64)i * v->obs_dim);
    e->reset(e, vecenv_env_seed(v, i));
  }
  e->observe(e, v->obs + (u64)i * v->obs_dim);

  v->dones[i] = (b8)(done || truncated);
  v->terminals[i] = (b8)done;
  v->truncateds[i] = (b8)truncated;
}

// Worker `tid` of `stride` handles every stride-th env starting at tid, so the
// split does not depend on how many envs there are.
static void vecenv_run_slice(VecEnv *v, u32 tid, u32 stride, VecEnvStepMode mode,
                             const u32 *disc_actions,
                             const real *cont_actions) {
  if (mode == VECENV_STEP_DISCRETE) {
    for (u32 i = tid; i < v->num_envs; i += stride) {
      vecenv_step_env(v, i, disc_actions);
    }
  } else {
    for (u32 i = tid; i < v->num_envs; i += stride) {
      vecenv_step_env_continuous(v, i, cont_actions);
    }
  }
}

// ── Threaded stepping ───────────────────────────────────────────────────────
//
// A reusable barrier made of a mutex and a condition variable. pthread_barrier_t
// is optional in POSIX and absent on macOS (a first-class target here), and a
// spinning C11 barrier would burn a core while the slowest sub-environment
// finishes. The same mutex is what publishes the action buffer to the workers.
//
// Win32 exposes the same two primitives under different names, so the pool below
// is written once against the wrappers in this block: POSIX gets the pthread
// mutex/condition/thread trio, Windows the critical-section / condition-variable
// / CreateThread family (Vista and later).

#if defined(RL_LIB_USE_PTHREADS) || defined(RL_LIB_USE_WIN32_THREADS)

#if defined(RL_LIB_USE_WIN32_THREADS)
typedef CRITICAL_SECTION RlMutex;
typedef CONDITION_VARIABLE RlCond;
typedef HANDLE RlThread;

static void rl_mutex_init(RlMutex *m) { InitializeCriticalSection(m); }
static void rl_mutex_lock(RlMutex *m) { EnterCriticalSection(m); }
static void rl_mutex_unlock(RlMutex *m) { LeaveCriticalSection(m); }
static void rl_cond_init(RlCond *c) { InitializeConditionVariable(c); }

// SleepConditionVariableCS is the pthread_cond_wait analogue: it releases the
// critical section while waiting and reacquires it before returning. These waits
// have no deadline, so the timeout is INFINITE.
static void rl_cond_wait(RlCond *c, RlMutex *m) {
  (void)SleepConditionVariableCS(c, m, INFINITE);
}
static void rl_cond_broadcast(RlCond *c) { WakeAllConditionVariable(c); }
#else
typedef pthread_mutex_t RlMutex;
typedef pthread_cond_t RlCond;
typedef pthread_t RlThread;

static void rl_mutex_init(RlMutex *m) {
  RL_LIB_ASSERT(pthread_mutex_init(m, NULL) == 0, "vecenv: mutex init failed");
}
static void rl_mutex_lock(RlMutex *m) { pthread_mutex_lock(m); }
static void rl_mutex_unlock(RlMutex *m) { pthread_mutex_unlock(m); }
static void rl_cond_init(RlCond *c) {
  RL_LIB_ASSERT(pthread_cond_init(c, NULL) == 0, "vecenv: cond init failed");
}
static void rl_cond_wait(RlCond *c, RlMutex *m) { pthread_cond_wait(c, m); }
static void rl_cond_broadcast(RlCond *c) { pthread_cond_broadcast(c); }
#endif

typedef struct {
  RlMutex mtx;
  RlCond cnd;
  u32 parties;  // participants per generation, counting the main thread
  u32 arrived;
  u32 generation;
} VecEnvBarrier;

typedef struct {
  VecEnvPool *pool;
  u32 tid;
} VecEnvWorker;

struct VecEnvPool {
  VecEnv *v;
  u32 num_threads;  // participants, including the calling thread
  RlThread *tids;   // num_threads - 1 spawned workers
  VecEnvWorker *workers;

  // Workers spin up before the pool knows how many of them actually started, so
  // they park on this gate until the barriers have their final party count.
  RlMutex gate_mtx;
  RlCond gate_cnd;
  b32 gate_open;

  VecEnvBarrier start;  // main -> workers: "here is the next step"
  VecEnvBarrier done;   // workers -> main: "slice finished"

  VecEnvStepMode mode;
  const u32 *disc_actions;
  const real *cont_actions;
};

#if defined(RL_LIB_USE_WIN32_THREADS)
static DWORD WINAPI vecenv_worker_entry(LPVOID arg);
static b32 rl_thread_start(RlThread *t, VecEnvWorker *w) {
  *t = CreateThread(NULL, 0, vecenv_worker_entry, (LPVOID)w, 0, NULL);
  return *t != NULL;
}
static void rl_thread_join(RlThread t) {
  WaitForSingleObject(t, INFINITE);
  CloseHandle(t);
}
#else
static void *vecenv_worker_entry(void *arg);
static b32 rl_thread_start(RlThread *t, VecEnvWorker *w) {
  return pthread_create(t, NULL, vecenv_worker_entry, (void *)w) == 0;
}
static void rl_thread_join(RlThread t) { pthread_join(t, NULL); }
#endif

static void vecenv_barrier_init(VecEnvBarrier *b, u32 parties) {
  b->parties = parties;
  b->arrived = 0;
  b->generation = 0;
  rl_mutex_init(&b->mtx);
  rl_cond_init(&b->cnd);
}

static void vecenv_pool_barrier(VecEnvBarrier *b) {
  rl_mutex_lock(&b->mtx);
  const u32 generation = b->generation;
  b->arrived++;
  if (b->arrived == b->parties) {
    b->arrived = 0;
    b->generation++;
    rl_cond_broadcast(&b->cnd);
  } else {
    while (b->generation == generation) {
      rl_cond_wait(&b->cnd, &b->mtx);
    }
  }
  rl_mutex_unlock(&b->mtx);
}

static void vecenv_worker_main(VecEnvWorker *w) {
  VecEnvPool *p = w->pool;

  rl_mutex_lock(&p->gate_mtx);
  while (!p->gate_open) {
    rl_cond_wait(&p->gate_cnd, &p->gate_mtx);
  }
  rl_mutex_unlock(&p->gate_mtx);

  for (;;) {
    vecenv_pool_barrier(&p->start);
    if (p->mode == VECENV_STEP_STOP) {
      return;
    }
    vecenv_run_slice(p->v, w->tid, p->num_threads, p->mode, p->disc_actions,
                     p->cont_actions);
    vecenv_pool_barrier(&p->done);
  }
}

#if defined(RL_LIB_USE_WIN32_THREADS)
static DWORD WINAPI vecenv_worker_entry(LPVOID arg) {
  vecenv_worker_main((VecEnvWorker *)arg);
  return 0;
}
#else
static void *vecenv_worker_entry(void *arg) {
  vecenv_worker_main((VecEnvWorker *)arg);
  return NULL;
}
#endif

#endif // RL_LIB_USE_PTHREADS || RL_LIB_USE_WIN32_THREADS

// Steps the whole vector: through the worker pool when one is installed, or
// straight through when serial.
static void vecenv_step_all(VecEnv *v, VecEnvStepMode mode,
                            const u32 *disc_actions,
                            const real *cont_actions) {
#if defined(RL_LIB_USE_PTHREADS) || defined(RL_LIB_USE_WIN32_THREADS)
  if (v->pool != NULL) {
    VecEnvPool *p = v->pool;
    p->mode = mode;
    p->disc_actions = disc_actions;
    p->cont_actions = cont_actions;
    // The barrier's mutex publishes mode/actions to the workers; the calling
    // thread then does its own slice instead of idling until they finish.
    vecenv_pool_barrier(&p->start);
    vecenv_run_slice(v, 0, p->num_threads, mode, disc_actions, cont_actions);
    vecenv_pool_barrier(&p->done);
    v->step_count++;
    return;
  }
#endif
  vecenv_run_slice(v, 0, 1, mode, disc_actions, cont_actions);
  v->step_count++;
}

void vecenv_step(VecEnv *v, const u32 *actions) {
  RL_LIB_ASSERT(v != NULL && actions != NULL, "vecenv_step: NULL argument");
  RL_LIB_ASSERT(!v->envs[0]->continuous,
                "vecenv_step: env is continuous (use vecenv_step_continuous)");
  vecenv_step_all(v, VECENV_STEP_DISCRETE, actions, NULL);
}

void vecenv_step_continuous(VecEnv *v, const real *actions) {
  RL_LIB_ASSERT(v != NULL && actions != NULL,
                "vecenv_step_continuous: NULL argument");
  vecenv_step_all(v, VECENV_STEP_CONTINUOUS, NULL, actions);
}

void vecenv_sample_uniform(VecEnv *v, Rng *rng, u32 *actions_out) {
  RL_LIB_ASSERT(v != NULL && rng != NULL && actions_out != NULL,
                "vecenv_sample_uniform: NULL argument");
  for (u32 i = 0; i < v->num_envs; i++) {
    actions_out[i] = rng_below(rng, v->act_dim);
  }
}

void vecenv_action_bounds(const VecEnv *v, const real **min_out,
                          const real **max_out) {
  RL_LIB_ASSERT(v != NULL && v->num_envs > 0,
                "vecenv_action_bounds: NULL or empty vecenv");
  RL_LIB_ASSERT(v->envs[0]->continuous && v->envs[0]->action_min != NULL &&
                    v->envs[0]->action_max != NULL,
                "vecenv_action_bounds: env has no continuous action bounds");
  if (min_out != NULL) {
    *min_out = v->envs[0]->action_min;
  }
  if (max_out != NULL) {
    *max_out = v->envs[0]->action_max;
  }
}

void vecenv_set_threads(Arena *arena, VecEnv *v, u32 num_threads) {
  RL_LIB_ASSERT(arena != NULL && v != NULL,
                "vecenv_set_threads: NULL argument");
  vecenv_shutdown(v);
  if (num_threads <= 1) {
    return;
  }
#if defined(RL_LIB_USE_PTHREADS) || defined(RL_LIB_USE_WIN32_THREADS)
  VecEnvPool *p = ARENA_OBJ(arena, VecEnvPool);
  p->v = v;
  p->mode = VECENV_STEP_STOP;
  rl_mutex_init(&p->gate_mtx);
  rl_cond_init(&p->gate_cnd);
  p->tids = ARENA_ARR(arena, RlThread, num_threads - 1);
  p->workers = ARENA_ARR(arena, VecEnvWorker, num_threads - 1);

  // Spawn first, then learn how many threads the system actually gave us: the
  // barriers have to be sized for the workers that exist, and a failed spawn
  // must not leave anyone waiting for a party that will never arrive. With none
  // of them the vecenv simply stays serial.
  u32 spawned = 0;
  for (u32 i = 0; i + 1 < num_threads; i++) {
    p->workers[i].pool = p;
    p->workers[i].tid = i + 1;
    if (!rl_thread_start(&p->tids[i], &p->workers[i])) {
      break;
    }
    spawned++;
  }
  if (spawned == 0) {
    return;
  }
  p->num_threads = spawned + 1;
  vecenv_barrier_init(&p->start, p->num_threads);
  vecenv_barrier_init(&p->done, p->num_threads);

  rl_mutex_lock(&p->gate_mtx);
  p->gate_open = true;
  rl_cond_broadcast(&p->gate_cnd);
  rl_mutex_unlock(&p->gate_mtx);

  v->pool = p;
#else
  RL_LIB_ASSERT(false, "vecenv_set_threads: this build has no thread support");
#endif
}

u32 vecenv_threads(const VecEnv *v) {
  RL_LIB_ASSERT(v != NULL, "vecenv_threads: NULL vecenv");
#if defined(RL_LIB_USE_PTHREADS) || defined(RL_LIB_USE_WIN32_THREADS)
  return v->pool != NULL ? v->pool->num_threads : 1u;
#else
  return 1u;
#endif
}

void vecenv_shutdown(VecEnv *v) {
  RL_LIB_ASSERT(v != NULL, "vecenv_shutdown: NULL vecenv");
#if defined(RL_LIB_USE_PTHREADS) || defined(RL_LIB_USE_WIN32_THREADS)
  VecEnvPool *p = v->pool;
  if (p == NULL) {
    return;
  }
  p->mode = VECENV_STEP_STOP;
  vecenv_pool_barrier(&p->start);  // release the workers so they can return
  for (u32 i = 0; i + 1 < p->num_threads; i++) {
    rl_thread_join(p->tids[i]);
  }
  v->pool = NULL;
#endif
}
