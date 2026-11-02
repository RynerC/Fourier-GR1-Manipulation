"""
rl/vec_env.py —— 并行环境：同时跑 N 个仿真，攒数据更快
========================================================
为什么要并行？PPO 是 on-policy（同策略）算法：每次更新只能用“当前策略”刚收集的数据，
用完即丢。数据需求量很大（百万~千万步），所以要多进程同时仿真。

  * SubprocVecEnv：每个环境一个子进程（真正利用多核 CPU）
  * DummyVecEnv  ：单进程顺序执行（调试用，报错信息更清晰）

自动重置：某个环境回合结束时，立刻 reset，并把“结束那一刻”的观测放进
info["final_obs"]，PPO 需要它来处理“时间截断”(truncation) 的价值自举。
"""
import multiprocessing as mp

import numpy as np


def _worker(remote, parent_remote, env_fn, seed):
    parent_remote.close()
    env = env_fn()
    obs, _ = env.reset(seed=seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, r, term, trunc, info = env.step(data)
                if term or trunc:
                    info["final_obs"] = obs
                    obs, _ = env.reset()
                remote.send((obs, r, term, trunc, info))
            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(obs)
            elif cmd == "spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, env_fns, base_seed=0):
        ctx = mp.get_context("spawn")        # spawn 在 Windows / Linux 上都安全
        self.n = len(env_fns)
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.procs = []
        for i, (wr, r, fn) in enumerate(zip(work_remotes, self.remotes, env_fns)):
            p = ctx.Process(target=_worker, args=(wr, r, fn, base_seed + i), daemon=True)
            p.start()
            wr.close()
            self.procs.append(p)
        self.remotes[0].send(("spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        res = [r.recv() for r in self.remotes]
        obs, rew, term, trunc, infos = zip(*res)
        return (np.stack(obs), np.array(rew, dtype=np.float32), np.array(term),
                np.array(trunc), list(infos))

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=2)


class DummyVecEnv:
    def __init__(self, env_fns, base_seed=0):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._seeds = [base_seed + i for i in range(self.n)]

    def reset(self):
        return np.stack([e.reset(seed=s)[0] for e, s in zip(self.envs, self._seeds)])

    def step(self, actions):
        out = []
        for e, a in zip(self.envs, actions):
            obs, r, term, trunc, info = e.step(a)
            if term or trunc:
                info["final_obs"] = obs
                obs, _ = e.reset()
            out.append((obs, r, term, trunc, info))
        obs, rew, term, trunc, infos = zip(*out)
        return (np.stack(obs), np.array(rew, dtype=np.float32), np.array(term),
                np.array(trunc), list(infos))

    def close(self):
        pass
