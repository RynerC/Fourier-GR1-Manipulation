"""
scripts/eval_manip_aurora.py —— 在快速环境里评估 Aurora PdStand 版操作策略（确定性动作）
=========================================================================================
  python scripts/eval_manip_aurora.py --policy checkpoints/manip_aurora_bc/bc.pt --episodes 300
  python scripts/eval_manip_aurora.py --policy scripted                       # 脚本老师（对照）
三种模式轮流（抓放 / 抓起端持 / 放置），输出每种的成功率、95% 区间、平均用时、手臂碰桌比例、掉落数。
比较两个模型时用同一个 --seed，区间重叠就不算真的有差别。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.manip_aurora_env import AuroraManipEnv, ScriptedManip  # noqa: E402

MODE_NAME = {0: "抓放", 1: "抓起端持", 2: "放置"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load(name, env):
    if name == "scripted":
        t = ScriptedManip(env)
        return t, t.reset
    if name.endswith(".npz"):
        from aurora_bridge.np_policy import NumpyPolicy
        return NumpyPolicy(name), None
    from rl.ppo import Policy
    return Policy(name), None


def run(name, episodes, seed, randomize=True, noise=True):
    env = AuroraManipEnv(seed=seed, randomize=randomize, obs_noise=noise)
    pi, reset_fn = load(name, env)
    R = {k: [] for k in range(3)}
    for ep in range(episodes):
        mode = ep % 3
        obs, _ = env.reset(seed=seed * 1000 + ep, options={"mode": mode})
        if reset_fn:
            reset_fn()
        hits = 0
        for t in range(env.max_steps):
            a = pi() if reset_fn else pi(obs)
            obs, r, term, trunc, info = env.step(a)
            hits += info["arm_hit"] or info["body_hit"]
            if term or trunc:
                break
        R[mode].append((info["success"], (t + 1) * 0.05, hits / (t + 1), info["dropped"], info.get("knocked", False)))
    tot = sum(len(v) for v in R.values())
    ok = sum(sum(x[0] for x in v) for v in R.values())
    lo, hi = wilson(ok, tot)
    print(f"{name}  随机化={randomize}  总成功 {ok / tot * 100:5.1f}%（95% 区间 {lo * 100:.1f}–{hi * 100:.1f}%，{tot} 回合）")
    for k, v in R.items():
        if not v:
            continue
        a = np.array(v, float)
        s = a[:, 0] > 0
        lo, hi = wilson(int(s.sum()), len(a))
        print(f"   {MODE_NAME[k]:6s} 成功 {s.mean() * 100:5.1f}%（{lo * 100:.0f}–{hi * 100:.0f}%）  "
              f"用时(成功) {a[s, 1].mean() if s.any() else float('nan'):4.1f} s  碰桌步数占比 {a[:, 2].mean() * 100:4.1f}%  "
              f"掉落 {int(a[:, 3].sum())}  碰倒/推开别的物体 {int(a[:, 4].sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "checkpoints", "manip_aurora_bc", "bc.pt"))
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--no-randomize", action="store_true")
    a = ap.parse_args()
    run(a.policy, a.episodes, a.seed, randomize=not a.no_randomize)


if __name__ == "__main__":
    main()
