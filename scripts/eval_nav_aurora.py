"""
scripts/eval_nav_aurora.py —— 在“Aurora 步态模型”环境里评估导航（快，几秒钟几百回合）
=====================================================================================
  python scripts/eval_nav_aurora.py --policy checkpoints/nav_aurora/best.pt     # 新策略
  python scripts/eval_nav_aurora.py --policy follower                           # 几何基线
  python scripts/eval_nav_aurora.py --policy old                                # 旧策略（理想底盘上训练的）
指标：到达率、最终位置/朝向误差、每回合倒退距离（总路程里倒着走的比例）、横移距离、用时。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aurora_bridge.locomotion import ForwardFirstFollower  # noqa: E402
from sim.nav_aurora_env import VX_BACK, VX_FWD, VY_MAX, WZ_MAX, AuroraNavEnv  # noqa: E402


def cmd_to_action(cmd):
    vx, vy, wz = cmd
    return np.clip(np.array([vx / VX_FWD if vx >= 0 else vx / VX_BACK, vy / VY_MAX, wz / WZ_MAX]), -1, 1)


class OldPolicyAdapter:
    """旧导航策略：观测里速度按 (0.6,0.35,1.2) 归一化，动作 ×(0.6,0.35,1.2)，前后对称。"""

    def __init__(self, path):
        from rl.ppo import Policy
        self.pi = Policy(path)

    def __call__(self, obs):
        o = obs.copy()
        o[8:11] = o[8:11] * np.array([0.5, 0.3, 0.8]) / np.array([0.6, 0.35, 1.2])
        a = self.pi(o)
        return cmd_to_action(np.clip(a, -1, 1) * np.array([0.6, 0.35, 1.2]))   # 后退会被裁到 0.15


def run(policy_name, episodes, seed, randomize):
    env = AuroraNavEnv(seed=seed, randomize=randomize)
    if policy_name == "follower":
        fol = ForwardFirstFollower(env.geo)
        pi = None
    elif policy_name == "old":
        pi = OldPolicyAdapter(os.path.join(ROOT, "checkpoints", "nav", "best.pt"))
    else:
        from rl.ppo import Policy
        pi = Policy(policy_name)
    R = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        if pi is None:
            fol.set_goal(*env.skill.goal)
        path_len = 0.0
        for t in range(env.max_steps):
            if pi is None:
                cmd, _ = fol(env.pose, env.vel)
                a = cmd_to_action(cmd)
            else:
                a = pi(obs)
            p0 = env.pose[:2].copy()
            obs, r, term, trunc, info = env.step(a)
            path_len += np.linalg.norm(env.pose[:2] - p0)
            if term or trunc:
                break
        R.append((info["success"], info["dist"], info["yaw_err"], info["back_dist"], path_len, (t + 1) * 0.1,
                  info["collision"]))
    R = np.array(R, float)
    ok = R[:, 0] > 0
    # 95% 置信区间（Wilson 区间，成功率接近 100% 或回合少时也可靠）：200 回合、98% 附近约 ±2%，
    # 比较两个模型时差距小于区间就不算真的有差别 —— 用同一个 --seed、足够多的 --episodes
    n, p, z = len(ok), ok.mean(), 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    print(f"{policy_name:40s} 随机化={randomize}  到达 {p * 100:5.1f}%（95% 区间 {(c - h) * 100:.1f}–{(c + h) * 100:.1f}%，{n} 回合）  "
          f"位置误差中位 {np.median(R[:, 1]) * 100:4.1f} cm  朝向误差中位 {np.degrees(np.median(R[:, 2])):4.1f}°  "
          f"倒退 平均 {R[:, 3].mean():.2f} m / 最大 {R[:, 3].max():.2f} m（占路程 {R[:, 3].sum() / R[:, 4].sum() * 100:4.1f}%）  "
          f"用时(到达) {R[ok, 5].mean() if ok.any() else float('nan'):4.1f} s")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "checkpoints", "nav_aurora", "best.pt"))
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nominal", action="store_true", help="不做域随机化（名义模型）")
    a = ap.parse_args()
    run(a.policy, a.episodes, a.seed, not a.nominal)
