"""
scripts/pretrain_bc_nav_aurora.py —— 用几何跟踪器做“老师”，行为克隆预训练导航策略
===================================================================================
为什么要这一步？
  在 Aurora 步态模型下从零训练 PPO 很难：要走 3–6 m、绕桌角、提前刹车（刹车还要滑 20 cm）、
  最后在 6 cm 内停稳 0.8 s —— 随机探索几乎碰不到“成功”，前 75 万步成功率一直是 0。
  先让网络模仿 ForwardFirstFollower（它已经能 100% 到达、不倒着走），再用 PPO 微调，
  PPO 只需要在“好的起点”附近改进（更快、更平滑、更稳），这是机器人学习里常见的
  “模仿学习初始化 + 强化学习微调”。
  为了让学生见过“偏离老师轨迹”的状态，一部分回合执行加了噪声的动作，但标签仍是老师在
  该状态下的动作（DART 思路），否则部署时一出偏差就不会纠正。

用法：
  python scripts/pretrain_bc_nav_aurora.py --episodes 1500        # → checkpoints/nav_aurora_bc/bc.pt
  python scripts/train.py --task nav_aurora --resume checkpoints/nav_aurora_bc/bc.pt --lr 1e-4 \\
         --steps 2000000 --run-name nav_aurora
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aurora_bridge.locomotion import ForwardFirstFollower  # noqa: E402
from config import CKPT_DIR, PPOCfg  # noqa: E402
from scripts.eval_nav_aurora import cmd_to_action  # noqa: E402
from sim.nav_aurora_env import AuroraNavEnv  # noqa: E402


def collect(episodes, seed, noise_frac=0.5, noise_std=0.3):
    env = AuroraNavEnv(seed=seed)
    rng = np.random.default_rng(seed)
    O, A = [], []
    n_ok = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        teacher = ForwardFirstFollower(env.geo)
        teacher.set_goal(*env.skill.goal)
        noisy = rng.random() < noise_frac
        for t in range(env.max_steps):
            cmd, _ = teacher(env.pose, env.vel)
            a = np.clip(cmd_to_action(cmd), -1, 1)
            O.append(obs); A.append(a)
            act = np.clip(a + rng.normal(0, noise_std, 3), -1, 1) if noisy and rng.random() < 0.3 else a
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        n_ok += info["success"]
        if (ep + 1) % 100 == 0:
            print(f"  已收集 {ep + 1} 回合，{len(O)} 个样本，老师成功率 {n_ok / (ep + 1):.2f}", flush=True)
    return np.array(O, np.float32), np.array(A, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "nav_aurora_bc", "bc.pt"))
    a = ap.parse_args()
    import torch
    from rl.ppo import ActorCritic, RunningMeanStd
    torch.set_num_threads(2)
    O, A = collect(a.episodes, a.seed)
    rms = RunningMeanStd((O.shape[1],))
    rms.update(O.astype(np.float64))
    X = torch.as_tensor(rms.normalize(O).astype(np.float32))
    Y = torch.as_tensor(A)
    cfg = PPOCfg()
    ac = ActorCritic(O.shape[1], 3, cfg.hidden, init_log_std=-1.0)
    opt = torch.optim.Adam(ac.actor.parameters(), lr=1e-3)
    n = len(X)
    idx = np.arange(n)
    for ep in range(a.epochs):
        np.random.shuffle(idx)
        tot = 0.0
        for k in range(0, n, 2048):
            b = idx[k:k + 2048]
            loss = ((ac.actor(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d}  MSE {tot / n:.4f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"model": ac.state_dict(), "obs_rms": rms.state_dict(), "obs_dim": O.shape[1], "act_dim": 3,
                "hidden": cfg.hidden, "steps": 0}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
