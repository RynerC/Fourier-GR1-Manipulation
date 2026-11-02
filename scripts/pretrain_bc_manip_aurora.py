"""
scripts/pretrain_bc_manip_aurora.py —— 用脚本老师 ScriptedManip 做行为克隆 + DAgger，训练 Aurora 版操作策略
=====================================================================================================
和导航不同，这里模仿学习就是最终的训练方法（部署的 checkpoints/manip_aurora_bc/bc.npz 就是本脚本的输出）：
先模仿脚本老师（ScriptedManip：降低 + 前俯 → 安全高度平移到物体上方 → 竖直下降 → 抓 → 抬起 →
目标上方 → 下降 → 松手 → 抬手，快速环境里成功率约 97%）。
一部分回合给执行的动作加噪声，但标签仍是老师在该状态下的动作（DART），学生才学得会“偏了怎么回来”。
之后再做几轮 DAgger：让学生自己开（80% 学生动作），老师给学生实际走到的每个状态打标签，合并数据重训
（老师的“先升到安全高度、到正上方再竖直下降”分阶段动作，学生只模仿老师轨迹学不全，DAgger 能补上）。
PPO 微调（下面第二条命令）试过，没有提升、到 Aurora 里更差，只作为可选实验，见 docs/rl_training_guide.md 第 8 节。
采集用多进程（每个回合独立），--workers 默认 = CPU 核数。

用法：
  python scripts/pretrain_bc_manip_aurora.py --episodes 1500 --dagger 2 --out checkpoints/my_manip_bc/bc.pt
  python scripts/train.py --task manip_aurora --resume checkpoints/my_manip_bc/bc.pt --lr 3e-5 --fix-std 0.15 \\
         --freeze-obs-norm --critic-warmup 30 --steps 1000000 --run-name my_manip_s1      # 可选
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CKPT_DIR, PPOCfg  # noqa: E402


def _collect(args):
    seed, episodes, noise_frac, noise_std = args[:4]
    student_path = args[4] if len(args) > 4 else None
    from sim.manip_aurora_env import AuroraManipEnv, ScriptedManip
    env = AuroraManipEnv(seed=seed)
    teacher = ScriptedManip(env)
    rng = np.random.default_rng(seed)
    student = None
    if student_path:       # DAgger：学生来开，老师在每个状态打标签（学生走偏到的状态也有正确答案）
        from aurora_bridge.np_policy import NumpyPolicy
        student = NumpyPolicy(student_path)
    O, A, ok, n_mode = [], [], np.zeros(3), np.zeros(3)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed * 100000 + ep)
        teacher.reset()
        noisy = rng.random() < noise_frac
        for t in range(env.max_steps):
            a = teacher()
            O.append(obs); A.append(a)
            if student is not None:
                act = student(obs) if rng.random() > 0.2 else a
            else:
                act = np.clip(a + rng.normal(0, noise_std, a.shape), -1, 1) if noisy and rng.random() < 0.3 else a
            obs, r, term, trunc, inf = env.step(act)
            if term or trunc:
                break
        ok[info["mode"]] += inf["success"]; n_mode[info["mode"]] += 1
    return np.array(O, np.float32), np.array(A, np.float32), ok, n_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "manip_aurora_bc", "bc.pt"))
    ap.add_argument("--dagger", type=int, default=2, help="DAgger 轮数（学生开、老师标注、合并重训）")
    ap.add_argument("--dagger-episodes", type=int, default=600)
    a = ap.parse_args()
    from sim import gr1t2_manip_scene
    gr1t2_manip_scene.build()
    nw = max(1, min(a.workers, a.episodes))
    per = [a.episodes // nw + (1 if i < a.episodes % nw else 0) for i in range(nw)]
    print(f"用 {nw} 个进程采集 {a.episodes} 回合 ...", flush=True)
    with mp.get_context("spawn").Pool(nw) as pool:
        parts = pool.map(_collect, [(a.seed * 1000 + i, n, 0.5, 0.3) for i, n in enumerate(per)])
    O = np.concatenate([p[0] for p in parts]); A = np.concatenate([p[1] for p in parts])
    ok = sum(p[2] for p in parts); nm = sum(p[3] for p in parts)
    print(f"  {len(O)} 个样本；老师成功率：抓放 {ok[0] / max(nm[0], 1):.2f}，抓起端持 {ok[1] / max(nm[1], 1):.2f}，"
          f"放置 {ok[2] / max(nm[2], 1):.2f}（带 DART 噪声的回合也算在内）", flush=True)
    import mujoco
    bad = ~np.all(np.abs(O) < 50, axis=1)
    if ok.sum() / max(nm.sum(), 1) < 0.85 or bad.any():
        print(f"[警告] 老师成功率偏低 / 有 {int(bad.sum())} 个样本数值异常（物体飞出）。MuJoCo {mujoco.__version__}。"
              f"先跑 python scripts/eval_manip_aurora.py --policy scripted 确认环境正常（应 ≥95%），否则学生学不好。", flush=True)
    O, A = O[~bad], A[~bad]

    ac, rms = train(O, A, a.epochs)
    save(ac, rms, O.shape[1], A.shape[1], a.out)
    for k in range(a.dagger):
        npz = export_npz(ac, rms, os.path.splitext(a.out)[0] + f"_dagger{k}.npz")
        nd = max(1, a.dagger_episodes)
        per = [nd // nw + (1 if i < nd % nw else 0) for i in range(nw)]
        print(f"DAgger 第 {k + 1}/{a.dagger} 轮：学生开 {nd} 回合，老师打标签 ...", flush=True)
        with mp.get_context("spawn").Pool(nw) as pool:
            parts = pool.map(_collect, [((a.seed + 100 * (k + 1)) * 1000 + i, n, 0.0, 0.0, npz) for i, n in enumerate(per)])
        O2 = np.concatenate([p[0] for p in parts]); A2 = np.concatenate([p[1] for p in parts])
        ok = sum(p[2] for p in parts); nm = sum(p[3] for p in parts)
        print(f"  学生（80% 学生动作）成功率：抓放 {ok[0] / max(nm[0], 1):.2f}，抓起端持 {ok[1] / max(nm[1], 1):.2f}，"
              f"放置 {ok[2] / max(nm[2], 1):.2f}；新增 {len(O2)} 个样本", flush=True)
        good = np.all(np.abs(O2) < 50, axis=1)
        O, A = np.concatenate([O, O2[good]]), np.concatenate([A, A2[good]])
        ac, rms = train(O, A, a.epochs)
        save(ac, rms, O.shape[1], A.shape[1], a.out)
        os.remove(npz)


def export_npz(ac, rms, path):
    sd = ac.state_dict()
    lin = sorted({k.split(".")[1] for k in sd if k.startswith("actor.") and k.endswith(".weight")}, key=int)
    out = {"n_layers": np.array(len(lin)), "obs_mean": rms.mean, "obs_var": rms.var}
    for i, k in enumerate(lin):
        out[f"W{i}"] = sd[f"actor.{k}.weight"].numpy().astype(np.float64)
        out[f"b{i}"] = sd[f"actor.{k}.bias"].numpy().astype(np.float64)
    np.savez(path, **out)
    return path


def save(ac, rms, obs_dim, act_dim, path):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": ac.state_dict(), "obs_rms": rms.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
                "hidden": PPOCfg().hidden, "steps": 0}, path)
    print("saved", path, flush=True)


def train(O, A, epochs):
    import torch
    from rl.ppo import ActorCritic, RunningMeanStd
    torch.set_num_threads(max(1, min(8, os.cpu_count())))
    rms = RunningMeanStd((O.shape[1],))
    rms.update(O.astype(np.float64))
    X = torch.as_tensor(rms.normalize(O).astype(np.float32))
    Y = torch.as_tensor(A)
    cfg = PPOCfg()
    ac = ActorCritic(O.shape[1], A.shape[1], cfg.hidden, init_log_std=-1.0)
    opt = torch.optim.Adam(ac.actor.parameters(), lr=1e-3)
    n = len(X)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot = 0.0
        for k in range(0, n, 2048):
            b = idx[k:k + 2048]
            loss = ((ac.actor(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  MSE {tot / n:.4f}", flush=True)
    return ac, rms


if __name__ == "__main__":
    main()
