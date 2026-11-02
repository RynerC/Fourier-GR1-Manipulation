"""把 PPO 检查点（.pt）导出成 numpy 可读的 .npz（只含 actor 和观测归一化）。
用法：python scripts/export_policy_npz.py checkpoints/nav_aurora/best.pt  → 同目录 best.npz"""
import os
import sys

import numpy as np
import torch

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
sd = ck["model"]
lin = sorted({k.split(".")[1] for k in sd if k.startswith("actor.") and k.endswith(".weight")}, key=int)
out = {"n_layers": np.array(len(lin)), "obs_mean": ck["obs_rms"]["mean"], "obs_var": ck["obs_rms"]["var"]}
for i, k in enumerate(lin):
    out[f"W{i}"] = sd[f"actor.{k}.weight"].numpy().astype(np.float64)
    out[f"b{i}"] = sd[f"actor.{k}.bias"].numpy().astype(np.float64)
dst = os.path.splitext(path)[0] + ".npz"
np.savez(dst, **out)
print("saved", dst, [out[f"W{i}"].shape for i in range(len(lin))])
