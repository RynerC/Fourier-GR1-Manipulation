"""
aurora_bridge/np_policy.py —— 不依赖 PyTorch 的策略推理（给 Aurora 容器用）
=========================================================================
Aurora 镜像里只有 C++ 的 libtorch，没有 Python 版 torch。导航策略只是一个 36→256→256→3
的 MLP，用 numpy 做前向就够了：先在训练机上 `python scripts/export_policy_npz.py` 把
.pt 转成 .npz，再在容器里加载这个类。
"""
import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.W = [z[f"W{i}"] for i in range(int(z["n_layers"]))]
        self.b = [z[f"b{i}"] for i in range(int(z["n_layers"]))]
        self.mean, self.var = z["obs_mean"], z["obs_var"]

    def __call__(self, obs, deterministic=True):
        x = np.clip((np.asarray(obs, np.float64) - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            # 不用 x @ W.T：那会走 OpenBLAS，多核机器上会开满线程并忙等，抢走 AuroraCore 的 CPU（见 aurora_nav_eval.py 顶部）。
            # einsum(optimize=False) 用 numpy 自己的单线程循环，这么小的网络一样快。
            x = np.einsum("ij,j->i", W, x, optimize=False) + b
            if i < len(self.W) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))     # ELU
        return np.clip(x, -1, 1)
