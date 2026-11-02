"""把 aurora_nav_eval.py 输出的若干 JSON 画成俯视轨迹对比图（倒退的片段标红）。
用法：python scripts/plot_aurora_nav.py follower.json:跟踪器 rl.json:新PPO --out docs/img/aurora_nav_compare.png"""
import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aurora_bridge.locomotion import AuroraNavRunner  # noqa: E402
from config import SceneCfg  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="file.json:标题")
    ap.add_argument("--out", default="aurora_nav_compare.png")
    a = ap.parse_args()
    sc = SceneCfg()
    fig, axs = plt.subplots(1, len(a.runs), figsize=(6 * len(a.runs), 6.2), squeeze=False)
    for ax, spec in zip(axs[0], a.runs):
        path, title = spec.split(":", 1)
        D = json.load(open(path))
        segs = D["segments"]
        back = [AuroraNavRunner.motion_stats(sg["traj"])[0] for sg in segs]
        reached = sum(sg["reached"] for sg in segs)
        ax.add_patch(Rectangle((sc.table_pos[0] - sc.table_half[0], sc.table_pos[1] - sc.table_half[1]),
                               2 * sc.table_half[0], 2 * sc.table_half[1], color="#c8a27a", alpha=0.8))
        for sg in segs:
            tr = np.array(sg["traj"])
            if len(tr) < 3:
                continue
            t, x, y, yaw = tr[:, 0], tr[:, 1], tr[:, 2], np.unwrap(tr[:, 3])
            ts = np.arange(t[0], t[-1], 1.8)
            X, Y, W = np.interp(ts, t, x), np.interp(ts, t, y), np.interp(ts, t, yaw)
            for i in range(1, len(ts)):
                f = np.cos(W[i - 1]) * (X[i] - X[i - 1]) + np.sin(W[i - 1]) * (Y[i] - Y[i - 1])
                ax.plot(X[i - 1:i + 1], Y[i - 1:i + 1], color="#d62728" if f < -0.03 else "#1f77b4", lw=2.2)
            g = sg["goal"]
            ax.plot(g[0], g[1], "k*", ms=12)
            ax.arrow(g[0], g[1], 0.15 * np.cos(g[2]), 0.15 * np.sin(g[2]), head_width=0.04, color="k")
        err = np.mean([sg["pos_err"] for sg in segs]) * 100
        tm = np.mean([sg["time"] for sg in segs])
        ax.set_title(f"{title}\n到达 {reached}/{len(segs)}，倒退合计 {sum(back):.1f} m，平均误差 {err:.1f} cm，平均 {tm:.0f} s",
                     fontsize=11)
        ax.set_aspect("equal"); ax.set_xlim(-2.2, 1.6); ax.set_ylim(-1.6, 1.6); ax.grid(alpha=0.3)
    axs[0][0].plot([], [], color="#1f77b4", label="前进/原地")
    axs[0][0].plot([], [], color="#d62728", label="倒退（1.8 s 内后退 >3 cm）")
    axs[0][0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Aurora（AuroraCore + MuJoCo 全物理）里连续走到桌边各个站位：骨盆轨迹俯视图", fontsize=13)
    plt.tight_layout()
    plt.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
