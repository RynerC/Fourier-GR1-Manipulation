"""
scripts/plot_progress.py —— 画训练曲线（回报 / 成功率 / 回合长度 / 策略标准差）
用法： python scripts/plot_progress.py --task nav_aurora
看曲线是 RL 调参最重要的习惯：回报上升但成功率不变 → 奖励设计可能被“钻空子”；
std 过早降到很小 → 探索不足；KL 经常触发早停 → 学习率偏大。
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import CKPT_DIR, VIDEO_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="nav_aurora / manip_aurora，或任意 checkpoints/ 下的子目录名（--run-name）")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = os.path.join(CKPT_DIR, args.task)
    # progress.csv = 主训练；progress_finetune*.csv = 之后的微调（从某个检查点接着训练）
    # progress_stage*.csv = nav_aurora 的两阶段记录
    extra = sorted(x for x in os.listdir(d) if x.startswith("progress_") and x.endswith(".csv"))
    files = ([("progress.csv", "#2a6fdb", "train")] if os.path.exists(os.path.join(d, "progress.csv")) else []) + [
        (f, c, f[9:-4]) for f, c in zip(extra, ["#e8710a", "#1e8e3e", "#a142f4", "#c5221f"])]
    # 每个点 = 最近 100 个回合的平均。回合之间起点/目标远近差很多（回报、长度天生不同），
    # 100 个回合的平均仍会有 ±几% 的统计噪声 —— 所以画“原始值（淡）+ 滑动平均（粗）”，
    # 成功率固定 0–1 的纵轴，避免自动缩放把 0.98↔1.00 的抖动放大成满屏锯齿。
    keys = ["ep_return", "success", "ep_len", "std"]
    titles = ["episode return", "success rate", "episode length", "policy std"]
    rows0 = list(csv.DictReader(open(os.path.join(d, files[0][0]))))
    for k, t in (("back_dist", "backward dist / ep (m)"), ("dist", "final dist (m)")):
        if k in rows0[0]:
            keys.append(k); titles.append(t)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4))
    win = 10

    def smooth(v):
        v = np.asarray(v, float)
        out = np.full_like(v, np.nan)
        for i in range(len(v)):
            seg = v[max(0, i - win + 1):i + 1]
            seg = seg[~np.isnan(seg)]
            out[i] = seg.mean() if len(seg) else np.nan
        return out
    for fname, color, label in files:
        rows = list(csv.DictReader(open(os.path.join(d, fname))))
        steps = [int(r["steps"]) / 1e6 for r in rows]
        for ax, key in zip(axes, keys):
            v = [float(r[key]) if r.get(key) not in (None, "", "nan") else np.nan for r in rows]
            ax.plot(steps, v, color=color, lw=0.8, alpha=0.3)
            ax.plot(steps, smooth(v), color=color, lw=1.8, label=label)
    for ax, title, key in zip(axes, titles, keys):
        ax.set_title(title); ax.set_xlabel("env steps (M)"); ax.grid(alpha=0.3)
        if key == "success":
            ax.set_ylim(0, 1.02)
        if key in ("back_dist", "dist", "std"):
            ax.set_ylim(bottom=0)
    if len(files) > 1:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(VIDEO_DIR, exist_ok=True)
    out = os.path.join(VIDEO_DIR, f"curve_{args.task}.png")
    fig.savefig(out, dpi=110)
    print("已保存", out)


if __name__ == "__main__":
    main()
