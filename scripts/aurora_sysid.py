"""
scripts/aurora_sysid.py —— 测 Aurora 行走控制器的速度响应（系统辨识），结果用来更新 aurora_bridge/gait_model.py
==================================================================================================================
在 Aurora 镜像里（先 bash aurora_bridge/start_stack.sh --truth-log /tmp/truth.csv）：
  python3 scripts/aurora_sysid.py --truth /tmp/truth.csv
按顺序发一串分段速度指令（前进/后退/横移/转向/弧线，每段后停 3 s），然后读仿真真值日志，
输出每段的稳态机体速度、横摆幅度、偏航率、63% 上升时间、停车滑行距离。
"""
import argparse
import json
import os
import sys
import time

# 必须在 import numpy 之前：Aurora 镜像里的 numpy 用 OpenBLAS（pthread 版），默认按 CPU 核数开线程。
# 导航策略每次 256×256 的矩阵乘都会被拆给全部线程（28 核机器上 28 个），算完后这些线程还会忙等约 0.1 s，
# 10 Hz 调用 → 几十个核一直被占满 → AuroraCore 的行走任务 26% 的周期超时 → 步态失稳摔倒。网络很小，单线程足够（<0.1 ms）。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEGMENTS = [("stand", [0, 0, 0], 3), ("fwd0.15", [0.15, 0, 0], 6), ("stop", [0, 0, 0], 3),
            ("fwd0.3", [0.3, 0, 0], 6), ("stop", [0, 0, 0], 3), ("fwd0.45", [0.45, 0, 0], 6), ("stop", [0, 0, 0], 3),
            ("back-0.15", [-0.15, 0, 0], 5), ("stop", [0, 0, 0], 3), ("left0.2", [0, 0.2, 0], 5), ("stop", [0, 0, 0], 3),
            ("right-0.2", [0, -0.2, 0], 5), ("stop", [0, 0, 0], 3), ("yaw0.5", [0, 0, 0.5], 5), ("stop", [0, 0, 0], 3),
            ("yaw-0.5", [0, 0, -0.5], 5), ("stop", [0, 0, 0], 3), ("yaw0.15", [0, 0, 0.15], 5), ("stop", [0, 0, 0], 3),
            ("fwd0.12", [0.12, 0, 0], 5), ("stop", [0, 0, 0], 3), ("fwd0.10", [0.10, 0, 0], 5), ("stop", [0, 0, 0], 3),
            ("arc", [0.3, 0, 0.4], 6), ("stop", [0, 0, 0], 3)]


def run(marks_path):
    import lcm
    from aurora_bridge.aurora_compat import AuroraClient
    from aurora_bridge.locomotion import AuroraLoco
    lc = lcm.LCM()
    c = AuroraClient.get_instance(domain_id=123, robot_name="gr1t2")
    lo = AuroraLoco(c, sim_ctrl=lambda s: lc.publish("sim_ctrl", s.encode()))
    c.set_fsm_state(0); time.sleep(0.5); lo.sim_ctrl("reset"); time.sleep(1.0)
    lo.startup()
    lo.walk_mode()
    marks = []
    for tag, cmd, dur in SEGMENTS:
        marks.append((time.time(), tag, cmd))
        t0 = time.time()
        while time.time() - t0 < dur:          # 直接发原始指令（不经过变化率限制），测的是 Aurora 本身
            c.set_velocity(*cmd); time.sleep(0.05)
    c.set_velocity(0, 0, 0)
    marks.append((time.time(), "end", [0, 0, 0]))
    json.dump(marks, open(marks_path, "w"))


def analyze(truth, marks_path):
    T = np.loadtxt(truth, delimiter=",")
    M = json.load(open(marks_path))
    k = 50
    sm = lambda x: np.convolve(x, np.ones(k) / k, mode="same")  # noqa: E731
    for (t0, tag, cmd), (t1, _, _) in zip(M[:-1], M[1:]):
        S = T[(T[:, 0] >= t0) & (T[:, 0] < t1)]
        if not len(S):
            continue
        Q = S[len(S) // 2:]
        yr = np.diff(np.unwrap(Q[:, 5])) / np.diff(Q[:, 1])
        ax = 9 if abs(cmd[0]) >= abs(cmd[1]) else 10
        tgt = cmd[0] if ax == 9 else cmd[1]
        t63 = float("nan")
        if abs(tgt) > 0:
            v = sm(S[:, ax]) * np.sign(tgt)
            ss = np.mean(v[len(v) // 2:])
            i = np.argmax(v >= 0.63 * ss)
            t63 = S[i, 0] - S[0, 0] if ss > 0.02 else float("nan")
        disp = np.linalg.norm(S[-1, 2:4] - S[0, 2:4])
        print(f"{tag:9s} cmd={str(cmd):16s} 稳态 vB=({Q[:, 9].mean():+.2f},{Q[:, 10].mean():+.2f}) "
              f"偏航率 {yr.mean():+.2f}  横摆 {Q[:, 10].std():.2f}  t63 {t63:.1f}s  位移 {disp:.2f} m"
              f"{'  (停车滑行)' if tag == 'stop' else ''}  z[{S[:, 4].min():.2f},{S[:, 4].max():.2f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="sim_server.py --truth-log 写的 CSV")
    ap.add_argument("--marks", default="/tmp/sysid_marks.json")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        run(a.marks)
        time.sleep(1.0)
    analyze(a.truth, a.marks)
    os._exit(0)
