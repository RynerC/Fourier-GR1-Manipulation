"""
scripts/aurora_nav_eval.py —— 在 Aurora（AuroraCore + MuJoCo 全物理）里验证导航
================================================================================
在 Aurora 镜像里运行（先启动 sim_server.py --scene tabletop 和 AuroraCore，见 aurora_bridge/README）：
  python3 scripts/aurora_nav_eval.py --policy checkpoints/nav_aurora/best.npz --tour 6
  python3 scripts/aurora_nav_eval.py --policy follower --tour 6
机器人从桌子 -x 侧出发，依次走到桌子四周随机的抓取站位（与任务规划器的输出同分布），
连续走完一圈不重置。每一段记录：是否到达、用时、最终误差、倒退距离、横移距离；摔倒则重置。
结果写到 --out（JSON，含轨迹）。
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

from aurora_bridge.aurora_compat import AuroraClient  # noqa: E402
from aurora_bridge.locomotion import AuroraLoco, AuroraNavRunner, ForwardFirstFollower  # noqa: E402
from aurora_bridge.np_policy import NumpyPolicy  # noqa: E402
from config import RobotCfg, SceneCfg  # noqa: E402
from planning.geometry import TableGeometry  # noqa: E402


def tour_goals(geo, n, rng):
    goals = []
    sides = [s for s, *_ in geo.sides()]
    for i in range(n):
        for _ in range(100):
            h = geo.half - 0.06
            target = geo.c + rng.uniform(-h, h)
            side = sides[(i * 3 + 1) % 4] if i % 2 == 0 else rng.choice(sides)
            pose, ok, _ = geo.stand_pose_for(target, side)
            if ok:
                goals.append(np.array([pose.x, pose.y, pose.yaw]))
                break
    return goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "checkpoints", "nav_aurora", "best.npz"))
    ap.add_argument("--tour", type=int, default=6, help="连续走几个站位")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--spawn", default="-1.6,0.0,0.0")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--domain-id", type=int, default=123)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-guard", action="store_true", help="关闭安全层（桌前限速 + 平衡保护），对比实验用")
    a = ap.parse_args()

    import lcm
    lc = lcm.LCM()
    sim_ctrl = lambda s: lc.publish("sim_ctrl", s.encode())  # noqa: E731
    client = AuroraClient.get_instance(domain_id=a.domain_id, robot_name="gr1t2")
    geo = TableGeometry(SceneCfg(), RobotCfg())
    if a.policy == "follower":
        policy, name = ForwardFirstFollower(geo), "follower"
    else:
        policy, name = NumpyPolicy(a.policy), os.path.basename(os.path.dirname(a.policy))
    loco = AuroraLoco(client, sim_ctrl=sim_ctrl)

    def boot():
        client.set_velocity_source(0); client.set_fsm_state(0); time.sleep(0.5)
        sim_ctrl("reset:" + a.spawn); time.sleep(1.0)
        loco.mode = "default"; loco.upper_hold = {}
        loco.startup()
        loco.walk_mode()

    boot()
    runner = AuroraNavRunner(loco, geo, policy, safety=not a.no_guard)
    rng = np.random.default_rng(a.seed)
    results = []
    for i, g in enumerate(tour_goals(geo, a.tour, rng)):
        loco.log(f"[{name}] 目标 {i + 1}/{a.tour}: ({g[0]:+.2f}, {g[1]:+.2f}, {np.degrees(g[2]):+.0f}°)")
        r = runner.go_to(g, timeout=a.timeout)
        z = client.get_true_data("pos_W")[2]
        r["fell"] = bool(z < 0.6)
        r["goal"] = g.tolist()
        loco.log(f"    → 到达={r['reached']} 用时 {r['time']:.1f}s 误差 {r['pos_err'] * 100:.1f}cm/"
                 f"{np.degrees(r['yaw_err']):.1f}° 倒退 {r['back_dist']:.2f}m 横移 {r['lateral_dist']:.2f}m"
                 f" 限速介入 {r['guard_steps']} 步 离桌最近 {r['min_front']:.2f}m 平衡保护 {r['balance_steps']} 步 最大横滚 {r['max_roll']:.2f}"
                 + ("  摔倒!" if r["fell"] else ""))
        results.append(r)
        if r["fell"]:
            boot()
    ok = [r for r in results if r["reached"]]
    back = sum(r["back_dist"] for r in results)
    summ = {"policy": name, "n": len(results), "reached": len(ok), "falls": sum(r["fell"] for r in results),
            "mean_pos_err_cm": float(np.mean([r["pos_err"] for r in results]) * 100),
            "mean_yaw_err_deg": float(np.degrees(np.mean([r["yaw_err"] for r in results]))),
            "total_back_m": back, "max_back_m": max(r["back_dist"] for r in results),
            "total_lateral_m": sum(r["lateral_dist"] for r in results),
            "mean_time_s": float(np.mean([r["time"] for r in results])),
            "max_roll": max(r["max_roll"] for r in results)}
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summ, "segments": results}, f)
    loco.stop()
    os._exit(0)


if __name__ == "__main__":
    main()
