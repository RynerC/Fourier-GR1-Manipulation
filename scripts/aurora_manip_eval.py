"""
scripts/aurora_manip_eval.py —— 在 Aurora 全物理仿真里验收 PdStand 版操作策略（或脚本老师）
=============================================================================================
这是子任务的“单元测试”：每回合重新开机、重新摆场景；“放置”模式开局直接把物体放进手里（为了单独测放置），
不是连续任务。完整的连续流程（一次开机、桌面只摆一次、真的走过去抓再端过去放）见 scripts/aurora_manip_task.py。
场景（站位、物体、托盘、任务模式）用快速环境 AuroraManipEnv 的同一套采样，再在 Aurora 里复现：
  sim_server reset:x,y,yaw 把机器人放到站位 → 吊着开机、站稳（PdStand）→ 按站位坐标系摆物体和托盘
  → 每 0.05 s：观测 → 策略 → 末端目标 → RightArmIK → 右臂 + 腰关节指令，站姿 → set_stand_pose，抓取 → grasp/drop
在 Aurora 镜像里：
  bash aurora_bridge/start_stack.sh --scene tabletop
  python3 scripts/aurora_manip_eval.py --policy scripted --episodes 6
  python3 scripts/aurora_manip_eval.py --policy checkpoints/manip_aurora_bc/bc.npz --episodes 9 --out /tmp/manip.json
"""
import argparse
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aurora_bridge.arm_ik import RightArmIK  # noqa: E402
from aurora_bridge.aurora_compat import AuroraClient  # noqa: E402
from aurora_bridge.locomotion import AuroraLoco  # noqa: E402
from aurora_bridge.manip_runtime import AuroraManipRuntime, SimObjects  # noqa: E402
from aurora_bridge.np_policy import NumpyPolicy  # noqa: E402
from sim.manip_aurora_env import (ARM_TUCK, GRASP_ABOVE, MODE_PICK_HOLD, MODE_PLACE, PELVIS_X0,  # noqa: E402
                                  AuroraManipEnv, AuroraManipSkill, ScriptedManip, carry_point_local)

MODE_NAME = {0: "抓放", 1: "抓起端持", 2: "放置"}


class _EnvLike:
    def __init__(self, world, skill):
        self.world, self.skill = world, skill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(ROOT, "checkpoints", "manip_aurora_bc", "bc.npz"),
                    help="策略 .npz，或 scripted = 脚本老师")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--modes", default="0,1,2", help="只跑这些模式（0 抓放 1 抓起端持 2 放置），轮流")
    ap.add_argument("--trace", action="store_true", help="每 0.5 s 打印一次状态")
    ap.add_argument("--domain-id", type=int, default=123)
    a = ap.parse_args()

    import lcm
    lc_sub, lc_pub = lcm.LCM(), lcm.LCM()
    sim_ctrl = lambda s: lc_pub.publish("sim_ctrl", s.encode())  # noqa: E731
    objs = SimObjects(lc_sub)
    client = AuroraClient.get_instance(domain_id=a.domain_id, robot_name="gr1t2")
    loco = AuroraLoco(client, sim_ctrl=sim_ctrl)
    ik = RightArmIK(os.environ.get("GR1T2_XML", "/usr/local/resources/GR1T2/mjcf/main.xml"))
    env = AuroraManipEnv(seed=a.seed, randomize=False, obs_noise=False)      # 只用来采样场景
    rt = AuroraManipRuntime(client, loco, ik, objs, sim_ctrl, env.world.sc)
    policy = None if a.policy == "scripted" else NumpyPolicy(a.policy)
    results = []
    for ep in range(a.episodes):
        modes = [int(v) for v in a.modes.split(",")]
        mode = modes[ep % len(modes)]
        env.reset(seed=a.seed * 1000 + ep, options={"mode": mode})
        fw, fs = env.world, env.skill
        stance = np.array([fw.ps[0], fw.ps[1], fw.syaw])
        local = {n: fw.to_local(fw.obj_pos(n)) for n in fw.obj_names}
        tray_local = fw.to_local(fw.tray_pos())
        goal_local = fs.goal_local
        target = fs.target
        # ---------- Aurora 里复现场景 ----------
        c, s = np.cos(stance[2]), np.sin(stance[2])
        spawn = stance[:2] + PELVIS_X0 * np.array([c, s])
        for attempt in range(3):
            client.set_velocity_source(0); client.set_fsm_state(0); time.sleep(0.5)
            sim_ctrl("drop")
            sim_ctrl(f"reset:{spawn[0]:.4f},{spawn[1]:.4f},{stance[2]:.4f}"); time.sleep(1.0)
            loco.mode = "default"; loco.upper_hold = {}; loco.stand_pose[:] = 0; loco._pose_active = False
            loco.startup()
            if loco.stable_level() >= 15 and client.get_true_data("pos_W")[2] > 0.7:
                break
            loco.log("[eval] 开机没站稳（新启动的 AuroraCore 第一次预热偶尔双腿不对称），重来")
        loco.upper_hold = {"right_manipulator": np.zeros(7), "waist": np.zeros(3)}
        objs.wait()
        xy, yaw = rt.begin()
        for n, p in local.items():
            if mode == MODE_PLACE and n == target:
                continue
            w = rt.to_world(p)
            sim_ctrl(f"place:{n}:{w[0]:.4f},{w[1]:.4f}")
        tw = rt.to_world(tray_local)
        sim_ctrl(f"place:tray:{tw[0]:.4f},{tw[1]:.4f}")
        rt.grip, rt.up_axis = 0.0, None
        if mode == MODE_PLACE:
            loco.ramp_upper({"right_manipulator": ARM_TUCK, "waist": [0, 0, 0]}, T=2.0)
            rt.refresh()
            e = rt.ee_pos()
            for _ in range(3):                   # 物体瞬移到手下方 7 cm 并抓住（= 快速环境模式 C 的初始状态）
                sim_ctrl(f"place:{target}:{e[0]:.4f},{e[1]:.4f},{e[2] - GRASP_ABOVE:.4f},0")
                sim_ctrl("grasp:" + target)          # 紧接着发（中间等待的话物体已经开始下落）
                t1 = time.time()
                while objs.data.get("held") != target and time.time() - t1 < 1.0:
                    time.sleep(0.05)
                if objs.data.get("held") == target:
                    break
            rt.refresh()
            rt.grip, rt.held = 1.0, target
            rt.up_axis = rt._ee_R.T @ np.array([0.0, 0.0, 1.0])
        loco.wait(0.5)
        rt.refresh()
        sk = AuroraManipSkill(rt, obs_noise=False)
        if mode == MODE_PICK_HOLD:
            sk.set_task(target, goal_local=goal_local, release=False)
        else:
            sk.set_task(target, goal_world=sk.tray_goal(name=target), release=True)
        sk.begin()
        teacher = ScriptedManip(_EnvLike(rt, sk)) if policy is None else None
        if teacher:
            teacher.reset()
        sim_ctrl(f"label:{MODE_NAME[mode]} {target}")
        loco.log(f"[ep {ep}] {MODE_NAME[mode]} {target}：站位 ({xy[0]:+.2f},{xy[1]:+.2f},{np.degrees(yaw):+.0f}°)，"
                 f"物体在站位系 {np.round(local[target], 2)}，托盘 {np.round(tray_local, 2)}")
        t0 = time.time()
        info = {"success": False}
        for t in range(a.max_steps):
            o = sk.obs()
            act = teacher() if teacher else policy(o)
            sk.act(act)
            r, succ, dropped, info = sk.evaluate()
            if a.trace and t % 10 == 0:
                loco.log(f"      t={t * 0.05:4.1f}s 手 {np.round(rt.to_local(rt.ee_pos()), 3)} 物体 {np.round(rt.to_local(rt.obj_pos(target)), 3)} "
                         f"目标 {np.round(rt.to_local(sk.goal_world), 3)} grip {rt.grip:.2f} held {rt.held} 站姿 {np.round(rt.stand, 2)} "
                         f"动作 {np.round(act, 2)}")
            if succ or dropped:
                break
        z = float(client.get_true_data("pos_W")[2])
        res = {"ep": ep, "mode": mode, "target": target, "success": bool(info["success"]), "steps": t + 1,
               "time": time.time() - t0, "d_goal": info.get("d_goal"), "dropped": bool(info.get("dropped")),
               "fell": z < 0.6, "tilt": rt.obj_tilt(target)}
        loco.log(f"    → 成功={res['success']} 步数 {t + 1}（{(t + 1) * 0.05:.1f} s）d_goal {res['d_goal'] * 100:.1f} cm "
                 f"倾斜 {res['tilt']:.0f}°" + ("  掉了!" if res["dropped"] else "") + ("  摔倒!" if res["fell"] else ""))
        results.append(res)
        loco.ramp_upper({"right_manipulator": np.zeros(7), "waist": [0, 0, 0]}, T=1.5)
        loco.ramp_stand_pose((0, 0, 0))
    summ = {"policy": a.policy, "n": len(results), "success": sum(r["success"] for r in results),
            "falls": sum(r["fell"] for r in results),
            "per_mode": {MODE_NAME[k]: f"{sum(r['success'] for r in results if r['mode'] == k)}/"
                                       f"{sum(1 for r in results if r['mode'] == k)}" for k in range(3)},
            "mean_time_s": float(np.mean([r["steps"] * 0.05 for r in results]))}
    print("SUMMARY", json.dumps(summ, ensure_ascii=False), flush=True)
    if a.out:
        json.dump({"summary": summ, "episodes": results}, open(a.out, "w"), ensure_ascii=False, indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
