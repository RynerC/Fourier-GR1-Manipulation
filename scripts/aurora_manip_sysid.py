"""
scripts/aurora_manip_sysid.py —— 测 Aurora PdStand 下“站姿 + 上身”的响应，给快速操作训练环境 sim/manip_aurora_env.py 定参数
==============================================================================================================================
在 Aurora 镜像里（先 bash aurora_bridge/start_stack.sh）：
  python3 scripts/aurora_manip_sysid.py --out docs/aurora_results/manip_sysid.json
测三件事：
  1. 站姿 (Δz, Δpitch, Δyaw) → 骨盆位姿（相对双脚中点）的稳态映射
  2. 右臂 + 腰关节指令 → 实际关节角的稳态误差（重力下垂）、阶跃响应（时间常数）、末端误差
  3. 手上负载（0.15 / 0.5 kg）引起的末端下垂
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

from aurora_bridge.arm_ik import GROUP_JOINTS, RightArmIK  # noqa: E402
from aurora_bridge.aurora_compat import AuroraClient  # noqa: E402
from aurora_bridge.locomotion import AuroraLoco  # noqa: E402

STAND_POSES = [(0, 0, 0), (-0.06, 0, 0), (-0.12, 0, 0), (0, 0.2, 0), (0, 0.4, 0), (-0.06, 0.2, 0),
               (0, 0, 0.3), (0, 0, -0.3), (-0.06, 0.2, 0.3), (-0.06, 0.2, -0.3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "aurora_results", "manip_sysid.json"))
    ap.add_argument("--domain-id", type=int, default=123)
    a = ap.parse_args()
    import lcm
    lc = lcm.LCM()
    sim_ctrl = lambda s: lc.publish("sim_ctrl", s.encode())  # noqa: E731
    c = AuroraClient.get_instance(domain_id=a.domain_id, robot_name="gr1t2")
    lo = AuroraLoco(c, sim_ctrl=sim_ctrl)
    ik = RightArmIK(os.environ.get("GR1T2_XML", "/usr/local/resources/GR1T2/mjcf/main.xml"))
    m = ik.m
    lf, rf = m.body("left_foot_roll_link").id, m.body("right_foot_roll_link").id
    base = m.body("base_link").id
    torso = m.body("torso_link").id

    c.set_velocity_source(0); c.set_fsm_state(0); time.sleep(0.5)
    sim_ctrl("reset:0,0,0"); time.sleep(1.0)
    lo.mode = "default"; lo.upper_hold = {}
    lo.startup()
    lo.upper_hold = {"right_manipulator": np.zeros(7), "left_manipulator": np.zeros(7), "waist": np.zeros(3)}
    lo.wait(1.0)

    def snapshot(n=10, dt=0.05):
        """平均 n 次：骨盆/躯干/右手末端 在“双脚坐标系”下的位置和姿态，以及关节角。"""
        acc = []
        for _ in range(n):
            ik.sync(c)
            d = ik.d
            feet = 0.5 * (d.xpos[lf] + d.xpos[rf])
            Rf = d.xmat[lf].reshape(3, 3) + d.xmat[rf].reshape(3, 3)
            fyaw = np.arctan2(Rf[1, 0], Rf[0, 0])
            cy, sy = np.cos(fyaw), np.sin(fyaw)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            loc = lambda p: Rz.T @ (p - feet)  # noqa: E731
            Rb = Rz.T @ d.xmat[base].reshape(3, 3)
            Rt = Rz.T @ d.xmat[torso].reshape(3, 3)
            rpy = lambda R: [np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])]  # noqa: E731
            q = {g: np.array(c.get_group_state(g, "position")) for g in ("waist", "right_manipulator")}
            acc.append(np.concatenate([loc(d.xpos[base]), rpy(Rb), loc(d.xpos[torso]), rpy(Rt),
                                       loc(ik.ee_pos()), q["waist"], q["right_manipulator"]]))
            lo.wait(dt)
        v = np.mean(acc, 0)
        return {"pelvis": v[0:3].tolist(), "pelvis_rpy": v[3:6].tolist(), "torso": v[6:9].tolist(),
                "torso_rpy": v[9:12].tolist(), "ee": v[12:15].tolist(), "waist_q": v[15:18].tolist(),
                "arm_q": v[18:25].tolist()}

    out = {"stand": [], "arm": [], "step": [], "payload": []}
    # ---------------- 1. 站姿 ----------------
    for sp in STAND_POSES:
        lo.ramp_stand_pose(sp, settle=1.5)
        s = snapshot()
        s["cmd"] = list(sp)
        out["stand"].append(s)
        lo.log(f"[stand] {sp} → pelvis {np.round(s['pelvis'], 3)} rpy {np.round(s['pelvis_rpy'], 3)}")
    lo.ramp_stand_pose((0, 0, 0))

    # ---------------- 2. 手臂稳态 ----------------
    MANIP = (-0.06, 0.2, 0.0)
    lo.ramp_stand_pose(MANIP, settle=1.0)
    rng = np.random.default_rng(0)
    for k in range(14):
        # 双脚坐标系下的目标：前 0.35–0.62、右 -0.32–0.05、高 0.82–1.05（桌面 0.80）
        tgt_l = np.array([rng.uniform(0.35, 0.62), rng.uniform(-0.32, 0.05), rng.uniform(0.82, 1.05)])
        ik.sync(c)
        d = ik.d
        feet = 0.5 * (d.xpos[lf] + d.xpos[rf])
        tgt = feet + tgt_l          # 本测试从 yaw=0 出生，双脚朝向≈0
        q, waist, res = ik.solve(tgt)
        cmd = {"right_manipulator": np.array(q), "waist": np.array(waist)}
        lo.ramp_upper(cmd, T=2.0, settle=1.2)
        s = snapshot()
        # 用“指令关节角”在同一个骨盆位姿下做正运动学：= 理想末端位置
        ik.sync(c)
        for n, v in zip(GROUP_JOINTS["right_manipulator"], q):
            ik.d.qpos[ik.qadr[n]] = v
        for n, v in zip(GROUP_JOINTS["waist"], waist):
            ik.d.qpos[ik.qadr[n]] = v
        import mujoco
        mujoco.mj_kinematics(m, ik.d)
        s.update({"target_local": tgt_l.tolist(), "ik_res": res, "cmd_arm": list(map(float, q)),
                  "cmd_waist": list(map(float, waist)),
                  "ee_cmd_fk_err": float(np.linalg.norm(ik.ee_pos() - (feet + np.array(s["ee"]))))})
        out["arm"].append(s)
        lo.log(f"[arm {k}] 目标 {np.round(tgt_l, 2)} IK 残差 {res * 100:.1f}cm 关节误差 "
               f"{np.round(np.array(s['arm_q']) - q, 3)} 末端偏差 {s['ee_cmd_fk_err'] * 100:.1f}cm")

    # ---------------- 3. 阶跃响应（肘、肩 pitch、腰 pitch） ----------------
    base_q = {g: np.array(c.get_group_state(g, "position")) for g in ("right_manipulator", "waist")}
    for g, j, dq in (("right_manipulator", 3, -0.3), ("right_manipulator", 0, -0.3), ("waist", 1, 0.15),
                     ("right_manipulator", 4, 0.4)):
        q0 = np.array(lo.upper_hold.get(g, base_q[g]), float)
        q1 = q0.copy(); q1[j] += dq
        lo.upper_hold[g] = q1
        t0 = time.time(); tr = []
        while time.time() - t0 < 1.5:
            lo._tick()
            tr.append((time.time() - t0, float(c.get_group_state(g, "position")[j])))
            time.sleep(0.01)
        out["step"].append({"group": g, "joint": j, "q0": float(q0[j]), "q1": float(q1[j]), "trace": tr})
        tr = np.array(tr)
        frac = (tr[:, 1] - tr[0, 1]) / (q1[j] - tr[0, 1])
        t63 = tr[np.argmax(frac > 0.63), 0] if (frac > 0.63).any() else None
        lo.log(f"[step] {g}[{j}] {dq:+.2f} rad：t63 {t63} s，终值误差 {q1[j] - tr[-1, 1]:+.3f}")
        lo.upper_hold[g] = q0
        lo.wait(1.0)

    # ---------------- 4. 负载 ----------------
    s0 = snapshot()
    for kg in (0.15, 0.5, 1.0):
        sim_ctrl(f"payload:{kg}"); lo.wait(1.5)
        s = snapshot()
        drop = np.array(s["ee"]) - np.array(s0["ee"])
        out["payload"].append({"kg": kg, "ee_shift": drop.tolist(), "pelvis_shift": (np.array(s["pelvis"]) - np.array(s0["pelvis"])).tolist(),
                               "arm_q_shift": (np.array(s["arm_q"]) - np.array(s0["arm_q"])).tolist()})
        lo.log(f"[payload {kg} kg] 末端位移 {np.round(drop * 100, 1)} cm")
    sim_ctrl("payload:0")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out, flush=True)
    lo.ramp_upper({"right_manipulator": np.zeros(7), "waist": np.zeros(3)}, T=2.0)
    lo.ramp_stand_pose((0, 0, 0))
    os._exit(0)


if __name__ == "__main__":
    main()
