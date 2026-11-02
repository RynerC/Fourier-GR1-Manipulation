"""
aurora_bridge/manip_runtime.py —— 把快速环境训练的操作策略接到 Aurora（PdStand）上
===================================================================================
AuroraManipRuntime 提供和 sim/manip_aurora_env.AuroraManipWorld 一样的接口（ee_pos、obj_pos、to_local、
solve_ik、set_arm、step …），所以 AuroraManipSkill（观测、动作→指令、成功判定）和 ScriptedManip 老师
在 Aurora 上一行不改就能用：
  * 末端 / 骨盆：RightArmIK.sync(client)（真值位置 + IMU 姿态 + 关节角）做正运动学；
  * 物体：仿真里来自 LCM 'sim_objects'（位置 + 姿态 + 是否抓住），真机上换成感知结果；
  * 手臂 + 腰：IK 解出的关节目标写进 AuroraLoco.upper_hold（每个控制周期 set_group_cmd 发出）；
  * 站姿：AuroraLoco.stand_pose（set_stand_pose），按 Aurora 允许的斜坡速度积分；
  * 抓取：和快速环境同一规则（抓取点 = 末端下方 7 cm，离物体中心 < 3.5 cm 且手已握紧）→ 仿真发 'grasp:<物体>'，
    松手发 'drop'；真机上换成手部控制组的开合指令。
"""
import json
import threading
import time

import numpy as np

from sim.manip_aurora_env import CTRL_DT, GRASP_ABOVE, GRASP_RADIUS, STAND_LIM, STAND_RATE, _rz
from sim import gr1t2_manip_scene as S


class SimObjects:
    """订阅 sim_server 的 'sim_objects'（JSON：位置、姿态、held）。"""

    def __init__(self, lc):
        self.data, self.t = {}, 0.0
        lc.subscribe("sim_objects", self._on)

        def loop():
            while True:
                lc.handle_timeout(50)
        threading.Thread(target=loop, daemon=True).start()

    def _on(self, ch, msg):
        self.data = json.loads(msg.decode())
        self.t = time.time()

    def wait(self, timeout=5.0):
        t0 = time.time()
        while not self.data and time.time() - t0 < timeout:
            time.sleep(0.05)
        return bool(self.data)


class AuroraManipRuntime:
    def __init__(self, client, loco, ik, objs, sim_ctrl, scene_cfg, rng=None):
        self.c, self.loco, self.ik, self.objs, self.sim_ctrl = client, loco, ik, objs, sim_ctrl
        self.sc = scene_cfg
        self.rng = rng or np.random.default_rng(0)
        self.obj_names = list(scene_cfg.objects)
        self.obj_half_h = {n: (s[1][2] if s[0] == "box" else s[1][1]) for n, s in scene_cfg.objects.items()}
        self.table_h = scene_cfg.table_height
        m = ik.m
        self.lo_arm = np.array([m.jnt_range[m.joint(j).id] for j in S.RIGHT_ARM])
        self.lf, self.rf = m.body("left_foot_roll_link").id, m.body("right_foot_roll_link").id
        self.grip, self.held_local = 0.0, None
        self.up_axis = None
        self._prev = {}

    # ------------------------------------------------------------------ 站位坐标系 = 双脚中点 + 朝向
    def begin(self):
        self.refresh()
        d = self.ik.d
        feet = 0.5 * (d.xpos[self.lf] + d.xpos[self.rf])
        Rf = d.xmat[self.lf].reshape(3, 3) + d.xmat[self.rf].reshape(3, 3)
        yaw = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
        self.ps, self.Rs, self.syaw = np.array([feet[0], feet[1], 0.0]), _rz(yaw), yaw
        self.stand = np.array(self.loco.stand_pose, float)
        return self.ps[:2].copy(), yaw

    def refresh(self):
        self.ik.sync(self.c)
        now = time.time()
        ee = self.ik.ee_pos()
        if "ee" in self._prev:
            dt = max(now - self._prev["t"], 1e-3)
            self._ee_vel = (ee - self._prev["ee"]) / dt
        else:
            self._ee_vel = np.zeros(3)
        self._prev.update(ee=ee, t=now)
        self._ee = ee
        self._ee_R = self.ik.d.xmat[self.ik.ee].reshape(3, 3).copy()
        od = self.objs.data
        self._obj = {n: np.array(od[n]) for n in self.obj_names if n in od}
        self._obj_q = od.get("quat", {})
        self._tray = np.array(od["tray"]) if "tray" in od else None
        for n, p in self._obj.items():
            key = "o_" + n
            if key in self._prev and self.objs.t > self._prev[key][2]:
                p0, t0, _ = self._prev[key]
                self._prev["v_" + n] = (p - p0) / max(self.objs.t - t0, 1e-3)
                self._prev[key] = (p, self.objs.t, self.objs.t)
            elif key not in self._prev:
                self._prev[key] = (p, self.objs.t, self.objs.t)
        self.held = od.get("held")

    # ------------------------------------------------------------------ 与 AuroraManipWorld 相同的接口
    def to_local(self, p):
        return self.Rs.T @ (np.asarray(p) - self.ps)

    def to_world(self, p):
        return self.ps + self.Rs @ np.asarray(p)

    def vec_local(self, v):
        return self.Rs.T @ np.asarray(v)

    def ee_pos(self):
        return self._ee.copy()

    def ee_vel(self):
        return self._ee_vel.copy()

    def grasp_point(self):
        return self._ee - np.array([0, 0, GRASP_ABOVE])

    def obj_pos(self, n):
        return self._obj[n].copy()

    def obj_vel(self, n):
        return self._prev.get("v_" + n, np.zeros(3)).copy()

    def obj_tilt(self, n):
        q = self._obj_q.get(n)
        if q is None:
            return 0.0
        w, x, y, z = q
        zz = 1 - 2 * (x * x + y * y)               # 旋转矩阵 R[2,2]
        return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))

    def tray_pos(self):
        return self._tray.copy()

    def arm_q(self):
        return np.array(self.c.get_group_state("right_manipulator", "position"))

    def waist_q(self):
        return np.array(self.c.get_group_state("waist", "position"))

    def show_goal(self, p):
        pass

    def contacts(self):
        return False, False

    def solve_ik(self, target_world, iters=25):
        up = self.up_axis if self.held is not None else None
        q, waist, err = self.ik.solve(target_world, iters=iters, up_axis=up)
        return np.asarray(q), np.asarray(waist), err

    def set_arm(self, q_arm, q_waist, instant=False):
        q_arm = np.clip(q_arm, self.lo_arm[:, 0], self.lo_arm[:, 1])
        self.loco.upper_hold["right_manipulator"] = np.asarray(q_arm, float)
        self.loco.upper_hold["waist"] = np.asarray(q_waist, float)

    def step(self, stand_rate, grip_cmd):
        lo = self.loco
        self.stand = np.clip(self.stand + np.clip(stand_rate, -1, 1) * STAND_RATE * CTRL_DT,
                             STAND_LIM[:, 0], STAND_LIM[:, 1])
        lo.stand_pose = self.stand.copy()
        lo._pose_active = True
        self.grip = float(np.clip(self.grip + (0.25 if grip_cmd > 0 else -0.25), 0, 1))
        # 抓取规则与快速环境一致
        if self.held is None and self.grip > 0.6:
            gp = self.grasp_point()
            best, bd = None, GRASP_RADIUS
            for n, p in self._obj.items():
                dd = np.linalg.norm(p - gp)
                if dd < bd:
                    best, bd = n, dd
            if best is not None:
                self.sim_ctrl("grasp:" + best)
                R = self._ee_R
                q = self._obj_q.get(best, [1, 0, 0, 0])
                w, x, y, z = q
                obj_up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
                self.up_axis = R.T @ obj_up
                self.held = best
        elif self.held is not None and self.grip < 0.3:
            self.sim_ctrl("drop")
            self.held = None
            self.up_axis = None
        lo.wait(CTRL_DT)
        self.refresh()
