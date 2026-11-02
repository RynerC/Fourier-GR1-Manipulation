"""
sim/manip_aurora_env.py —— Aurora PdStand 版操作技能的快速训练环境（GR1T2，站着不挪脚）
=====================================================================================
一个目标条件策略做三种子任务
  模式 A  pick & place ：抓起物体放到托盘上并松手
  模式 B  pick & hold  ：抓起物体端到“携带点”（之后交给导航，端着走）
  模式 C  place        ：手里已有物体（端持姿态），放到托盘上并松手
不同的是机器人模型和“能动什么”都换成部署时 Aurora 真正允许的：

  * 不能挪脚（PdStand 下没有底盘动作）。代替底盘微调的是站姿 (Δz, Δpitch, Δyaw)：降低高度、整体前俯、
    原地转身，Aurora 用全身控制器保持平衡。实测（scripts/aurora_manip_sysid.py）：骨盆高度 = 0.836 + Δz、
    俯仰 = Δpitch、偏航 = Δyaw；伸手时骨盆会往后让 1–4 cm（质心补偿）→ 本环境按拟合公式复现。
  * 手臂 + 腰：动作给“末端目标增量”，环境里用与部署相同的 IK（aurora_bridge/arm_ik.RightArmIK：
    右臂 7 关节 + 腰 yaw/pitch 阻尼最小二乘）解出关节目标，按 Aurora PdStand 的 kp/kd 位置伺服，
    加 0.02–0.06 s 指令延迟（实测阶跃响应 10% 上升前有约 0.07 s 死区）。
  * 没有手指：抓取辅助和 Aurora 仿真 sim_server 一样（末端在物体正上方时“吸住”）。

动作（7 维，[-1, 1]）：
  [0:3] 末端目标在“站位坐标系”下的增量，每步最多 1.5 cm（20 Hz → 0.3 m/s）
  [3]   抓取 (>0 抓)
  [4:7] 站姿变化率 Δz、Δpitch、Δyaw，满幅 = Aurora 允许的斜坡速度 0.05 m/s、0.15 rad/s、0.15 rad/s

部署（Aurora 里）：策略输出 → 末端目标 → RightArmIK.solve → set_group_cmd（右臂 + 腰），
站姿 → set_stand_pose，抓取 → 真机手 / 仿真 grasp:<物体>。见 scripts/aurora_manip_eval.py。
"""
import os

import mujoco
import numpy as np

try:
    import gymnasium as gym
except ImportError:          # Aurora 镜像里没有 gymnasium：验收脚本只用到环境的场景采样，给一个最小替身
    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class gym:  # noqa: N801
        Env = object
        spaces = type("spaces", (), {"Box": _Box})
import numpy as np

from config import RobotCfg, SceneCfg
from planning.geometry import LAT_MAX, LAT_MIN, REACH_DEPTH, TableGeometry
from sim import gr1t2_manip_scene as S

MODE_PICK_PLACE, MODE_PICK_HOLD, MODE_PLACE = 0, 1, 2
CTRL_DT = 0.05                     # 策略 20 Hz（= AuroraLoco 的指令频率）
EE_STEP = 0.015                    # 每步末端目标最大增量 (m)
STAND_RATE = np.array([0.05, 0.15, 0.15])                       # Aurora 站姿斜坡上限
STAND_LIM = np.array([[-0.12, 0.0], [0.0, 0.35], [-0.5, 0.5]])  # 本任务允许的站姿范围
GRASP_ABOVE = 0.07                 # 抓取时末端在物体中心正上方 7 cm（物体半高 6 cm → 末端离物体顶 1 cm）
GRASP_RADIUS = 0.035
TRAY_CORNER = 0.06                 # 托盘四角的放置点（离中心各 6 cm，相邻两角相距 12 cm）
ARM_TUCK = np.array([-0.35, -0.10, 0.0, -1.70, 0.0, 0.0, 0.0])    # 与 locomotion.ARM_TUCK 相同
# 骨盆位置（站位坐标系）= 中性 + 站姿影响 + 质心补偿（对 aurora_manip_sysid 的 24 组数据拟合，残差 x 9 mm / y 2 mm）
PELVIS_X0, PELVIS_Z0 = 0.014, 0.836
COM_FIT = np.array([[0.015, 0.000], [-0.343, -0.022], [-0.203, -0.335]])   # [1, com_x, com_y] → 骨盆 (x, y)
OBS_DIM = 46


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class AuroraManipWorld:
    """GR1T2 站在桌边（PdStand），只管腰 + 右臂 + 站姿 + 抓取辅助。"""

    def __init__(self, seed=None, xml=None):
        from aurora_bridge.arm_ik import RightArmIK
        xml = xml or S.OUT_XML
        if not os.path.exists(xml):
            S.build(xml)
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.ik = RightArmIK(xml)
        self.rng = np.random.default_rng(seed)
        self.sc, self.rc = SceneCfg(), RobotCfg()
        m = self.m
        self.n_sub = int(round(CTRL_DT / m.opt.timestep))
        self.aid = {m.actuator(i).name: i for i in range(m.nu)}
        self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in S.RIGHT_ARM + S.WAIST + S.LEGS}
        self.dadr = {n: m.jnt_dofadr[m.joint(n).id] for n in S.RIGHT_ARM + S.WAIST}
        self.arm_act = [self.aid[j] for j in S.RIGHT_ARM]
        self.waist_act = [self.aid[j] for j in S.WAIST]
        self.ee_bid = m.body("right_end_effector_link").id
        self.waist_bid = m.body("waist_yaw_link").id
        self.base_mocap = m.body_mocapid[m.body("base_link").id]
        assert self.base_mocap >= 0, "base_link 必须是 mocap 物体（重新运行 python -m sim.gr1t2_manip_scene）"
        self.tray_mocap = m.body_mocapid[m.body("tray").id]
        self.goal_mocap = m.body_mocapid[m.body("goal_marker").id]
        self.obj_names = list(self.sc.objects)
        self.obj_bid = {n: m.body(n).id for n in self.obj_names}
        self.obj_gid = {n: m.geom(f"{n}_geom").id for n in self.obj_names}
        self.obj_jadr = {n: m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.obj_names}
        self.obj_vadr = {n: m.jnt_dofadr[m.joint(f"{n}_free").id] for n in self.obj_names}
        self.obj_half_h = {n: (s[1][2] if s[0] == "box" else s[1][1]) for n, s in self.sc.objects.items()}
        self.weld = {n: m.equality(f"grasp_{n}").id for n in self.obj_names}
        self.table_gid = m.geom("table_top").id
        self.tray_gid = m.geom("tray_geom").id
        self.arm_gids = {m.geom(f"{b}.STL").id for b in ("right_upper_arm_yaw_link", "right_hand_yaw_link")}
        self.body_gids = {m.geom(f"{b}.STL").id for b in ("torso_link", "head_yaw_link", "left_upper_arm_yaw_link",
                                                          "left_hand_yaw_link")}
        self.table_h = self.sc.table_height
        self.lo_arm = np.array([m.jnt_range[m.joint(j).id] for j in S.RIGHT_ARM])
        self.base_gravcomp = m.body_gravcomp.copy()
        self.base_mass = {n: float(m.body_mass[self.obj_bid[n]]) for n in self.obj_names}
        self.base_inertia = {n: m.body_inertia[self.obj_bid[n]].copy() for n in self.obj_names}
        self.base_fric = {n: m.geom_friction[self.obj_gid[n]].copy() for n in self.obj_names}

    def _set_eq(self, i, v):
        # MuJoCo ≥3.0：eq_active 在 MjData；Aurora 镜像里的 2.3.6 在 MjModel
        (self.d if hasattr(self.d, "eq_active") else self.m).eq_active[i] = v

    # ------------------------------------------------------------------ 坐标
    def to_local(self, p):
        return self.Rs.T @ (np.asarray(p) - self.ps)

    def to_world(self, p):
        return self.ps + self.Rs @ np.asarray(p)

    def vec_local(self, v):
        return self.Rs.T @ np.asarray(v)

    def ee_pos(self):
        return self.d.xpos[self.ee_bid].copy()

    def ee_vel(self):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.ee_bid, v, 0)
        return v[3:]

    def obj_pos(self, n):
        return self.d.xpos[self.obj_bid[n]].copy()

    def obj_vel(self, n):
        a = self.obj_vadr[n]
        return self.d.qvel[a:a + 3].copy()

    def tray_pos(self):
        return self.d.mocap_pos[self.tray_mocap].copy()

    def obj_tilt(self, n):
        """物体竖直轴与世界竖直方向的夹角（度）。"""
        return float(np.degrees(np.arccos(np.clip(self.d.xmat[self.obj_bid[n]][8], -1, 1))))

    def show_goal(self, p):
        self.d.mocap_pos[self.goal_mocap] = p

    def arm_q(self):
        return np.array([self.d.qpos[self.qadr[j]] for j in S.RIGHT_ARM])

    def waist_q(self):
        return np.array([self.d.qpos[self.qadr[j]] for j in S.WAIST])

    # ------------------------------------------------------------------ 随机化 + 重置
    def randomize(self):
        r, m = self.rng, self.m
        m.body_gravcomp[:] = self.base_gravcomp * 0 + np.where(self.base_gravcomp > 0, r.uniform(0.65, 0.95), 0)
        for n in self.obj_names:
            # 质量和转动惯量要一起缩放（只改质量，惯量不变 → 物体变成“细长杆”那样的病态惯量，
            # 新版 MuJoCo（3.3 之后）里 weld 抓住时会数值爆炸，物体飞出几公里）
            k = r.uniform(0.7, 2.0)
            b = self.obj_bid[n]
            m.body_mass[b] = self.base_mass[n] * k
            m.body_inertia[b] = self.base_inertia[n] * k
            m.geom_friction[self.obj_gid[n]][0] = self.base_fric[n][0] * r.uniform(0.7, 1.2)
        if hasattr(mujoco, "mj_setConst"):
            mujoco.mj_setConst(m, self.d)          # 重新计算依赖质量的常量（invweight 等，约束的软硬按它算）
        self.delay_sub = min(int(round(r.uniform(0.02, 0.045) / m.opt.timestep)), self.n_sub - 1)   # 指令延迟（< 一个控制周期）
        self.q_bias = r.normal(0, 0.008, 7)                     # 关节静差（摩擦、标定）
        self.pelvis_bias = r.normal(0, [0.005, 0.004, 0.003])   # 骨盆位姿偏差
        self.com_gain = r.uniform(0.7, 1.3)                     # 质心补偿强度

    def reset(self, stance, randomize=True):
        """stance = (x, y, yaw)：双脚中点 / 朝向（= 导航停靠的站位）。"""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        if randomize:
            self.randomize()
        else:
            self.delay_sub = min(int(round(0.03 / m.opt.timestep)), self.n_sub - 1)
            self.q_bias, self.pelvis_bias, self.com_gain = np.zeros(7), np.zeros(3), 1.0
        for n in self.obj_names:
            m.geom_conaffinity[self.obj_gid[n]] = 7
            self._set_eq(self.weld[n], 0)
        self.held = None
        self.up_axis = None
        self.grip = 0.0
        self.ps = np.array([stance[0], stance[1], 0.0])
        self.Rs = _rz(stance[2])
        self.syaw = float(stance[2])
        self.stand = np.zeros(3)
        self.pelvis_xy = np.array([PELVIS_X0, 0.0])
        for j in S.LEGS:
            k = [kk for kk in S.LEG_STAND if kk in j]
            v = S.LEG_STAND[k[0]] if k else 0.0
            d.qpos[self.qadr[j]] = v
            d.ctrl[self.aid[j]] = v
        self.set_arm(np.zeros(7), np.zeros(3), instant=True)
        self.settle_pelvis()

    def settle_pelvis(self):
        """把骨盆直接放到当前手臂姿态对应的稳态位置（重置 / 瞬移手臂之后调用）。"""
        for _ in range(3):
            self._place_pelvis(1.0)
            mujoco.mj_forward(self.m, self.d)

    def set_arm(self, q_arm, q_waist, instant=False):
        q_arm = np.clip(q_arm, self.lo_arm[:, 0], self.lo_arm[:, 1])
        self.q_cmd = np.concatenate([q_arm, q_waist])
        if instant:
            for j, v in zip(S.RIGHT_ARM, q_arm):
                self.d.qpos[self.qadr[j]] = v
            for j, v in zip(S.WAIST, q_waist):
                self.d.qpos[self.qadr[j]] = v
            self.d.ctrl[self.arm_act] = q_arm
            self.d.ctrl[self.waist_act] = q_waist
            self._pending = None
            mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------------ 骨盆（mocap）
    def _place_pelvis(self, alpha):
        """按站姿 + 质心补偿摆骨盆。alpha：质心补偿的低通系数（1 = 立即到位）。"""
        d = self.d
        dz, dp, dy = self.stand
        com = self.vec_local(d.subtree_com[self.waist_bid] - d.mocap_pos[self.base_mocap])   # 上身质心相对骨盆
        tgt = COM_FIT[0] + self.com_gain * (com[0] * COM_FIT[1] + com[1] * COM_FIT[2])
        self.pelvis_xy = (1 - alpha) * self.pelvis_xy + alpha * tgt
        p_loc = np.array([self.pelvis_xy[0] + self.pelvis_bias[0], self.pelvis_xy[1] + self.pelvis_bias[1],
                          PELVIS_Z0 + dz + self.pelvis_bias[2]])
        d.mocap_pos[self.base_mocap] = self.to_world(p_loc)
        yaw = self.syaw + dy
        cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(dp / 2), np.sin(dp / 2)
        d.mocap_quat[self.base_mocap] = [cy * cp, -sy * sp, cy * sp, sy * cp]     # Rz(yaw)·Ry(pitch)

    # ------------------------------------------------------------------ IK（与部署相同的求解器）
    def solve_ik(self, target_world, iters=25):
        ik = self.ik
        ik.d.qpos[:] = self.d.qpos
        ik.d.mocap_pos[:] = self.d.mocap_pos
        ik.d.mocap_quat[:] = self.d.mocap_quat
        up = self.up_axis if self.held is not None else None
        q, waist, err = ik.solve(target_world, iters=iters, up_axis=up)
        return np.asarray(q), np.asarray(waist), err

    # ------------------------------------------------------------------ 仿真一步（0.05 s）
    def step(self, stand_rate, grip_cmd):
        m, d = self.m, self.d
        self.stand = np.clip(self.stand + np.clip(stand_rate, -1, 1) * STAND_RATE * CTRL_DT,
                             STAND_LIM[:, 0], STAND_LIM[:, 1])
        self.grip = float(np.clip(self.grip + (0.25 if grip_cmd > 0 else -0.25), 0, 1))
        new_arm = self.q_cmd[:7] + self.q_bias
        p0, q0 = d.mocap_pos[self.base_mocap].copy(), d.mocap_quat[self.base_mocap].copy()
        self._place_pelvis(CTRL_DT / 0.3)
        p1, q1 = d.mocap_pos[self.base_mocap].copy(), d.mocap_quat[self.base_mocap].copy()
        for k in range(self.n_sub):
            if k == self.delay_sub:
                d.ctrl[self.arm_act] = new_arm
                d.ctrl[self.waist_act] = self.q_cmd[7:]
            a = (k + 1) / self.n_sub
            d.mocap_pos[self.base_mocap] = (1 - a) * p0 + a * p1
            qq = (1 - a) * q0 + a * q1
            d.mocap_quat[self.base_mocap] = qq / np.linalg.norm(qq)
            mujoco.mj_step(m, d)
        self._update_grasp()

    # ------------------------------------------------------------------ 抓取辅助
    def grasp_point(self):
        return self.ee_pos() - np.array([0, 0, GRASP_ABOVE])

    def _update_grasp(self):
        d, m = self.d, self.m
        if self.held is None and self.grip > 0.6:
            gp = self.grasp_point()
            best, bd = None, GRASP_RADIUS
            for n in self.obj_names:
                dist = np.linalg.norm(self.obj_pos(n) - gp)
                if dist < bd:
                    best, bd = n, dist
            if best is not None:
                self.attach(best)
        elif self.held is not None and self.grip < 0.3:
            self._set_eq(self.weld[self.held], 0)
            self._released = self.held
            self.held = None
        # 松手后手离开物体 > 6 cm 再恢复手-物体碰撞
        r = getattr(self, "_released", None)
        if r is not None and np.linalg.norm(self.obj_pos(r) - self.grasp_point()) > 0.06:
            m.geom_conaffinity[self.obj_gid[r]] = 7
            self._released = None

    def attach(self, n):
        m, d = self.m, self.d
        e = self.weld[n]
        p1, R1 = d.xpos[self.ee_bid], d.xmat[self.ee_bid].reshape(3, 3)
        rel_p = R1.T @ (d.xpos[self.obj_bid[n]] - p1)
        q1_inv = np.zeros(4)
        mujoco.mju_negQuat(q1_inv, d.xquat[self.ee_bid])
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, q1_inv, d.xquat[self.obj_bid[n]])
        m.eq_data[e, 0:3] = 0.0
        m.eq_data[e, 3:6] = rel_p
        m.eq_data[e, 6:10] = rel_q
        m.eq_data[e, 10] = 1.0
        self._set_eq(e, 1)
        m.geom_conaffinity[self.obj_gid[n]] = 3          # 抓住后不和机器人碰撞
        self.held = n
        # 物体的竖直轴在手坐标系里的方向：端着时 IK 让它保持竖直（物体不歪，放下时才立得住）
        self.up_axis = R1.T @ d.xmat[self.obj_bid[n]].reshape(3, 3)[:, 2]

    # ------------------------------------------------------------------ 场景摆放
    def place_object(self, n, xy, yaw=0.0, z=None):
        a = self.obj_jadr[n]
        z = self.table_h + self.obj_half_h[n] + 0.001 if z is None else z
        self.d.qpos[a:a + 3] = [xy[0], xy[1], z]
        self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        v = self.obj_vadr[n]
        self.d.qvel[v:v + 6] = 0

    def place_tray(self, xy):
        self.d.mocap_pos[self.tray_mocap] = [xy[0], xy[1], self.table_h + self.sc.tray_half[2]]

    def on_table(self, xy, margin=0.05):
        tx, ty = self.sc.table_pos
        hx, hy = self.sc.table_half
        return abs(xy[0] - tx) <= hx - margin and abs(xy[1] - ty) <= hy - margin

    def contacts(self):
        """(手臂碰桌/托盘, 躯干/头/左臂碰桌)"""
        arm = body = False
        statics = {self.table_gid, self.tray_gid}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g = {c.geom1, c.geom2}
            if g & statics:
                arm |= bool(g & self.arm_gids)
                body |= bool(g & self.body_gids)
        return arm, body


# ============================================================================
class AuroraManipSkill:
    """观测 / 动作 / 奖励（与环境解耦，部署时也用这一份来构造观测）。"""

    def __init__(self, world: AuroraManipWorld, obs_noise=True):
        self.w = world
        self.obs_noise = obs_noise
        self.n_act = 7
        self.target = world.obj_names[0]
        self.goal_world = np.zeros(3)
        self.goal_local = None
        self.release = True

    def set_task(self, target, goal_world=None, goal_local=None, release=True):
        self.target = target
        self.goal_local = None if goal_local is None else np.asarray(goal_local, float)
        if goal_world is not None:
            self.goal_world = np.asarray(goal_world, float)
        if self.goal_local is not None:
            self.goal_world = self.w.to_world(self.goal_local)
        self.release = release
        self.w.show_goal(self.goal_world)

    def tray_goal(self, name=None, tray_xyz=None):
        w = self.w
        tp = w.tray_pos() if tray_xyz is None else np.asarray(tray_xyz)
        return np.array([tp[0], tp[1], w.table_h + 2 * w.sc.tray_half[2] + w.obj_half_h[name or self.target]])

    def begin(self, obj_estimate=None):
        w = self.w
        self.prev_action = np.zeros(self.n_act)
        self.ee_target = w.ee_pos()
        self.obs_bias = w.rng.normal(0, 0.008, 3) if self.obs_noise else np.zeros(3)
        if obj_estimate is not None:
            self.obs_bias = np.asarray(obj_estimate) - w.obj_pos(self.target)
        self.stable = 0
        self.phi = self.potential()
        self.others0 = {n: w.obj_pos(n) for n in w.obj_names if n != self.target}
        self.knocked = False

    def state(self):
        w = self.w
        gp = w.grasp_point()
        op = w.obj_pos(self.target)
        d_ho = np.linalg.norm(op - gp)
        d_og = np.linalg.norm(op - self.goal_world)
        d_og_xy = np.linalg.norm(op[:2] - self.goal_world[:2])
        held = w.held == self.target
        tilt = w.obj_tilt(self.target)
        placed = (not held) and self.release and d_og_xy < 0.04 and \
            abs(op[2] - self.goal_world[2]) < 0.015 and tilt < 10.0
        return gp, op, d_ho, d_og, held, placed

    def potential(self):
        _, _, d_ho, d_og, held, placed = self.state()
        have = held or placed
        close = lambda d: 1.0 - 0.5 * (np.tanh(3.0 * d) + np.tanh(10.0 * d))  # noqa: E731
        return (1.0 if have else close(d_ho)) + 2.0 * have + 3.0 * (close(d_og) if have else 0.0)

    def obs(self):
        w = self.w
        gp, op, _, _, held, _ = self.state()
        noise = self.obs_bias + w.rng.normal(0, 0.002, 3) if (self.obs_noise and not held) else np.zeros(3)
        ee = w.to_local(w.ee_pos())
        obj = w.to_local(op + noise)
        goal = w.to_local(self.goal_world)
        q = w.arm_q()
        qn = 2 * (q - w.lo_arm[:, 0]) / (w.lo_arm[:, 1] - w.lo_arm[:, 0]) - 1
        wq = w.waist_q()[:2] / np.array([0.4, 0.35])
        sp = (w.stand - STAND_LIM[:, 0]) / (STAND_LIM[:, 1] - STAND_LIM[:, 0]) * 2 - 1
        ee_v, obj_v = w.vec_local(w.ee_vel()), w.vec_local(w.obj_vel(self.target))
        if self.obs_noise:      # 部署时速度是 20 Hz 差分出来的（关节编码器 / 感知），有噪声
            ee_v = ee_v + w.rng.normal(0, 0.03, 3)
            obj_v = obj_v + w.rng.normal(0, 0.03, 3)
        o = np.concatenate([
            ee, ee_v, w.vec_local(self.ee_target - w.ee_pos()) * 10.0,
            [w.grip], [float(held)],
            obj, obj - (ee - [0, 0, GRASP_ABOVE]), obj_v,
            goal, goal - obj, [float(self.release)],
            qn, wq, sp, self.prev_action])
        return o.astype(np.float32)

    def act(self, action):
        w = self.w
        a = np.clip(np.asarray(action, float), -1, 1)
        ee = w.ee_pos()
        tgt = self.ee_target + w.Rs @ (a[:3] * EE_STEP)
        self.ee_target = ee + np.clip(tgt - ee, -0.05, 0.05)       # 目标别离手太远（被挡住时不“跑飞”）
        q, waist, _ = w.solve_ik(self.ee_target)
        w.set_arm(q, waist)
        w.step(a[4:7], a[3])
        self._a = a

    def evaluate(self):
        w = self.w
        a = self._a
        if self.goal_local is not None:
            self.goal_world = w.to_world(self.goal_local)
        phi = self.potential()
        r = 2.0 * (phi - self.phi)
        self.phi = phi
        arm_hit, body_hit = w.contacts()
        r -= 0.05 * arm_hit + 0.2 * body_hit + 0.005 * float(np.sum(a ** 2))
        self.prev_action = a
        _, op, d_ho, d_og, held, placed = self.state()
        speed = np.linalg.norm(w.obj_vel(self.target))
        if self.release:
            self.stable = self.stable + 1 if (placed and speed < 0.05 and w.grip < 0.3) else 0
            success = self.stable >= 4
        else:
            success = held and d_og < 0.05
        dropped = op[2] < w.table_h - 0.05
        # 碰倒 / 推开别的物体（托盘上已放好的、桌上其他的）：一次性扣 3 分
        if not self.knocked and any(np.linalg.norm(w.obj_pos(n) - p0) > 0.03 or w.obj_tilt(n) > 20
                                    for n, p0 in self.others0.items()):
            self.knocked = True
            r -= 3.0
        r += (10.0 if success else 0.0) - (5.0 if dropped else 0.0)
        info = {"success": bool(success), "held": bool(held), "d_goal": float(d_og), "d_hand": float(d_ho),
                "dropped": bool(dropped), "arm_hit": arm_hit, "body_hit": body_hit, "knocked": self.knocked}
        return r, bool(success), bool(dropped), info


# ============================================================================
def carry_point_local(world: AuroraManipWorld):
    """端持姿态 ARM_TUCK、站姿为 0 时，“物体中心”在站位坐标系下的位置（= 模式 B 的目标）。"""
    w = world
    ik = w.ik
    ik.d.qpos[:] = w.d.qpos
    ik.d.mocap_pos[:] = w.d.mocap_pos
    ik.d.mocap_quat[:] = w.d.mocap_quat
    for j, v in zip(S.RIGHT_ARM, ARM_TUCK):
        ik.d.qpos[ik.qadr[j]] = v
    for j in S.WAIST:
        ik.d.qpos[ik.qadr[j]] = 0.0
    mujoco.mj_kinematics(ik.m, ik.d)
    return w.to_local(ik.d.xpos[ik.ee]) - np.array([0, 0, GRASP_ABOVE])


class AuroraManipEnv(gym.Env):
    """训练环境：机器人已停在桌边站位（带 ±4 cm / ±0.08 rad 导航误差），PdStand 站着。"""
    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_steps=300, mode_probs=(0.5, 0.25, 0.25), obs_noise=True, randomize=True):
        self.world = AuroraManipWorld(seed=seed)
        self.skill = AuroraManipSkill(self.world, obs_noise)
        self.geo = TableGeometry(self.world.sc, self.world.rc)
        self.max_steps = max_steps
        self.mode_probs = np.array(mode_probs, float) / np.sum(mode_probs)
        self.randomize = randomize
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    def _sample_reachable_xy(self, pose, avoid=(), min_dist=0.12, margin=0.06):
        w = self.world
        for _ in range(200):
            fwd = self.geo.standoff + w.rng.uniform(0.05, REACH_DEPTH - 0.02)
            lat = w.rng.uniform(LAT_MIN + 0.02, LAT_MAX - 0.02)
            c, s = np.cos(pose[2]), np.sin(pose[2])
            xy = pose[:2] + np.array([c * fwd - s * lat, s * fwd + c * lat])
            if w.on_table(xy, margin) and all(np.linalg.norm(xy - a) > min_dist for a in avoid):
                return xy
        return None

    def _scatter(self, fixed):
        w = self.world
        placed = list(fixed.values())
        for n in w.obj_names + ["__tray__"]:
            if n in fixed:
                xy = fixed[n]
            else:
                for _ in range(200):
                    tx, ty = w.sc.table_pos
                    hx, hy = w.sc.table_half
                    m = w.sc.tray_half[0] + 0.01 if n == "__tray__" else 0.07
                    xy = np.array([tx + w.rng.uniform(-hx + m, hx - m), ty + w.rng.uniform(-hy + m, hy - m)])
                    if all(np.linalg.norm(xy - p) > (0.14 + (w.sc.tray_half[0] - 0.02 if n == "__tray__" else 0)) for p in placed):
                        break
                placed.append(xy)
            if n == "__tray__":
                w.place_tray(xy)
            else:
                w.place_object(n, xy, yaw=w.rng.uniform(-np.pi, np.pi))

    def reset(self, seed=None, options=None):
        w, sk = self.world, self.skill
        if seed is not None:
            w.rng = np.random.default_rng(seed)
        options = options or {}
        mode = int(options.get("mode", w.rng.choice(3, p=self.mode_probs)))
        for _ in range(50):
            tx, ty = w.sc.table_pos
            hx, hy = w.sc.table_half
            anchor = np.array([tx + w.rng.uniform(-hx + 0.07, hx - 0.07), ty + w.rng.uniform(-hy + 0.07, hy - 0.07)])
            sides = [s for s, *_ in self.geo.sides() if self.geo.stand_pose_for(anchor, s)[1]]
            if not sides:
                continue
            pose, _, _ = self.geo.stand_pose_for(anchor, w.rng.choice(sides))
            p = pose.as_array() + w.rng.normal(0, [0.03, 0.03, 0.06])
            obj_xy = self._sample_reachable_xy(p)
            th = w.sc.tray_half[0]
            tray_xy = self._sample_reachable_xy(p, avoid=[obj_xy], min_dist=th + 0.08, margin=th + 0.01) \
                if obj_xy is not None else None
            if obj_xy is not None and (tray_xy is not None or mode == MODE_PICK_HOLD):
                break
        w.reset(p, randomize=self.randomize)
        target = w.obj_names[int(w.rng.integers(len(w.obj_names)))]
        fixed = {target: obj_xy}
        if tray_xy is not None:
            fixed["__tray__"] = tray_xy
        self._scatter(fixed)
        # 托盘上已经有物体的情况（连续任务里放第 2、3 个物体时就是这样）：一半回合往托盘四角放 1–2 个别的物体，
        # 目标改成空着的角；托盘空着时一半回合目标在中心、一半在随机一角
        tray_off = np.zeros(2)
        if tray_xy is not None:
            corners = [np.array([sx, sy]) * TRAY_CORNER for sx in (-1, 1) for sy in (-1, 1)]
            def reach(xy):    # 站位系下在手的可达范围内
                c_, s_ = np.cos(p[2]), np.sin(p[2])
                d = xy - p[:2]
                fwd, lat = c_ * d[0] + s_ * d[1], -s_ * d[0] + c_ * d[1]
                return self.geo.standoff <= fwd <= self.geo.standoff + REACH_DEPTH + 0.02 and LAT_MIN - 0.02 <= lat <= LAT_MAX + 0.02
            corners = [c for c in corners if w.on_table(tray_xy + c, 0.05) and reach(tray_xy + c)]
            w.rng.shuffle(corners)
            others = [n for n in w.obj_names if n != target]
            k = min(int(w.rng.choice([0, 1, 2], p=[0.4, 0.3, 0.3])), len(corners) - 1)
            for n, c in zip(others[:k], corners[:k]):
                w.place_object(n, tray_xy + c, yaw=w.rng.uniform(-np.pi, np.pi),
                               z=w.table_h + 2 * w.sc.tray_half[2] + w.obj_half_h[n] + 0.001)
            if corners and (k > 0 or w.rng.random() < 0.5):
                tray_off = corners[k]
            mujoco.mj_forward(w.m, w.d)
        if mode == MODE_PLACE:
            # 物体已在手里：端持姿态 ARM_TUCK
            w.set_arm(ARM_TUCK + w.rng.normal(0, 0.03, 7), np.zeros(3), instant=True)
            ep = w.ee_pos()
            w.place_object(target, ep[:2], yaw=w.rng.uniform(-np.pi, np.pi), z=ep[2] - GRASP_ABOVE)
            mujoco.mj_forward(w.m, w.d)
            w.grip = 1.0
            w.attach(target)
        else:
            w.set_arm(w.rng.normal(0, 0.05, 7), np.zeros(3), instant=True)
        w.settle_pelvis()
        # 先让物体落稳
        for _ in range(5):
            w.step(np.zeros(3), 1.0 if mode == MODE_PLACE else -1.0)
        if mode == MODE_PICK_HOLD:
            sk.set_task(target, goal_local=carry_point_local(w), release=False)
        else:
            g = sk.tray_goal(name=target)
            g[:2] += tray_off
            sk.set_task(target, goal_world=g, release=True)
        self.mode = mode
        self.t = 0
        sk.begin()
        return sk.obs(), {"mode": mode}

    def step(self, action):
        self.skill.act(action)
        self.t += 1
        r, success, dropped, info = self.skill.evaluate()
        info["mode"] = self.mode
        return self.skill.obs(), float(r), success or dropped, self.t >= self.max_steps, info


# ============================================================================
class ScriptedManip:
    """几何脚本老师（用真值），**只看当前状态、没有内部阶段**（学生网络也没有记忆，这样才模仿得像）：
      * “漏斗”式接近：目标点上方允许的高度 = 水平距离 × 5（最高到安全高度）。离得远就在安全高度平移，
        到目标正上方附近才下降 —— 端着物体不会从托盘上已放好的物体旁边横扫过去；
      * 比漏斗低就先竖直上升（抓起后先抬起来再走）；
      * 末端到抓取点水平 1.5 cm、高度 3 cm 内才握紧；物体到目标正上方 1.2 cm 内、离托盘面 1.5 cm 内才松手；
        端着时按“手相对物体的实际偏移”去对准（物体对准目标，而不是手）；放好后竖直抬手。
    用于：检查环境是否可解、给 PPO 做行为克隆初始化、当基线。"""

    MANIP_STAND = np.array([-0.06, 0.20, 0.0])
    SLOPE = 5.0

    def __init__(self, env: AuroraManipEnv):
        self.env = env
        self.phase = "approach"

    def reset(self):
        self.phase = "approach"

    def _go(self, tgt, gain=1.0):
        w, sk = self.env.world, self.env.skill
        e = w.vec_local(tgt - sk.ee_target)
        return np.clip(e / EE_STEP * gain, -1, 1), float(np.linalg.norm(tgt - w.ee_pos()))

    def _reach_z(self, xy, z_lo, z_hi):
        """xy 正上方 [z_lo, z_hi] 里够得到的最高点（远处够不到安全高度时就降低一些）。"""
        w = self.env.world
        for z in np.linspace(z_hi, z_lo, 5):
            _, _, err = w.solve_ik(np.array([xy[0], xy[1], z]), iters=40)
            if err < 0.01:
                return float(z)
        return float(z_lo)

    def _funnel(self, p, z_safe, vertical_first):
        """去往末端目标 p：返回这一步的末端目标点。"""
        ee = self.env.world.ee_pos()
        z_safe = self._reach_z(p[:2], min(z_safe, p[2] + 0.08), z_safe)
        dxy = float(np.linalg.norm(ee[:2] - p[:2]))
        zt = p[2] + min(max(z_safe - p[2], 0.0), max(0.0, dxy - 0.01) * self.SLOPE)
        if ee[2] < zt - 0.02 and dxy > 0.02:
            # 太低：先升上去。手里有物体时纯竖直（否则低着横移会碰倒旁边的物体）；
            # 空手时水平也走 30%（手刚从身体旁边抬起来，纯竖直有时够不到会卡住）
            stuck = self.env.skill.ee_target[2] > zt - 0.01      # 已经在往上要了，但升不上去（够不到）
            xy = ee[:2] + (0.0 if vertical_first and not stuck else 0.3) * (p[:2] - ee[:2])
            return np.array([xy[0], xy[1], zt])
        return np.array([p[0], p[1], zt])

    def __call__(self, obs=None):
        env = self.env
        w, sk = env.world, env.skill
        held = w.held == sk.target
        op = w.obj_pos(sk.target) + (sk.obs_bias if not held else 0)
        goal = sk.goal_world
        up = np.array([0, 0, GRASP_ABOVE])
        ee = w.ee_pos()
        z_safe = w.table_h + GRASP_ABOVE + 0.23        # 手里物体底面比 12 cm 高的物体再高 5 cm
        stand_tgt = self.MANIP_STAND.copy()
        focus = goal if held else op
        fl = w.to_local(focus)
        stand_tgt[2] = np.clip(np.arctan2(fl[1] + 0.15, fl[0]) * 0.8, -0.4, 0.4) if fl[0] > 0.2 else 0.0
        off = (ee - op) if held else up                 # 手相对物体的实际偏移（抓的位置不一定正好在正上方 7 cm）
        if held and not sk.release:                     # 抓起端持：抬起来送到端持点
            tgt = self._funnel(goal + off, z_safe, True)
            grip, self.phase = 1.0, "carry"
            stand_tgt = np.zeros(3)
            a, _ = self._go(tgt)
        elif held:                                      # 去放：漏斗下降到目标上方 0.5 cm，到了就松手
            place = goal + off + [0, 0, 0.005]
            tgt = self._funnel(place, z_safe, True)
            # 按物体判断：物体到了目标正上方 1.2 cm 内、离托盘面 <1.5 cm 才松手
            near = np.linalg.norm(op[:2] - goal[:2]) < 0.012 and op[2] - goal[2] < 0.015
            grip, self.phase = (-1.0, "release") if near else (1.0, "carry")
            a, _ = self._go(tgt, 0.6 if np.linalg.norm(ee[:2] - place[:2]) < 0.03 else 1.0)
        else:
            placed = sk.release and np.linalg.norm(op[:2] - goal[:2]) < 0.04 and abs(op[2] - goal[2]) < 0.015
            if placed:                                  # 放好了：竖直抬手
                tgt = np.array([ee[0], ee[1], z_safe])
                grip, self.phase = -1.0, "retreat"
                a, _ = self._go(tgt)
            else:                                       # 去抓：漏斗下降到物体正上方 7 cm，到了就握紧
                g = op + up
                tgt = self._funnel(g, z_safe, False)
                near = np.linalg.norm(ee[:2] - g[:2]) < 0.015 and abs(ee[2] - g[2]) < 0.03
                grip, self.phase = (1.0, "grasp") if near else (-1.0, "approach")
                a, _ = self._go(tgt, 0.6 if np.linalg.norm(ee[:2] - g[:2]) < 0.03 else 1.0)
        s_rate = np.clip((stand_tgt - w.stand) / (STAND_RATE * CTRL_DT), -1, 1)
        return np.concatenate([a, [grip], s_rate]).astype(np.float32)
