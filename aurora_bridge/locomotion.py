"""
aurora_bridge/locomotion.py —— 用 Aurora SDK 控制 GR1T2 的“走 / 站 / 操作姿态”
==============================================================================
项目原来的做法是：骨盆挂在 3 个滑动关节上（平面底盘），腿只是动画。这在仿真里能跑，
但有几个和真机完全对不上的姿态问题（见 docs/aurora_posture_issues.md）。
这里把“下半身”整个交给 Aurora：

    FSM 2 PdStand          站立平衡（全身模型控制）+ 上身关节指令 + 站姿 Δz/Δpitch/Δyaw
                           → 用来“操作”（伸手、俯身、降高度）
    FSM 3 UserController_A LowerBodyCpgRLTask（CPG+RL 行走，管腿和腰）+ 上身状态机
                           → 用来“走路”（速度指令 vx, vy, wz）；上身状态 2 = 远程控制，可端持物体

AuroraLoco 封装了实测下来**必须遵守**的几条规则（每条都在仿真里摔过跤）：
  1. 进 PdStand 后立刻把腰和手臂回正。零力矩状态下腰会在重力下前倾到 0.4–1.2 rad，
     PdStand 会“冻结”进入时的上身姿态 → 躯干前倾 → 站立判定 isStanceState 一碰就失败 →
     切到 Hanging（屈膝悬挂）→ 摔倒。JointStand 只使能腿，不能解决这个问题。
  2. 吊绳 / 龙门吊要慢慢放，直到脚完全承重再摘（仿真里用 'lower'）。
  3. 站姿 Δz/Δpitch/Δyaw、上身关节指令都要按速率斜坡发送，不能跳变。
  4. Aurora v1.2.1 的速度指令没有超时：停车必须显式发 0。
  5. |v_xy|≤0.1 且 |wz|≤0.1（乘 0.9 安全系数后）持续 1.35 s → 停止迈步。想“慢慢挪”
     是不行的：要么 ≥0.12，要么 0。
  6. 从走切到站（3→2）前要先停稳（等停止迈步）；从站切到走（2→3）前站姿要回到 0、
     停 ≥1 s。端持物体用贴身姿态（TUCK），手离身体越远，切换时的晃动越大。

坐标：位姿默认取 true_data（仿真真值；真机上请换成外部定位，Aurora 自带估计器的位置
会漂，实测 ~0.2 m），朝向取 base_data 的 IMU 姿态。
"""
import time

import numpy as np

from aurora_bridge.gait_model import (MIN_EFFECTIVE, VX_RANGE, VY_RANGE, WZ_RANGE,
                                      stopping_distance)

ARM_ZERO = [0.0] * 7
# 端持姿态（右臂，7 个关节：肩 pitch/roll/yaw、肘 pitch、腕 yaw/roll/pitch）
ARM_TUCK = [-0.35, -0.10, 0.0, -1.70, 0.0, 0.0, 0.0]     # 手在身前 0.31 m、高 1.2 m（推荐）
ARM_CARRY = [-0.60, -0.10, 0.0, -1.20, 0.0, 0.0, 0.0]    # 手在身前 0.37 m
STAND_POSE_RATE = np.array([0.05, 0.15, 0.15])           # Δz m/s, Δpitch rad/s, Δyaw rad/s
STAND_POSE_LIMITS = np.array([[-0.20, 0.02], [-0.30, 0.45], [-0.50, 0.50]])
ACC_LIMIT = np.array([0.6, 0.6, 1.5])                    # 指令变化率限制 (m/s², rad/s²)


class AuroraLoco:
    FSM_DEFAULT, FSM_JOINTSTAND, FSM_PDSTAND, FSM_WALK = 0, 1, 2, 3

    def __init__(self, client, sim_ctrl=None, pose_source="truth", rate_hz=20.0, log=print):
        """client: aurora_compat.AuroraClient；sim_ctrl: 发送仿真控制字符串的函数（真机为 None）。"""
        self.c = client
        self.sim_ctrl = sim_ctrl
        self.pose_source = pose_source
        self.dt = 1.0 / rate_hz
        self.log = log
        self.cmd = np.zeros(3)
        self.stand_pose = np.zeros(3)
        self.upper_hold = {}              # 每个周期都会重发的上身关节目标（组名 → 关节角）
        self._pose_active = False
        self.mode = "default"

    # ------------------------------------------------------------------ 状态
    def pose(self):
        """(x, y, yaw) 世界系。"""
        if self.pose_source == "truth" and "pos_W" in self.c._true_data:
            p = self.c.get_true_data("pos_W")
        else:
            p = self.c.get_base_data("pos_W")
        yaw = self.c.get_base_data("rpy")[2]
        return np.array([p[0], p[1], yaw], float)

    def vel_body(self):
        if self.pose_source == "truth" and "vel_B" in self.c._true_data:
            v = self.c.get_true_data("vel_B")
        else:
            v = self.c.get_base_data("vel_B")
        w = self.c.get_base_data("omega_B")[2]
        return np.array([v[0], v[1], w], float)

    def stable_level(self):
        return self.c.get_stand_pose()[3]

    # ------------------------------------------------------------------ 底层：周期发送
    def _tick(self):
        if self.upper_hold:
            self.c.set_group_cmd({g: list(v) for g, v in self.upper_hold.items()})
        if self.mode == "walk":
            self.c.set_velocity(*self.cmd)
        elif self.mode == "stand" and self._pose_active:
            # 只在调整站姿时发送；站姿一直为 0 时不发（实测持续发 0 会干扰 PdStand 的站立判定）
            self.c.set_stand_pose(*self.stand_pose)

    def wait(self, dur, fn=None):
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= dur:
                break
            if fn is not None and fn(t) is True:
                break
            self._tick()
            time.sleep(self.dt)

    def ramp_upper(self, targets, T=2.0, settle=0.3):
        """上身关节组从当前位置线性插值到目标（PdStand 或 上身远程控制状态下有效）。"""
        start = {}
        for g in targets:
            try:
                start[g] = np.array(self.c.get_group_state(g, "position"))
            except KeyError:
                start[g] = np.array(targets[g], float)
        goal = {g: np.array(v, float) for g, v in targets.items()}

        def f(t):
            a = min(1.0, t / T)
            for g in goal:
                self.upper_hold[g] = (1 - a) * start[g] + a * goal[g]
        self.wait(T + settle, f)
        self.upper_hold.update(goal)

    def ramp_stand_pose(self, target, settle=1.0):
        """站姿增量（仅 PdStand）：按 STAND_POSE_RATE 限速斜坡。"""
        target = np.clip(np.asarray(target, float), STAND_POSE_LIMITS[:, 0], STAND_POSE_LIMITS[:, 1])
        p0 = self.stand_pose.copy()
        T = max(float(np.max(np.abs(target - p0) / STAND_POSE_RATE)), 0.1)

        def f(t):
            self.stand_pose = p0 + min(1.0, t / T) * (target - p0)
        self._pose_active = True
        self.wait(T + settle, f)
        self.stand_pose = target
        self._pose_active = bool(np.any(np.abs(target) > 1e-4))

    def _knees(self):
        try:
            return (round(self.c.get_group_state("left_leg")[3], 2), round(self.c.get_group_state("right_leg")[3], 2))
        except (KeyError, IndexError):
            return None

    def _legs_symmetric(self, tol=0.08):
        try:
            l, r = np.array(self.c.get_group_state("left_leg")), np.array(self.c.get_group_state("right_leg"))
        except KeyError:
            return True
        # 左右腿的 pitch 关节（髋 pitch、膝、踝 pitch）应基本相同
        return bool(np.all(np.abs(l[[2, 3, 4]] - r[[2, 3, 4]]) < tol))

    # ------------------------------------------------------------------ 启动与模式切换
    def startup(self, lower_timeout=12.0, warmup=True):
        """吊着开机 → 站稳。真机上 sim_ctrl=None：本函数在“放龙门吊”那一步等操作员。
        warmup：还吊着的时候先进出一次行走状态。AuroraCore 每次进 FSM3 都会重新加载行走策略
        (gr1t2_mix.pt)；冷启动（磁盘缓存是空的）时实测加载用了 2.7 s，这段时间腿没有控制器，
        已经站在地上的机器人会直接瘫倒。吊着时预热一次，之后的加载只要几十毫秒。"""
        self.log("[loco] PdStand + 上身回正")
        self.c.set_fsm_state(self.FSM_PDSTAND)
        self.mode = "stand"
        self.stand_pose[:] = 0
        time.sleep(0.3)
        self.ramp_upper({"waist": [0, 0, 0]}, T=2.0)
        if warmup:
            self.log("[loco] 吊着预热行走策略（进出一次 FSM3）")
            hold, self.upper_hold = self.upper_hold, {}
            self.c.set_fsm_state(self.FSM_WALK)
            time.sleep(4.0)
            self.c.set_fsm_state(self.FSM_PDSTAND)
            self.upper_hold = hold
            self.ramp_upper({"waist": [0, 0, 0]}, T=1.0)
            # 行走策略在空中踢过腿，PdStand 的悬挂态要用 hanging_slerp_time=3 s 把双腿拉回对称的屈膝姿态。
            # 没等它做完就放吊绳，会一条腿伸直、一条腿弯着落地 → 单腿受力摔倒（实测）。
            self.wait(10.0, lambda t: t > 3.5 and self._legs_symmetric())
            self.log(f"[loco] 双腿已对称（膝 {self._knees()}）")
        if self.sim_ctrl is not None:
            self.log("[loco] 慢放吊绳，直到双脚完全承重")
            self.sim_ctrl("lower")
            self.wait(lower_timeout, lambda t: t > 9.0 and self.stable_level() >= 15)
        else:
            input("请缓慢放下龙门吊，直到机器人双脚完全承重、吊绳松弛，然后按回车 ...")
        self.wait(1.0)
        self.log(f"[loco] 站稳，stable_level={self.stable_level():.0f}")

    def walk_mode(self, carry=None, settle=1.0):
        """PdStand → 行走。carry：端持时右臂姿态（None = 上身交给 Aurora 默认状态）。"""
        if self.mode == "walk":
            return
        if np.any(np.abs(self.stand_pose) > 1e-3):
            self.ramp_stand_pose([0, 0, 0])
        self.ramp_upper({"waist": [0, 0, 0]}, T=0.5, settle=settle)
        self.upper_hold.pop("waist", None)         # 行走时腰归行走策略管
        self.c.set_fsm_state(self.FSM_WALK)
        self.mode = "walk"
        self.cmd[:] = 0
        self.wait(1.0)
        if carry is not None:
            self.c.set_upper_fsm_state(2)             # 上身远程控制：接收手臂指令
            self.upper_hold = {"right_manipulator": np.array(carry, float)}
        else:
            self.upper_hold = {}
        self.c.set_velocity_source(2)                 # 速度来自 DDS（导航）
        self.wait(1.0)
        self.log("[loco] 行走模式")

    def stand_mode(self):
        """行走 → PdStand（准备操作）：先停稳，等停止迈步，再切换。"""
        if self.mode == "stand":
            return
        self.stop(timeout=2.5)                     # 不必等踏步完全停下：PdStand 接管后会立即站住
        self.c.set_fsm_state(self.FSM_PDSTAND)
        self.mode = "stand"
        self.stand_pose[:] = 0
        self.wait(1.0)
        hold = {"waist": [0, 0, 0]}
        for g in ("right_manipulator", "left_manipulator"):
            try:
                hold[g] = self.c.get_group_state(g, "position")
            except KeyError:
                pass
        self.ramp_upper(hold, T=0.5)
        self.log("[loco] 站立/操作模式")

    # ------------------------------------------------------------------ 速度指令
    def command(self, vx, vy, wz, forward_only=False):
        """写入下一周期的速度指令：限幅 + 变化率限制。真正发送在 _tick() 里（与 wait() 同频）。"""
        tgt = np.array([np.clip(vx, VX_RANGE[0], VX_RANGE[1]), np.clip(vy, *VY_RANGE),
                        np.clip(wz, *WZ_RANGE)])
        if forward_only:
            tgt[0] = max(0.0, tgt[0])
        step = ACC_LIMIT * self.dt
        self.cmd = self.cmd + np.clip(tgt - self.cmd, -step, step)

    def stop(self, timeout=4.0):
        """发 0 并等到停止迈步（速度很小且超过 Aurora 的 1.35 s 停步延时）。"""
        self.cmd[:] = 0
        t_still = [None]

        def f(t):
            v = self.vel_body()
            if np.hypot(v[0], v[1]) < 0.04 and abs(v[2]) < 0.08:
                t_still[0] = t_still[0] if t_still[0] is not None else t
                return t - t_still[0] > 1.5
            t_still[0] = None
            return False
        self.wait(timeout, f)


# ======================================================================
# 导航执行器：把“导航技能”（PPO 策略或几何跟踪器）接到 Aurora 行走上
# ======================================================================
class ForwardFirstFollower:
    """几何基线 / 部署兜底控制器（不倒着走）。
    走法和人一样：
      path      沿绕桌最短路径走到“预停靠点”（站位后方 0.35 m）。航向误差大就先原地转，再前进
      turn      在预停靠点原地转到站位朝向（按当前角速度预测停转后的滑转，提前松指令）
      approach  正对桌子直线前进 0.35 m，按实测刹车距离提前停（只前进）
      trim      剩下的几厘米：横向小步 + 前进小步 + 最后修朝向；只有冲过头 >4 cm 才允许后退一小步
    参数来自 Aurora 实测：最小有效指令 0.12、刹车距离 0.08+0.43v、转停后多转 ~0.5·ω。"""
    PRE_DOCK = 0.35
    MAX_TRIMS = 4

    def __init__(self, geo, pos_tol=0.06, yaw_tol=0.07):
        self.geo, self.pos_tol, self.yaw_tol = geo, pos_tol, yaw_tol
        self.set_goal(0, 0, 0)

    def set_goal(self, x, y, yaw):
        self.goal = np.array([x, y, yaw], float)
        self.pre = self.goal[:2] - self.PRE_DOCK * np.array([np.cos(yaw), np.sin(yaw)])
        if self.geo.inside_inflated(self.pre):
            self.pre = self.geo.project_out(self.pre)[0]
        self.phase = "path"
        self.trims, self._pulse_left, self._pulse_cmd, self._calm = 0, 0, np.zeros(3), 0
        self.retries = 0

    @staticmethod
    def _still(vel):
        return np.hypot(vel[0], vel[1]) < 0.04 and abs(vel[2]) < 0.05

    def _turn_to(self, e_yaw, vel):
        """原地转向到误差 e_yaw；返回指令或 None（已对正且停稳）。"""
        toward = 1.0 if np.sign(vel[2]) == np.sign(e_yaw) else -1.0
        remain = abs(e_yaw) - 0.5 * abs(vel[2]) * toward
        if remain > self.yaw_tol * 0.6:
            return np.array([0.0, 0.0, np.sign(e_yaw) * np.clip(0.9 * abs(e_yaw), 0.16, 0.5)])
        if not self._still(vel):
            return np.zeros(3)
        if abs(e_yaw) > self.yaw_tol:
            return np.array([0.0, 0.0, np.sign(e_yaw) * 0.16])
        return None

    CREEP = 0.03     # Aurora 迈步时有约 +0.03 m/s 的前向“蠕动”（原地转向时也会往前蹭）

    def __call__(self, pose, vel):
        """返回 (vx, vy, wz) 指令，以及是否已经到达并站稳。"""
        cmd, done = self._step(pose, vel)
        if abs(cmd[0]) < 1e-6 and (abs(cmd[1]) > 1e-6 or abs(cmd[2]) > 1e-6):
            cmd = cmd.copy()
            cmd[0] = -self.CREEP          # 抵消蠕动：原地转/横移时不往前蹭（这不是倒着走，合速度≈0）
        return cmd, done

    def _step(self, pose, vel):
        g = self.goal
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
        c, s = np.cos(pose[2]), np.sin(pose[2])
        d = g[:2] - pose[:2]
        ex, ey = c * d[0] + s * d[1], -s * d[0] + c * d[1]          # 站位在机体系下的位置
        dist = float(np.linalg.norm(d))
        e_yaw = wrap(g[2] - pose[2])
        if self.phase == "path":
            dp = self.pre - pose[:2]
            if np.linalg.norm(dp) < 0.12 or dist < 0.25:
                self.phase = "turn"
            else:
                path, _ = self.geo.geodesic_path(pose[:2], self.pre)
                look = path[1] if len(path) > 1 else self.pre
                if len(path) > 2 and np.linalg.norm(look - pose[:2]) < 0.25:
                    look = path[2]
                v = look - pose[:2]
                head_err = wrap(np.arctan2(v[1], v[0]) - pose[2])
                if abs(head_err) > 0.6:                     # 先原地转向，不倒着走
                    return np.array([0.0, 0.0, np.clip(1.2 * head_err, -0.6, 0.6)]), False
                remain = np.linalg.norm(dp) - stopping_distance(max(vel[0], 0.0))
                vx = np.clip(0.8 * remain, MIN_EFFECTIVE + 0.03, 0.45) * max(0.3, np.cos(head_err))
                return np.array([vx, 0.0, np.clip(1.5 * head_err, -0.6, 0.6)]), False
        if self.phase == "turn":
            cmd = self._turn_to(e_yaw, vel)
            if cmd is not None:
                return cmd, False
            self.trims = 0
            # 在预停靠点（离桌 0.6 m，安全）先把横向偏差修掉，再正对桌子直线走过去
            self.phase = "pre_shift" if (abs(ey) > self.pos_tol and ex > 0.2) else \
                ("approach" if ex > self.pos_tol + 0.06 else "trim")
        if self.phase == "pre_shift":
            if abs(ey) <= self.pos_tol * 0.8 or self.trims >= 3:
                if self._pulse_left == 0 and self._calm_enough(vel):
                    self.phase, self.trims = "approach", 0
                return self._pulse_or_wait(vel), False
            return self._trim_step(vel, e_yaw, ex, ey, lateral_only=True)
        if self.phase == "approach":
            # 正对站位直线前进；横向误差用很小的航向修正吸收
            if ex - (0.7 * max(vel[0], 0.0) + 0.03) > 0:
                wz = np.clip(0.4 * np.arctan2(ey, max(ex, 0.15)) + 0.6 * e_yaw, -0.2, 0.2)
                return np.array([0.18 if ex > 0.15 else 0.15, 0.0, wz]), False
            self.phase, self.trims = "trim", 0
        if self.phase == "trim":
            # Aurora 停指令后还会原地踏步好几秒（并带几厘米漂移），所以不等“完全静止”：
            # 速度基本降下来就判断误差；需要修就发一个定时小步（指令保持 T 秒），
            # 合格就结束 —— 之后由 AuroraNavRunner 切到 PdStand，PdStand 会立即停止踏步、站住。
            if self._pulse_left > 0 or not self._calm_enough(vel):
                return self._pulse_or_wait(vel), False
            if dist < self.pos_tol and abs(e_yaw) < self.yaw_tol:
                return np.zeros(3), True
            if self.trims >= self.MAX_TRIMS:
                if dist < 2 * self.pos_tol and abs(e_yaw) < 2 * self.yaw_tol:
                    return np.zeros(3), True              # 迈步本身有几厘米的随机性，够好了就收
                if self.retries < 2:                      # 偏得太多：退回预停靠点重新进场
                    self.retries += 1
                    self.phase = "path"
                return np.zeros(3), False
            return self._trim_step(vel, e_yaw, ex, ey)
        return np.zeros(3), False

    # ---------- 微调小步 ----------
    def _calm_enough(self, vel):
        calm = np.hypot(vel[0], vel[1]) < 0.07 and abs(vel[2]) < 0.12
        self._calm = self._calm + 1 if calm else 0
        return self._calm >= 5

    def _pulse_or_wait(self, vel):
        if self._pulse_left > 0:
            self._pulse_left -= 1
            return self._pulse_cmd
        return np.zeros(3)

    def _trim_step(self, vel, e_yaw, ex, ey, lateral_only=False):
        if self._pulse_left > 0 or not self._calm_enough(vel):
            return self._pulse_or_wait(vel), False
        self.trims += 1
        self._calm = 0
        if lateral_only:
            cmd, T = np.array([0.0, 0.22 * np.sign(ey), 0.0]), abs(ey) / 0.08 * 0.8
        elif abs(e_yaw) >= self.yaw_tol:                  # 先修朝向（每走一步都会带一点偏航）
            cmd, T = np.array([0.0, 0.0, 0.25 * np.sign(e_yaw)]), abs(e_yaw) / 0.2 * 0.8
        elif abs(ey) >= abs(ex) and abs(ey) > self.pos_tol * 0.7:
            cmd, T = np.array([0.0, 0.22 * np.sign(ey), 0.0]), abs(ey) / 0.08 * 0.8
        elif ex > 0:
            cmd, T = np.array([0.15, 0.0, 0.0]), ex / 0.10 * 0.8
        else:                                             # 冲过头才后退一小步
            cmd, T = np.array([-0.13, 0.0, 0.0]), -ex / 0.08 * 0.8
        self._pulse_cmd = cmd
        self._pulse_left = int(round(np.clip(T, 0.5, 1.2) / 0.1)) - 1
        return cmd, False
        return np.zeros(3), False


class AuroraNavRunner:
    """在 Aurora 上执行一次“走到站位”。policy 可以是：
         * PPO 策略（obs → action[-1,1]^3，用 sim.nav_aurora_env 的观测/动作定义）
         * ForwardFirstFollower 实例
    返回指标字典：是否到达、最终误差、用时、倒退距离、横移距离。"""

    # 安全限速：身体最前沿（躯干 + 走路时前摆的手臂）离骨盆约 0.15 m；停指令后的滑行按实测 0.08 + 0.6·v，
    # 再加上指令到步态响应的滞后（约 0.4 s 内还按当前速度走）。
    # 2026-10-31 用户环境的摔倒记录：骨盆以 0.53 m/s（实际速度，指令已降到 0.2–0.36）冲到 x=-0.77，
    # 桌沿在 -0.70，身体前沿撞上桌沿 → 被顶回 0.3 m → 侧摔。原来的限速只看指令、不看实际速度，滑行估计偏小。
    BODY_FRONT = 0.15
    SLIDE0, SLIDE_PER_SPEED, REACT_T = 0.08, 0.6, 0.4
    NEAR_TABLE, NEAR_TABLE_VX = 0.6, 0.25     # 骨盆离桌沿 <0.6 m 时前进指令 ≤0.25

    def __init__(self, loco: AuroraLoco, geo, policy, ctrl_hz=10.0, safety=True):
        from sim.nav_aurora_env import AuroraNavSkill
        self.loco, self.geo, self.policy = loco, geo, policy
        self.skill = AuroraNavSkill(geo)
        self.ctrl_dt = 1.0 / ctrl_hz
        self.safety = safety
        self.capped = 0
        self.min_front = 9.0
        self.balance_steps = 0
        self.max_roll = 0.0
        self._calm_since = None
        self._prev_vy = 0.0

    # 平衡保护：用户 Aurora 环境里的摔倒记录（truth_t.csv）显示，导航策略同时给大横移 (±0.25) 和大转向 (±0.7)、
    # 横移方向又来回翻转时，步态的左右晃动越来越大（横滚 0.13 → 0.27 rad、质心偏出双脚 0.2 m）直到侧摔；
    # 同一环境里按单轴指令走（aurora_sysid：横移 ±0.2、转向 ±0.5、前进 0.45）横滚始终 <0.03 rad、不摔。
    # 所以部署时把指令限制在实测稳定的范围里，并在机身横滚变大时先让步态自己恢复。
    VY_SAFE, WZ_SAFE = 0.15, 0.5            # 实测稳定范围（sysid 测过 ±0.2 / ±0.5）
    VY_WHEN_TURNING = 0.08                  # |wz| > 0.3 时横移上限：避免横移 + 转向叠加
    VY_RATE = 0.3                           # 横移指令变化率 (m/s²)，防止左右来回翻转
    ROLL_WARN, ROLL_CALM = 0.08, 0.04       # rad：超过就进入保护；回到 0.04 以下持续 0.5 s 才解除

    def balance_guard(self, cmd):
        cmd = np.array(cmd, float)
        roll = float(self.loco.c.get_base_data("rpy")[0])
        self.max_roll = max(self.max_roll, abs(roll))
        cmd[1] = np.clip(cmd[1], -self.VY_SAFE, self.VY_SAFE)
        cmd[2] = np.clip(cmd[2], -self.WZ_SAFE, self.WZ_SAFE)
        if abs(cmd[2]) > 0.3:
            cmd[1] = np.clip(cmd[1], -self.VY_WHEN_TURNING, self.VY_WHEN_TURNING)
        now = time.time()
        if abs(roll) > self.ROLL_WARN:
            self._calm_since = None
            self._guarding = True
        elif getattr(self, "_guarding", False):
            if abs(roll) < self.ROLL_CALM:
                self._calm_since = self._calm_since or now
                if now - self._calm_since > 0.5:
                    self._guarding = False
            else:
                self._calm_since = None
        if getattr(self, "_guarding", False):
            cmd[1] = 0.0
            cmd[2] *= 0.3
            cmd[0] = min(cmd[0], 0.2)
            self.balance_steps += 1
        step = self.VY_RATE * self.ctrl_dt
        cmd[1] = self._prev_vy + np.clip(cmd[1] - self._prev_vy, -step, step)
        self._prev_vy = cmd[1]
        return cmd

    def front_dist(self, pose):
        rays = self.skill.rays(pose)
        n = len(rays)
        return min(rays[0], rays[1] / np.cos(2 * np.pi / n) if rays[1] < 2 else 9,
                   rays[-1] / np.cos(2 * np.pi / n) if rays[-1] < 2 else 9)

    def speed_guard(self, pose, cmd, v_meas=0.0):
        """与策略无关的硬安全层：按正前方（±22.5°）到桌子的距离限制前进速度，保证“现在松指令，
        滑行之后身体前沿也碰不到桌子”。同时看指令和实测速度（机器人实际速度常常比指令大、减速慢）。"""
        front = self.front_dist(pose)
        self.min_front = min(self.min_front, front)
        room = front - self.BODY_FRONT - self.SLIDE0
        vmax = max(0.0, room / (self.SLIDE_PER_SPEED + self.REACT_T))
        if front < self.NEAR_TABLE:
            vmax = min(vmax, self.NEAR_TABLE_VX)
        cmd = np.array(cmd, float)
        v_meas = max(0.0, float(v_meas))
        if v_meas > vmax + 0.05 and cmd[0] > 0:     # 实际已经太快：立刻松指令（发 0，不发倒退），让步态自己刹车
            cmd[0] = 0.0
            self.capped += 1
        elif cmd[0] > vmax:
            cmd[0] = vmax if vmax >= 0.12 else 0.0       # 小于有效指令就干脆停（死区里本来也走不动）
            self.capped += 1
        return cmd

    @staticmethod
    def motion_stats(traj, window=1.8, floor=0.03):
        """倒退 / 横移距离：按两个步态周期 (1.8 s) 重采样位姿再差分，并忽略 <3 cm 的变化，
        消掉步态本身的前后/左右摆动（否则原地踏步也会被算成“倒退”）。"""
        tr = np.array([r[:4] for r in traj])
        if len(tr) < 2:
            return 0.0, 0.0
        ts = np.arange(tr[0, 0], tr[-1, 0] + 1e-6, window)
        P = np.stack([np.interp(ts, tr[:, 0], tr[:, k]) for k in (1, 2)], 1)
        Y = np.interp(ts, tr[:, 0], np.unwrap(tr[:, 3]))
        back = lat = 0.0
        for k in range(1, len(ts)):
            dp = P[k] - P[k - 1]
            c, s = np.cos(Y[k - 1]), np.sin(Y[k - 1])
            f, l = c * dp[0] + s * dp[1], -s * dp[0] + c * dp[1]
            back += -f if f < -floor else 0.0
            lat += abs(l) if abs(l) > floor else 0.0
        return back, lat

    def go_to(self, goal, timeout=90.0, log_every=2.0, finish="stand"):
        """finish='stand'：到达后切到 PdStand（停止踏步、原地站住，准备操作）；'walk'：停在行走模式。"""
        from sim.nav_aurora_env import action_to_cmd
        self.capped = 0
        self.min_front = 9.0
        self.balance_steps = 0
        self.max_roll = 0.0
        self._guarding, self._calm_since, self._prev_vy = False, None, 0.0
        lo = self.loco
        lo.walk_mode(carry=lo.upper_hold.get("right_manipulator"))
        is_rl = not isinstance(self.policy, ForwardFirstFollower)
        if is_rl:
            self.skill.set_goal(*goal)
            self.skill.begin(lo.pose())
        else:
            self.policy.set_goal(*goal)
        t0 = time.time(); last_log = t0
        vel_hist = []
        done = False
        traj = []
        while time.time() - t0 < timeout:
            pose = lo.pose()
            vel_hist = (vel_hist + [lo.vel_body()])[-10:]
            vel = np.mean(vel_hist[-5:], axis=0)          # 0.5 s 平均：给策略的观测（与训练一致）
            vel_cycle = np.mean(vel_hist, axis=0)         # 1 s ≈ 一个步态周期：判断“停稳”时消掉左右摆
            traj.append((time.time() - t0, *pose, *lo.cmd))
            if is_rl:
                a = self.policy(self.skill.obs(pose, vel))
                raw = getattr(self.policy, "outputs_cmd", False)     # 适配器直接输出速度指令
                _, done, info = self.skill.evaluate(pose, np.clip(a, -1, 1), vel_cycle, False, (0.0, 0.0))
                cmd = a if raw else action_to_cmd(a)
            else:
                cmd, done = self.policy(pose, vel)
            if done:
                break
            if self.safety:
                cmd = self.speed_guard(pose, cmd, vel[0])
                cmd = self.balance_guard(cmd)
            else:
                self.max_roll = max(self.max_roll, abs(float(lo.c.get_base_data("rpy")[0])))
            lo.command(*cmd)
            if time.time() - last_log > log_every:
                last_log = time.time()
                e = goal[:2] - pose[:2]
                lo.log(f"    t={time.time() - t0:5.1f}s pos=({pose[0]:+.2f},{pose[1]:+.2f}) yaw={pose[2]:+.2f} "
                       f"dist={np.linalg.norm(e):.2f} cmd=({lo.cmd[0]:+.2f},{lo.cmd[1]:+.2f},{lo.cmd[2]:+.2f})")
            lo.wait(self.ctrl_dt)
        if finish == "stand":
            lo.stand_mode()
        else:
            lo.stop()
        lo.wait(1.0)
        pose = lo.pose()
        traj.append((time.time() - t0, *pose, 0.0, 0.0, 0.0))
        back, lat = self.motion_stats(traj)
        e = goal[:2] - pose[:2]
        yaw_e = (goal[2] - pose[2] + np.pi) % (2 * np.pi) - np.pi
        return {"reached": bool(done), "time": time.time() - t0, "pos_err": float(np.linalg.norm(e)),
                "yaw_err": float(abs(yaw_e)), "back_dist": back, "lateral_dist": lat, "guard_steps": self.capped, "min_front": self.min_front,
                "balance_steps": self.balance_steps, "max_roll": self.max_roll, "traj": traj}
