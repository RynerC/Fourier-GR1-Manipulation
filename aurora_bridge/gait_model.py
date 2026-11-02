"""
aurora_bridge/gait_model.py —— Aurora GR1T2 行走控制器的“速度响应模型”（系统辨识结果）
=====================================================================================
为什么需要它？
  导航策略发出的是速度指令 (vx, vy, wz)，但真正让机器人动起来的是 Aurora 的
  LowerBodyCpgRLTask（CPG + RL 行走策略）。它不是一个理想的“底盘”：
    * 有滤波和步态惯性 → 指令变化后 ~0.6 s 才跟上（横移更慢 ~1.5 s）
    * 稳态不等于指令（前进约 +0.03 m/s，横移只有 45–65 %，转向约 88 %）
    * 有死区：|v_xy|≤0.1 且 |wz|≤0.1（乘安全系数 0.9 后）持续 1.35 s → 停止迈步
    * 停下来要滑一段（刹车距离 ≈ 0.08 + 0.43·v）
    * 直走时会慢慢偏航（≈ -0.015 rad/s）
  导航策略如果在“理想底盘”上训练，部署到 Aurora 上就会冲过头、停不准、绕圈。
  所以：先在 Aurora 仿真里测（scripts/aurora_sysid.py），再把测到的规律写成这个模型，
  用它搭一个快速的 2D 训练环境（sim/nav_aurora_env.py）—— 这就是“系统辨识 + 域随机化”的
  sim-to-sim 迁移思路，和 sim-to-real 是同一套方法。

数字来源：GR-1P(gr1t2) Aurora v1.2.1 + MuJoCo，正确启动流程下（见 locomotion.py）实测，
2026-10-29。config_gr1t2_mix.yaml：vx[-0.4,0.5] vy[-0.3,0.3] wz[-0.8,0.8]，安全系数 0.9，
指令滤波截止 0.5 Hz（50 Hz 下 α=0.9409），步态周期 0.9 s，抬脚 6 cm。
"""
from dataclasses import dataclass

import numpy as np

# Aurora LowerBodyCpgRLTask 的指令范围（config_gr1t2_mix.yaml）
VX_RANGE = (-0.40, 0.50)
VY_RANGE = (-0.30, 0.30)
WZ_RANGE = (-0.80, 0.80)
SAFETY_RATIO = 0.9
DEADBAND = 0.10            # GR1CpgLocoPolicy::updateCpgLocoState 里的阈值（反汇编得到）
STAND_DELAY = 1.35         # 低于死区持续 1.5×步态周期(0.9 s) 才切到 STAND
MIN_EFFECTIVE = DEADBAND / SAFETY_RATIO + 0.01   # ≈0.12：比它小的指令会被当成“停”

# 实测稳态响应（指令 → 实际，机体系）
MEASURED = {
    "vx": [(0.12, 0.15), (0.15, 0.18), (0.30, 0.33), (0.45, 0.48), (-0.15, -0.14), (-0.20, -0.19),
           (-0.40, -0.35)],
    "vy": [(0.20, 0.09), (-0.20, -0.13), (0.25, 0.15), (-0.25, -0.12)],
    "wz": [(0.50, 0.43), (-0.50, -0.47), (0.15, 0.11), (0.70, 0.61), (0.40, 0.34)],
    "t63": {"vx": 0.6, "vy": 1.5, "wz": 0.5},
    "stop_distance": [(0.15, 0.14), (0.30, 0.20), (0.45, 0.27)],
    "yaw_overshoot_after_turn_stop": (0.12, 0.21),
    "yaw_drift_straight": -0.015,
    "creep_while_turning_in_place": 0.028,     # m/s，前向
}


@dataclass
class GaitParams:
    """一组模型参数。训练时每个回合在 nominal 附近随机（域随机化）。"""
    gain_vx_fwd: float = 1.00
    bias_vx_fwd: float = 0.03
    gain_vx_back: float = 0.93
    gain_vy_left: float = 0.50
    gain_vy_right: float = 0.62
    gain_wz: float = 0.88
    tau_x: float = 0.30          # 滤波之后再串一个一阶惯性（总 t63 ≈ 0.6 s）
    tau_y: float = 1.00
    tau_w: float = 0.25
    yaw_drift: float = -0.015    # rad/s，迈步时
    sway_amp: float = 0.12       # 横向速度摆动幅值 (m/s)，1.11 Hz
    stop_slide: float = 0.06     # 停步时最后半步的额外滑移 (m)

    @staticmethod
    def randomized(rng, scale=1.0):
        p = GaitParams()
        j = lambda v, r: v * (1 + scale * rng.uniform(-r, r))  # noqa: E731
        p.gain_vx_fwd, p.gain_vx_back = j(p.gain_vx_fwd, 0.08), j(p.gain_vx_back, 0.08)
        p.bias_vx_fwd = p.bias_vx_fwd + scale * rng.uniform(-0.03, 0.02)
        p.gain_vy_left, p.gain_vy_right = j(p.gain_vy_left, 0.3), j(p.gain_vy_right, 0.3)
        p.gain_wz = j(p.gain_wz, 0.1)
        p.tau_x, p.tau_y, p.tau_w = j(p.tau_x, 0.4), j(p.tau_y, 0.4), j(p.tau_w, 0.4)
        p.yaw_drift = p.yaw_drift + scale * rng.uniform(-0.02, 0.02)
        p.sway_amp = j(p.sway_amp, 0.4)
        p.stop_slide = p.stop_slide * rng.uniform(0.5, 1.0 + 1.5 * scale)   # Aurora 实测刹车滑行常比模型长
        return p


class AuroraGaitModel:
    """离散时间仿真：输入速度指令，输出机体系实际速度。dt 是内部积分步长 (50 Hz，与 Aurora 一致)。"""
    DT = 0.02
    ALPHA = 0.9409               # Aurora 指令一阶滤波系数（cf=0.5 Hz, 50 Hz）

    def __init__(self, params: GaitParams = None, rng=None):
        self.p = params or GaitParams()
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.cmd_f = np.zeros(3)         # Aurora 内部滤波后的指令
        self.v = np.zeros(3)             # 实际机体速度 (vx, vy, wz)
        self.walking = False
        self.low_t = 0.0
        self.phase = self.rng.uniform(0, 2 * np.pi)
        self.slide_left = 0.0

    @staticmethod
    def clip_cmd(cmd):
        return np.array([np.clip(cmd[0], *VX_RANGE), np.clip(cmd[1], *VY_RANGE),
                         np.clip(cmd[2], *WZ_RANGE)])

    def _target(self, c):
        p = self.p
        # 迈步时整体有 ~+0.03 m/s 的前向偏置（实测：原地转 5 s 往前蹭 0.12–0.14 m；0.12→0.15）
        vx = (p.gain_vx_fwd if c[0] > 0 else p.gain_vx_back) * c[0] + p.bias_vx_fwd
        vy = c[1] * (p.gain_vy_left if c[1] > 0 else p.gain_vy_right)
        return np.array([vx, vy, p.gain_wz * c[2]])

    def step(self, cmd):
        """推进 DT 秒，返回 (vx, vy, wz) 机体系实际速度（含步态横摆）。"""
        p, dt = self.p, self.DT
        c = self.clip_cmd(cmd)
        self.cmd_f = self.ALPHA * self.cmd_f + (1 - self.ALPHA) * c
        cs = self.cmd_f * SAFETY_RATIO
        low = np.hypot(cs[0], cs[1]) <= DEADBAND and abs(cs[2]) <= DEADBAND
        if low:
            self.low_t += dt
            if self.walking and self.low_t >= STAND_DELAY:
                self.walking = False
                self.slide_left = p.stop_slide * np.sign(self.v[0]) if abs(self.v[0]) > 0.05 else 0.0
        else:
            self.low_t = 0.0
            self.walking = True
        tgt = self._target(self.cmd_f) if self.walking else np.zeros(3)
        tau = np.array([p.tau_x, p.tau_y, p.tau_w])
        self.v += (tgt - self.v) * (1 - np.exp(-dt / tau))
        out = self.v.copy()
        if self.walking:
            self.phase += 2 * np.pi * 1.11 * dt
            out[1] += p.sway_amp * np.sin(self.phase)
            out[2] += p.yaw_drift
        elif abs(self.slide_left) > 1e-4:          # 停步最后半步的滑移
            d = np.sign(self.slide_left) * min(abs(self.slide_left), 0.15 * dt)
            out[0] += d / dt
            self.slide_left -= d
        return out


def stopping_distance(v):
    """实测刹车距离（发 0 指令之后还会往前走多远）。"""
    return 0.08 + 0.43 * abs(v) if abs(v) > 0.05 else 0.0
