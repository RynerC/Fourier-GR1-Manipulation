"""
config.py —— 全局配置（所有“魔法数字”集中在这里）
=====================================================
为什么要集中配置？
  * 强化学习调参时，会反复修改桌子尺寸、奖励权重、学习率……
    如果这些数字散落在各个文件里，很容易改漏、改错。
  * 场景构建 (sim/)、任务规划 (planning/)、训练脚本 (scripts/) 都从这里读取，
    保证“同一个世界”只有一份定义。
"""
import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
VIDEO_DIR = os.path.join(ROOT, "videos")

# ----------------------------------------------------------------------------
# 1. 场景：桌子 / 物体 / 托盘
# ----------------------------------------------------------------------------
@dataclass
class SceneCfg:
    table_pos: tuple = (0.0, 0.0)          # 桌子中心 (x, y)
    table_half: tuple = (0.70, 0.32)       # 桌面半长、半宽 (m) → 1.4m x 0.64m
    table_height: float = 0.80             # 桌面高度 (m)
    table_thick: float = 0.03
    # 3 个可识别的物体：名字 → (形状, 尺寸, 颜色RGBA)
    objects: dict = field(default_factory=lambda: {
        "red_block":      ("box",      (0.025, 0.025, 0.06),  (0.90, 0.10, 0.10, 1)),
        "green_can":      ("cylinder", (0.028, 0.06),         (0.10, 0.75, 0.15, 1)),
        "blue_box":       ("box",      (0.022, 0.035, 0.06),  (0.10, 0.25, 0.95, 1)),
    })
    tray_half: tuple = (0.12, 0.12, 0.006)  # 放置托盘（紫色）。24×24 cm：要放下 3 个物体（四角各离中心 6 cm、相距 12 cm），
                                            # 16 cm 的旧托盘四角离边只剩 1 cm，放偏 2 cm 物体就从边上翻下去
    tray_rgba: tuple = (0.60, 0.20, 0.80, 1)


# ----------------------------------------------------------------------------
# 2. 机器人
# ----------------------------------------------------------------------------
@dataclass
class RobotCfg:
    # “站位”：面对桌边时骨盆离桌边的距离
    standoff: float = 0.25


# ----------------------------------------------------------------------------
# 3. PPO 超参数（导航 / 操作可以各自覆盖）
# ----------------------------------------------------------------------------
@dataclass
class PPOCfg:
    num_envs: int = 8               # 并行环境数（CPU 核数越多越好）
    rollout_steps: int = 256        # 每个环境每轮收集多少步
    total_steps: int = 5_000_000
    gamma: float = 0.99             # 折扣因子：越接近 1 越“有远见”
    gae_lambda: float = 0.95        # GAE 的偏差-方差折中
    clip_eps: float = 0.2           # PPO 的“信任域”裁剪幅度
    lr: float = 3e-4
    epochs: int = 5                 # 每批数据重复利用几遍
    minibatches: int = 4
    ent_coef: float = 0.0           # 熵奖励（鼓励探索）
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    hidden: tuple = (256, 256)
    init_log_std: float = -0.5
    target_kl: float = 0.03         # KL 过大时提前停止本轮更新，防止策略崩坏
    seed: int = 0
