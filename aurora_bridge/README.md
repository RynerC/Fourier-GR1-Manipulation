# aurora_bridge —— 用 Fourier Aurora SDK 驱动 GR1（GR-1P / gr1t2）

把项目里“平面底盘 + 腿部动画”的假行走换成 Aurora 的真实步态与站立平衡控制器，
并按实测的步态特性重新设计导航（不再倒着走）。

```
 规划器 / 导航技能 (PPO 或几何跟踪器)
          │  (vx, vy, wz)          上身关节 / 站姿 Δz Δpitch Δyaw
          ▼                                   ▼
   AuroraLoco (locomotion.py) ── DDS ──► AuroraCore (FSM)
      walk_mode()  → FSM 3 UserController_A：LowerBodyCpgRLTask 走路（腿+腰）+ 上身远程控制
      stand_mode() → FSM 2 PdStand：全身平衡站立，上身关节/站姿可调 → 操作
                                             │ LCM（PD 指令 / 传感器）
                                             ▼
                             sim_server.py（无界面 MuJoCo，GR1T2 + 桌子场景）
```

## 文件

| 文件 | 作用 |
|---|---|
| `aurora_compat.py` | 让 pip 版 `fourier_aurora_client==0.1.8` 能连 GR-1P 镜像里的 AuroraCore v1.2.1（消息定义、可选话题、初始化等待等 4 个兼容问题） |
| `sim_server.py` | 无界面仿真：吊绳（`lower` 慢放）、`reset[:x,y,yaw]`、`payload:<kg>`、`place:<物体>:x,y[,z,yaw]`、`grasp:`/`drop`、`--scene tabletop`、`--truth-log` |
| `start_stack.sh` / `stop_stack.sh` | 一键启动 / 停止 仿真 + AuroraCore（会清理残留的 AuroraCore —— 两个 Core 抢同一 DDS 域会让机器人摔倒） |
| `locomotion.py` | `AuroraLoco`（启动流程、走/站模式切换、限幅与变化率限制、平滑上身/站姿指令）、`ForwardFirstFollower`（几何基线导航）、`AuroraNavRunner`（把导航技能接到 Aurora 上并统计倒退/横移） |
| `gait_model.py` | 在 Aurora 仿真里辨识出来的速度响应模型（滞后、增益、死区、刹车滑行、前向蠕动、偏航漂移），用于快速训练环境 `sim/nav_aurora_env.py` |
| `np_policy.py` | 纯 numpy 的策略推理（Aurora 镜像里没有 Python 版 torch） |
| `arm_ik.py` | GR1T2 右臂（+腰 yaw/pitch）阻尼最小二乘 IK，PdStand 下伸手用；可选“保持物体竖直”（`up_axis`）、姿态目标、零空间偏好姿态 |
| `manip_runtime.py` | `AuroraManipRuntime`：把快速环境训练的操作策略接到 Aurora（同一套观测/动作/抓取规则），`scripts/aurora_manip_eval.py`（子任务单元测试）和 `scripts/aurora_manip_task.py`（连续任务）都用它 |
| `lcm_msgs/` | 放官方 SDK 的 LCM 消息定义（不随仓库分发，由 `scripts/fetch_assets.sh` 从镜像拷入） |

## 运行（在 GR-1P Aurora 镜像里）

```bash
# 宿主机（Windows 用 WSL2 + Docker Desktop 也可以），把本项目挂载到 /workspace：
docker load -i fourier_aurora_sdk_gr1p_v1.2.1.tar        # 镜像来自 SDK 压缩包
docker run -it --rm --net=host --ipc=host --cap-add=SYS_NICE \
    -v "$(pwd)":/workspace -w /workspace <gr1p 镜像名> bash

# 容器内（第一次）
pip install fourier_aurora_client==0.1.8
pip install imageio imageio-ffmpeg          # 只有要渲染视频时才需要（scripts/render_aurora_video.py）
apt-get install -y fonts-noto-cjk           # 可选：视频里的中文字幕需要中文字体
sed -i 's/UseJoyStick: true/UseJoyStick: false/' /opt/fftai/fourier_aurora/config/sys_config/config.yaml

# 启动仿真 + AuroraCore（桌子场景，机器人出生在桌子 -x 侧 1.6 m 处）
bash aurora_bridge/start_stack.sh --scene tabletop --spawn=-1.6,0,0
# 导航验证：几何跟踪器 / PPO 策略
python3 scripts/aurora_nav_eval.py --policy follower --tour 6
python3 scripts/aurora_nav_eval.py --policy checkpoints/nav_aurora/best.npz --tour 6
# 连续任务：PPO 导航 + 模仿学习操作策略，把 3 个物体依次放上托盘（启动时加 --qpos-log /tmp/q.log 可事后渲染视频）
python3 scripts/aurora_manip_task.py
python3 scripts/aurora_manip_task.py --layout keep --resume       # 接着上次结束的状态继续
bash aurora_bridge/stop_stack.sh
```

> 实时性：AuroraCore 的行走任务 50 Hz、PdStand 400 Hz，仿真 1 kHz。CPU 不够时
> （日志里大量 `time out`、`[sim] lag xxx ms`）机器人会无故摔倒 —— 至少给 4 个物理核，
> 别同时跑训练。
> 自己写的客户端脚本：**在 import numpy 之前设 `OPENBLAS_NUM_THREADS=1`**（本项目脚本已内置）。镜像里的 numpy 用多线程 OpenBLAS，
> 多核机器上每次矩阵乘都会叫醒全部核并忙等，抢走 AuroraCore 的 CPU，行走任务大量超时、走几秒就摔。

## 必须遵守的使用规则（每条都在仿真里摔过）

1. **进 PdStand 后立刻把腰回正。** 零力矩（Default）时腰会在重力下前倾到 0.4–1.2 rad，
   PdStand 会冻结进入时的上身姿态 → 躯干前倾 → `isStanceState`（双脚接触、CoM 在双脚中点 10 cm 内、
   俯仰在 [-0.45, 0.6]）一碰就不满足 → 切到 Hanging（屈膝悬挂）→ 摔倒。JointStand 只使能腿
   （`motor_enable: 0xFFF`），解决不了。
2. **吊着先进出一次 FSM3 预热**（冷启动加载行走策略要 2.7 s，期间腿是软的），回 PdStand 后**等双腿对称**
   （≥3.5 s）再**慢慢放吊绳 / 龙门吊**，直到脚完全承重再摘（仿真 `lower`）。半吊半站时突然松开会冲一下；
   一腿直一腿弯地落地会摔。
3. **站姿 Δz/Δpitch/Δyaw 和上身关节指令要斜坡发送**（0.05 m/s、0.15 rad/s），不能跳变；
   站姿一直是 0 时不要持续发送。
4. **v1.2.1 的速度指令没有超时**，停车必须显式发 0。
5. **死区**：|v_xy|≤0.1 且 |wz|≤0.1（乘 0.9 安全系数后）持续 1.35 s → 停止迈步。
   有效最小指令约 0.12 m/s / 0.12 rad/s。
6. 停指令后还会**原地踏步好几秒**并带 ~0.03 m/s 的前向蠕动；要精确停住就切到 PdStand。
7. **走 → 站**：先发 0 再切 FSM2；**站 → 走**：站姿回 0、等 ≥1 s 再切 FSM3，然后
   `set_velocity_source(2)`。端持物体用贴身姿态 `ARM_TUCK`。
8. 同一 DDS 域里只能有一个 AuroraCore。

## 实测步态特性（GR1T2，正确启动流程下）

| 指令 | 实际稳态 | 备注 |
|---|---|---|
| vx 0.12 / 0.15 / 0.30 / 0.45 | 0.15 / 0.18 / 0.33 / 0.48 | 约 +0.03 偏置，t63≈0.6 s |
| vx −0.15 / −0.40 | −0.14 / −0.35 | |
| vy +0.20 / −0.20 | +0.09 / −0.13 | 横移只有 45–65 %，t63≈1.5 s |
| wz 0.5 / −0.5 / 0.15 | 0.43 / −0.47 / 0.11 | 停转后还会多转 0.12–0.21 rad |
| 刹车距离 | 0.08 + 0.43·v | 0.3 m/s 时滑 0.20 m |
| 直走偏航漂移 | −0.01 ~ −0.02 rad/s | 需要闭环修正航向 |
