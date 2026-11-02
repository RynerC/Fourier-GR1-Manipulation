# GR1 桌面抓放：Aurora 步态 + 强化学习导航 + 模仿学习操作（MuJoCo）

Fourier GR1T2（GR-1P）人形机器人在 MuJoCo 全物理仿真里完成桌面整理：
**找到桌上的物体 → 用 Aurora 步态走到抓取站位 → 站着抓起 → 托盘够不着就端着绕桌走过去 → 放到托盘空位上**，
一次开机、状态连续，把 3 个物体依次放上托盘。

- **行走 / 平衡**：Fourier Aurora SDK（AuroraCore v1.2.1）。FSM 3 行走，FSM 2 PdStand 全身平衡站立（可调站姿 + 上身关节）。
- **导航技能**：强化学习。先行为克隆（几何跟踪器 `ForwardFirstFollower` 当老师）初始化，再用 PPO 分阶段训练（倒退代价逐步加大）。
  训练环境 `sim/nav_aurora_env.py` 是按 Aurora 实测步态辨识出来的快速模型。
- **操作技能**：模仿学习（行为克隆 + DART 噪声 + DAgger），老师是基于真值和 IK 的脚本策略 `ScriptedManip`。
  训练环境 `sim/manip_aurora_env.py` 是按 PdStand 实测特性建的快速近似环境，约 20 倍实时。
  PPO 微调也试过，没有提升（见 `docs/rl_training_guide.md` 第 8 节），所以部署的是纯模仿学习策略。
- **部署**：策略导出成纯 numpy（`.npz`），在 Aurora 镜像里通过 DDS 控制 AuroraCore；无界面 MuJoCo 仿真 `aurora_bridge/sim_server.py` 通过 LCM 与 AuroraCore 连接。

视频：`videos/aurora_manip_task.mp4`（连续任务，3 个物体全部放上托盘）。

## 结果（Aurora 全物理仿真）

| 测试 | 结果 |
| --- | --- |
| 连续任务 `aurora_manip_task.py`（3 个物体，一次开机） | `--layout default` 3/3（196 s），`--seed 1` 3/3（217 s），`--seed 2` 3/3（200 s），0 摔倒 |
| 导航：连续 6 个桌边站位，PPO `checkpoints/nav_aurora` | 6/6，平均误差 2.7 cm，平均 17 s，倒退合计 0.29 m，0 摔倒 |
| 操作策略，快速环境 300 回合（抓放 / 抓起端持 / 放置） | 学生 94%（90 / 100 / 92）；脚本老师 96.7% |

导航对比（`docs/aurora_results/nav_*.json`，轨迹图 `docs/img/aurora_nav_compare.png`）：

| 导航策略 | 到达 | 平均误差 | 平均用时 | 倒退合计 |
| --- | --- | --- | --- | --- |
| 旧策略（早期版本在理想底盘上训练，直接发给 Aurora；已删除） | 0/4 | 33 cm | 55 s（超时） | 18.3 m |
| 几何前进优先跟踪器 `ForwardFirstFollower` | 6/6 | 8.2 cm | 34 s | 0.65 m |
| **PPO `checkpoints/nav_aurora`** | **6/6** | **2.7 cm** | **17 s** | **0.29 m** |

## 1. 安装（训练机）

```bash
cd gr1_pickplace_rl
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate
pip install -r requirements.txt
# 有 NVIDIA GPU：先到 https://pytorch.org 按 CUDA 版本装 torch。仿真跑在 CPU 上，核越多越快
```

**本仓库不包含任何 Fourier 的内容**（SDK、镜像、机器人模型和网格、SDK 里的消息定义），需要自己从 Fourier 获取
GR-1P Aurora SDK（`fourier_aurora_sdk_gr1p_v1.2.1`），`docker load` 镜像后在项目根目录运行一次：

```bash
bash scripts/fetch_assets.sh <gr1p 镜像名>              # docker images 查看镜像名
```

它从镜像里拷出 GR1T2 模型（→ `assets/gr1t2/`）和 LCM 消息定义（→ `aurora_bridge/lcm_msgs/`）。
没有这一步，快速操作环境和 Aurora 仿真都无法运行（导航的快速环境是纯 numpy，不需要）。
镜像的容器启动和第一次配置见 `aurora_bridge/README.md`。

## 2. 训练（快速环境，训练机上）

逐步说明（每步的正常日志、参考结果、出问题改哪里）见 `docs/rl_training_guide.md`。

```bash
# 导航：行为克隆初始化 + PPO（约 30 分钟 CPU）
python scripts/pretrain_bc_nav_aurora.py --episodes 1500
python scripts/train.py --task nav_aurora --resume checkpoints/nav_aurora_bc/bc.pt --lr 1e-4 --steps 1000000
python scripts/eval_nav_aurora.py --policy checkpoints/nav_aurora/best.pt
python scripts/export_policy_npz.py checkpoints/nav_aurora/best.pt

# 操作：行为克隆 + DAgger（28 核几分钟，2 核约 35 分钟）
python scripts/eval_manip_aurora.py --policy scripted                  # 先确认环境正常：老师应 ≥95%
python scripts/pretrain_bc_manip_aurora.py --episodes 1500 --dagger 2 --out checkpoints/my_manip_bc/bc.pt
python scripts/eval_manip_aurora.py --policy checkpoints/my_manip_bc/bc.pt
python scripts/export_policy_npz.py checkpoints/my_manip_bc/bc.pt
```

## 3. 在 Aurora 里运行（GR-1P 镜像，项目挂载到 /workspace）

```bash
bash aurora_bridge/start_stack.sh --scene tabletop --spawn=-1.6,0,0 --qpos-log /tmp/q.log

python3 scripts/aurora_manip_task.py                                  # 连续任务：PPO 导航 + BC 操作，3 个物体放上托盘
python3 scripts/aurora_manip_task.py --layout keep --resume           # 接着上次结束的状态继续（物体、机器人都不重置）
python3 scripts/aurora_nav_eval.py --policy checkpoints/nav_aurora/best.npz --tour 6     # 只测导航
python3 scripts/aurora_manip_eval.py --policy checkpoints/manip_aurora_bc/bc.npz --episodes 9   # 操作子任务单元测试

bash aurora_bridge/stop_stack.sh
# 离线渲染视频（先停掉 Aurora，任何装了 mujoco 的机器都行）
MUJOCO_GL=egl python3 scripts/render_aurora_video.py /tmp/q.log --speed 3 --out videos/manip_task.mp4
```

`aurora_manip_task.py` 默认 `--layout random --seed 1`，每次运行会按 seed 重新摆桌面；要接着做必须加 `--layout keep`。
`aurora_manip_eval.py` 每回合重新开机，“放置”模式开局直接把物体放进手里（为了单独测放置），不是连续任务。

使用 Aurora 时必须遵守的规则（进 PdStand 先回正腰、吊着预热再放、站姿和关节指令斜坡发送、`OPENBLAS_NUM_THREADS=1` 等）
见 `aurora_bridge/README.md`；每个摔倒问题的原因和修法见 `docs/aurora_posture_issues.md`。

## 4. 目录

| 路径 | 内容 |
| --- | --- |
| `config.py` | 场景、机器人、训练的常数与超参数（托盘尺寸 `tray_half` 等） |
| `sim/nav_aurora_env.py` | Aurora 步态快速导航环境（前进优先、倒退代价） |
| `sim/manip_aurora_env.py` | PdStand 快速操作环境（抓放 / 抓起端持 / 放置），脚本老师 `ScriptedManip` |
| `sim/gr1t2_manip_scene.py` | 生成快速操作环境的 MJCF（`assets/gr1t2/mjcf/manip_scene.xml`） |
| `aurora_bridge/` | Aurora 接入：客户端兼容层、无界面仿真、走/站控制、步态模型、右臂 IK、操作运行时 |
| `planning/` | 桌子几何（站位、可达性、绕桌路径） |
| `rl/` | 从零实现的 PPO 和并行环境（导航训练、操作策略的网络结构） |
| `scripts/` | 训练、评估、导出、渲染（`aurora_*.py` 在 Aurora 镜像里运行） |
| `checkpoints/` | 部署用模型：`nav_aurora/best.npz`（导航）、`manip_aurora_bc/bc.npz`（操作） |
| `docs/` | 训练指南、姿态问题清单、`aurora_results/` 实测日志 |
| `assets/` | 机器人模型（不随仓库分发，由 `scripts/fetch_assets.sh` 准备） |

## 致谢与许可

本仓库只包含自己编写的代码、自己训练的模型、自己测得的结果和自己渲染的视频。以下内容属于 Fourier，**不随本仓库分发**，
请从官方渠道获取并遵守其许可：

- Fourier Aurora SDK、AuroraCore 和 GR-1P SDK 镜像。
- GR1T2 模型与网格（镜像内 `/usr/local/resources/GR1T2`）。
- LCM 消息定义 `simulator_pub_msg.py` / `simulator_sub_msg.py`（镜像内 `/workspace/aurora_sim`，或 [fourier_aurora_sdk](https://github.com/FFTAI/fourier_aurora_sdk) 的 `sim/mujoco/`）。
