# 自己训练：Aurora 步态版导航（nav_aurora）操作步骤

整个流程分两处：**训练机**（Windows/Linux 都行，有没有 GPU 都行）做训练，**Aurora 容器**（Docker，Linux）做验收。
训练不需要 Aurora，也不需要 MuJoCo —— `sim/nav_aurora_env.py` 是纯 numpy 环境。

## 0. 准备（只做一次）

训练机：
```bash
cd J:\robo@home\gr1_pickplace_rl            # Linux: cd .../gr1_pickplace_rl
python -m venv .venv
.venv\Scripts\activate                     # Linux: source .venv/bin/activate
pip install -r requirements.txt            # 有 NVIDIA GPU：先按 pytorch.org 装带 CUDA 的 torch
```
操作策略（第 8 节）和 Aurora 验收需要 Fourier 的 GR1T2 模型和 SDK 文件（不随仓库分发）：`docker load` SDK 镜像后在项目根目录运行一次
`bash scripts/fetch_assets.sh <gr1p 镜像名>`。只训练导航不需要。

导航网络很小，GPU 帮助不大；`--num-envs` 设成 CPU 核数（下文用 8，按你的机器改）。

## 1. 行为克隆预训练（约 5 分钟）

用几何跟踪器 `ForwardFirstFollower` 当老师采 1500 回合，网络先学会“转身—往前走—停靠”。
```bash
python scripts/pretrain_bc_nav_aurora.py --episodes 1500 --out checkpoints/my_nav_bc/bc.pt
```
看输出：老师成功率应 ≈0.9，MSE 从 ~0.02 降到 ~0.008。学生单独评估只有 ~15%（老师有内部状态，学生学不全）——正常，下一步 PPO 补上。

## 2. PPO 第一阶段（倒退代价 8 / 米，100 万步）

```bash
python scripts/train.py --task nav_aurora --resume checkpoints/my_nav_bc/bc.pt ^
       --lr 1e-4 --steps 1000000 --back-penalty 8 --num-envs 8 --run-name my_nav_s1
```
（命令太长换行时：Windows cmd 用行尾 `^`，PowerShell 用行尾反引号，Linux 用 `\`；也可以写成一行。）每轮一行日志，重点看：

| 字段 | 正常走势 |
|---|---|
| `success` | 一开始就 0.8 左右（BC 的功劳），几万步内到 0.99–1.0 |
| `len` | 回合长度（×0.1 s）：270 → 110 左右，越短说明走得越快 |
| `return` | 从 ~0 升到 15–18 |
| `kl` | 0.003–0.01；长期 >0.03 说明学习率太大 |
| `std` | 0.36 → 0.32 慢慢降 |

结果在 `checkpoints/my_nav_s1/`：`last.pt`（最新）、`best.pt`、`progress.csv`。
**注意：`best.pt` 按训练时（带随机噪声）的成功率挑的，微调时常常停在很早的一版；以第 4 步的确定性评估为准，通常 `last.pt` 更好。**

画曲线：`python scripts/plot_progress.py --task my_nav_s1` → `videos/curve_my_nav_s1.png`。

## 3. PPO 第二阶段（倒退代价 20 / 米，60 万步，学习率减半）

第一版放到 Aurora 上，停靠冲过头时还会退 0.1–0.3 m；把倒退代价调高再微调：
```bash
python scripts/train.py --task nav_aurora --resume checkpoints/my_nav_s1/last.pt ^
       --lr 5e-5 --steps 600000 --back-penalty 20 --num-envs 8 --run-name my_nav_s2
```

## 3b. 第三阶段：固定小探索噪声收尾（100 万步，强烈建议）

训练日志里的 `success` 是**带探索噪声**的策略跑出来的；部署和评估用的是**确定性策略**（取均值）。
到了站位附近只差几厘米时，策略的均值输出常常落在 Aurora 的速度死区里（≤0.1 m/s，机器人根本不动），
训练时靠噪声偶尔冲出死区才完成停靠 —— 所以训练曲线 99%，确定性评估可能只有 ~90%
（实测一个第二阶段模型：确定性 90.5%，失败的全部是停在离目标 6–9 cm 处、指令 95% 落在死区里）。
解决：把探索标准差固定成 0.1 再微调，策略不能再指望噪声，只能自己发出够大的指令：
```bash
python scripts/train.py --task nav_aurora --resume checkpoints/my_nav_s2/last.pt ^
       --lr 5e-5 --steps 1000000 --back-penalty 20 --fix-std 0.1 --num-envs 16 --run-name my_nav_s3
```
**挑模型看 `best.pt`（新版代码）。** nav_aurora 训练时每 10 轮会用确定性策略在固定的 100 个回合上评估一次（日志里的
`[确定性评估]` 行，记录在 `eval.csv`），分数最高的存为 `best.pt`。同一次训练里，相邻检查点的确定性到达率
能在 93%–100% 之间跳，`last.pt` 只是最后一个，不一定是最好的。
（2026-10-29 之前的代码里，`best.pt` 是按训练日志的 success 挑的：带探索噪声，而且一到 1.0 就不再更新，
常停在很早的一版 —— 用旧代码训出的 `best.pt` 往往比 `last.pt` 还差，别用。）
最终仍要用第 4 步在另一组回合（换 `--seed`）上复核，因为 best.pt 是在那 100 个回合上挑出来的，会略偏乐观。

开始时训练 `success` 会掉到 0.92 左右（正常，这正是去掉噪声后的真实水平），之后慢慢回到 0.99。
上面那个模型这样微调后：确定性到达 90.5% → 99.0%，位置误差中位 4.5 → 2.8 cm，平均 9.9 → 9.2 s。

“去掉噪声会不会变脆弱？”—— 这里的 std 是一个与状态无关的全局参数，只用于训练时探索，
不代表策略对某个状态有多“确定”，部署时也不用它。鲁棒性来自环境随机化（步态参数、定位噪声），
实测加 2 cm / 0.02 rad 定位噪声：第二阶段模型 81.7%，第三阶段 98.3%。
环境默认已带定位噪声 `loc_noise=(0.01, 0.01, 0.01)`（白噪声 m、每回合偏置 m、朝向 rad），
真机定位更差时可以调大后重训。

## 4. 在快速环境里评估（一两分钟）

比较模型时用同一个 `--seed`、至少 500 回合（默认就是 500），并看输出里的 95% 区间：两个模型的区间重叠就不算真的有差别（500 回合、98% 附近的区间约 96.4–98.9%）。评估默认带 1 cm 定位噪声（和训练环境一致）。

```bash
python scripts/eval_nav_aurora.py --policy checkpoints/my_nav_s3/last.pt --episodes 200
python scripts/eval_nav_aurora.py --policy checkpoints/my_nav_s3/best.pt --seed 11
python scripts/eval_nav_aurora.py --policy checkpoints/nav_aurora/best.pt --episodes 200   # 我训好的，对照
python scripts/eval_nav_aurora.py --policy follower --episodes 200                        # 几何跟踪器，对照
```
参考值（随机化模型）：到达 99%、位置误差中位 3.6 cm、倒退占路程 0.7%、平均 9.7 s。

## 5. 导出给 Aurora

```bash
python scripts/export_policy_npz.py checkpoints/my_nav_s3/last.pt      # → checkpoints/my_nav_s3/last.npz
```

## 6. 在 Aurora 全物理仿真里验收

```bash
# 宿主机：载入镜像（只做一次）。SDK zip 里只有一个 5 GB 的 fourier_aurora_sdk_gr1p:v1.2.1.tar；
# Windows 解压时文件名里的冒号可能被换成别的字符，按实际文件名填
docker load -i <解压出的 .tar 文件>
docker images                                     # 看载入后的镜像名
docker run -it --rm --net=host --ipc=host --cap-add=SYS_NICE ^
       -v "J:\robo@home\gr1_pickplace_rl:/workspace" -w /workspace <镜像名> bash

# 容器里（第一次）
pip install fourier_aurora_client==0.1.8
sed -i 's/UseJoyStick: true/UseJoyStick: false/' /opt/fftai/fourier_aurora/config/sys_config/config.yaml

# 每次验收
bash aurora_bridge/start_stack.sh --scene tabletop --spawn=-1.6,0,0 --qpos-log /tmp/q.log
python3 scripts/aurora_nav_eval.py --policy checkpoints/my_nav_s3/last.npz --tour 6 --out /tmp/my_nav.json
bash aurora_bridge/stop_stack.sh
```
最后一行 `SUMMARY` 就是结果。参考值（我训好的版本）：6/6 到达、平均误差 2.7 cm、平均 17 s、倒退合计 0.29 m、0 摔倒。
想看画面：`MUJOCO_GL=egl python3 scripts/render_aurora_video.py /tmp/q.log --speed 2 --out videos/my_nav.mp4`。

**Aurora 验收的注意事项**
- 至少 4 个 CPU 核给容器，验收时别同时训练或渲染视频（实时性不够机器人会无故摔倒，日志里会有大量 `time out`、`[sim] lag`）。
- 同时只能有一个 AuroraCore（`stop_stack.sh` 会清理残留）。

## 7. 结果不好时改哪里

| 现象 | 改什么 |
|---|---|
| 快速环境里好、Aurora 里冲过头 / 停不准 | 重新测步态：`python3 scripts/aurora_sysid.py --truth /tmp/truth.csv`（启动时加 `--truth-log /tmp/truth.csv`），把数字更新到 `aurora_bridge/gait_model.py` 的 `GaitParams`，或加大 `randomized()` 里的随机范围，再从第 2 步重训 |
| 还会倒着走 | `--back-penalty` 再调高（30–50），或把 `sim/nav_aurora_env.py` 的 `VX_BACK` 从 0.15 降到 0.12 |
| 横移太多、太慢 | `AuroraNavSkill.evaluate()` 里 `r_lat` 的系数从 1 调到 2–3 |
| 训练 success ≈99%，确定性评估只有 ~90% | 失败多半停在目标 6–9 cm 外、指令落在死区里：做第 3b 步（`--fix-std 0.1`） |
| 成功率上不去 | 先确认第 1 步老师成功率 ≈0.9；`success` 卡在 0 就别从零训练，一定从 BC 开始 |
| 想要更精确的停靠 | `evaluate()` 里成功条件 `dist < 0.06`、`SETTLE_STEPS`（要求停稳的步数） |

## 8. 操作策略（Aurora PdStand 版，GR1T2，站着不挪脚）

环境：`sim/manip_aurora_env.py`（三种子任务：抓放 / 抓起端持 / 放置，一个目标条件策略）。
和 Aurora 的对应关系（数字来自 `scripts/aurora_manip_sysid.py`，结果在 `docs/aurora_results/manip_sysid.json`）：

| 项目 | 快速环境 | Aurora 实测 |
|---|---|---|
| 骨盆 | mocap，按站姿摆：z = 0.836 + Δz，pitch = Δpitch，yaw = Δyaw；伸手时按质心补偿往后让 | 同左（误差 <2 mm）；质心补偿 1–4 cm |
| 腰 + 右臂 | 位置伺服，kp/kd = PdStand 配置，重力补偿 0.65–0.95 随机，静差 ±0.008 rad 随机 | 静差约 0.01 rad，末端偏差约 1.2 cm |
| 指令延迟 | 0.02–0.045 s 随机 | 阶跃响应 10% 前约 0.07 s 死区 |
| 抓取 | 末端在物体中心正上方 7 cm 内 3.5 cm、手已握紧 → weld 固定 | sim_server `grasp:` 同一规则（`AuroraManipRuntime`） |
| 端着物体 | IK 额外约束“物体竖直轴保持竖直”（`RightArmIK.solve(up_axis=…)`） | 同左 |
| 速度 | 400 步/s（单核，≈ 20 倍实时） | 实时 |

动作 7 维：末端目标增量（站位坐标系，每步 ≤1.5 cm）、抓取、站姿变化率（Δz、Δpitch、Δyaw）。

**先检查环境**：`python scripts/eval_manip_aurora.py --policy scripted --episodes 60` 应 ≥95%。
（2026-11-01 之前的版本在新版 MuJoCo（3.3 以后）里随机化物体质量会让抓住的物体数值爆炸，老师只有 ~47%、
学生学到的全是错的，训练成功率只有 10% 左右 —— 已修复，旧的 BC 模型要重新训练。）

```bash
# 1) 行为克隆（脚本老师 ScriptedManip，约 1–3 分钟，28 核会更快）
python scripts/pretrain_bc_manip_aurora.py --episodes 1500 --dagger 2 --out checkpoints/my_manip_bc/bc.pt   # 2 核约 35 分钟
# 2) 快速环境评估（老师 / 学生）
python scripts/eval_manip_aurora.py --policy scripted
python scripts/eval_manip_aurora.py --policy checkpoints/my_manip_bc/bc.pt
# 3) 导出 + Aurora 验收（容器里，桌子场景；每回合重新开机，9 回合约 7 分钟）
python scripts/export_policy_npz.py checkpoints/my_manip_bc/bc.pt
bash aurora_bridge/start_stack.sh --scene tabletop
python3 scripts/aurora_manip_eval.py --policy checkpoints/my_manip_bc/bc.npz --episodes 9 --out /tmp/manip.json
bash aurora_bridge/stop_stack.sh
```

`aurora_manip_eval.py` 是子任务的单元测试：每回合重新开机、按快速环境重新摆场景，“放置”模式开局直接把物体放进手里
（为了单独测放置）。**完整流程用连续任务**——一次开机、桌面只摆一次，之后状态一直连续：
每个物体 “走到抓取站位 → 用操作策略抓 → 托盘够不着就端着走过去 → 用操作策略放到托盘上的空位”，
手里的就是刚抓起来的那个物体，下一个物体从当前状态接着做：
```bash
bash aurora_bridge/start_stack.sh --scene tabletop --spawn=-1.6,0,0 --qpos-log /tmp/q.log
python3 scripts/aurora_manip_task.py --manip checkpoints/my_manip_bc/bc.npz --seed 1     # --layout default：用场景文件里的初始摆放
MUJOCO_GL=egl python3 scripts/render_aurora_video.py /tmp/q.log --speed 3 --out videos/my_manip_task.mp4
```
接着上一次运行继续（不重新摆物体；机器人还站着就连开机也跳过），已经在托盘上的物体会自动跳过：
```bash
python3 scripts/aurora_manip_task.py --objects red_block            # 第一次：摆场景（不加 --seed 就是 seed 1），只做 red_block
python3 scripts/aurora_manip_task.py --layout keep --resume         # 第二次：接着做，red_block 已在托盘上 → 跳过
```
注意：每次运行默认 `--layout random`，会按 seed 重新摆一遍（覆盖上次的结果）；要接着做必须加 `--layout keep`。
`--layout keep` 不加 `--resume`：机器人回出生点重新开机，但桌上物体保持原样。
参考（2026-11-02 新模型）：`--layout default` 3/3 放上托盘、0 摔倒、196 s；`--seed 1` 3/3、219 s
（`docs/aurora_results/manip_task_default.json`，视频 `videos/aurora_manip_task.mp4`）。
托盘上放第 2、3 个物体时（2026-11-02 修过一轮，旧视频里绿色罐子“消失”、蓝色盒子放置失败就是这些原因）：
- 托盘改成 24×24 cm（`config.py` 的 `tray_half`），3 个物体放在四角（离中心 6 cm、相距 12 cm）。16 cm 的旧托盘角上离边只剩 1 cm，放偏 2 cm 就从边上翻下去。
- 选位置：只选空着、桌面内、够得着的角，先放离机器人远的角（前臂不会压过已放好的物体）；都够不着时选空着里最近的，绝不放到已有物体上。
- 训练环境里一半回合托盘上已经有 1–2 个物体、目标在空角；老师改成“漏斗”式接近（离目标远时在安全高度平移，到正上方附近才下降），碰倒/推开别的物体扣分。
- 仿真（sim_server）：抓着的物体以前完全不碰撞，会穿进托盘上已放好的物体，松手后一恢复碰撞两者重叠 → 被弹飞几米；现在抓着时仍和桌子、托盘、其他物体碰撞（只是不和机器人碰），松手后要等物体离开机器人所有碰撞体才恢复和机器人的碰撞。
- 行为克隆加了 DAgger（`--dagger 2`，学生自己开、老师标注），学生在快速环境 94%（抓放 90 / 端持 100 / 放置 92）。

参考结果（子任务单元测试，同一组场景）：

| 策略 | 快速环境（300 回合） | Aurora 全物理（9 回合，3 种模式各 3） |
|---|---|---|
| 脚本老师 | 96.7%（90 回合） | **9/9**，0 摔倒，平均 2.9 s |
| BC 学生（旧版，无 DAgger、旧托盘） | 96.3%（抓放 92 / 端持 99 / 放置 98） | **8/9**，0 摔倒，平均 4.2 s |
| BC + DAgger 学生 `checkpoints/manip_aurora_bc/bc.npz`（当前默认，2026-11-02） | 94%（抓放 90 / 端持 100 / 放置 92，含托盘上已有物体的回合） | 连续任务 3 个布局各 3/3，0 摔倒 |
| BC + PPO 微调（模型已删除，命令见下） | 95.0%（抓放 94 / 端持 91 / 放置 100） | 6/9，0 摔倒 |

**PPO 微调（可选，目前没有带来提升）**：直接从 BC 接着训（和导航一样的参数）会把策略练坏（确定性成功率 99% → 72%），
因为 BC 只训练了演员、评论家是随机的，而且 PPO 一边采样一边改观测归一化。要微调请加这三个开关：
```bash
python scripts/train.py --task manip_aurora --resume checkpoints/my_manip_bc/bc.pt --lr 3e-5 --fix-std 0.15 \
       --freeze-obs-norm --critic-warmup 30 --steps 1000000 --num-envs 16 --run-name my_manip_s1
```
这样不会练坏（快速环境里 92–97% 来回），但也没有更好；到了 Aurora 里反而更差（6/9 对 8/9）——
PPO 学会了快速环境特有的“捷径”（更快地松手、端持时停得更靠边），在 Aurora 里不成立。
**部署先用 BC 学生**；想让 PPO 有用，先把快速环境和 Aurora 的差距再缩小（见下表），再微调。

| 现象 | 改什么 |
|---|---|
| Aurora 里放置时物体歪倒 / 掉出托盘 | 松手高度、`ScriptedManip` 里 lower 阶段的 +0.005；`placed` 判定的高度容差 |
| Aurora 里抓起端持 d_goal 总在 4.5 cm 左右（阈值 5 cm） | Aurora 的臂跟踪误差；可以放宽 `evaluate()` 里 `d_og < 0.05` 或在 `carry_point_local` 里补偿 |
| 想换物体 / 桌子 | `config.py` 的 `SceneCfg`，然后 `python -m sim.gr1t2_manip_scene` 重新生成 `assets/gr1t2/mjcf/manip_scene.xml` |
| Aurora 里手臂跟不上 / 下垂更多 | 重跑 `scripts/aurora_manip_sysid.py`，调 `gr1t2_manip_scene.build(gravcomp=…)` 和 `randomize()` 的范围 |
