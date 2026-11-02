"""
scripts/train.py —— 用 PPO 训练导航（nav_aurora），或对操作策略做可选的 PPO 微调（manip_aurora）
========================================================================================
用法：
  python scripts/train.py --task nav_aurora --resume checkpoints/nav_aurora_bc/bc.pt --lr 1e-4 \
         --steps 1000000 --back-penalty 8 --run-name nav_aurora_s1        # Aurora 步态版导航（详见 docs/rl_training_guide.md）
  python scripts/train.py --task nav_aurora --resume checkpoints/nav_aurora_s1/last.pt   # 断点续训
  python scripts/train.py --task manip_aurora --resume checkpoints/manip_aurora_bc/bc.pt --lr 3e-5 --fix-std 0.15 \
         --freeze-obs-norm --critic-warmup 30 --steps 1000000 --run-name manip_ppo   # 可选：操作策略 PPO 微调（目前没有提升）

训练过程写入 checkpoints/<task>/：
  progress.csv  每轮的回报、成功率、KL 等（可用 scripts/plot_progress.py 画曲线）
  best.pt       成功率最高的模型      last.pt  最新模型
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CKPT_DIR, PPOCfg  # noqa: E402
from rl.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402
# 注意：torch 只在 main() 里导入。子进程（仿真 worker）会重新导入本文件，
# 不导入 torch 可以让每个 worker 少占几百 MB 内存。


class make_env:
    """环境工厂：一个可被 pickle 的“无参构造函数”，子进程里调用它来创建环境。"""

    def __init__(self, task, **kw):
        self.task, self.kw = task, kw

    def __call__(self):
        sys.path.insert(0, ROOT)
        if self.task == "nav_aurora":
            from sim.nav_aurora_env import AuroraNavEnv
            return AuroraNavEnv(**self.kw)
        from sim.manip_aurora_env import AuroraManipEnv
        return AuroraManipEnv(**self.kw)


# 两个技能各自的 PPO 超参数（在 config.PPOCfg 默认值基础上覆盖）
TASK_CFG = {
    # Aurora 版：控制 10 Hz、回合更长（500 步 = 50 s），γ 取 0.995 让 50 s 尺度的回报仍有意义
    "nav_aurora": dict(total_steps=3_000_000, rollout_steps=256, epochs=5, minibatches=4,
                       lr=3e-4, ent_coef=0.0, gamma=0.995),
    # Aurora PdStand 版操作（GR1T2，20 Hz，回合 300 步 = 15 s）
    "manip_aurora": dict(total_steps=3_000_000, rollout_steps=512, epochs=5, minibatches=4,
                         lr=3e-4, ent_coef=0.0, gamma=0.99),
}
EXTRA_KEYS = {"nav_aurora": ("dist", "collision", "back_dist"),
              "manip_aurora": ("held", "d_goal", "mode", "arm_hit")}


def make_nav_aurora_eval(env_kw, episodes):
    """nav_aurora 的定期确定性评估：固定的一组回合（同样的起点/目标/步态参数），分数 = 到达率 − 用时惩罚。"""
    import numpy as np
    from sim.nav_aurora_env import AuroraNavEnv
    env = AuroraNavEnv(seed=12345, **env_kw)

    def eval_fn(act):
        ok, steps = 0, []
        for ep in range(episodes):
            o, _ = env.reset(seed=12345000 + ep)
            for t in range(env.max_steps):
                o, r, term, trunc, info = env.step(act(o))
                if term or trunc:
                    break
            ok += info["success"]
            steps.append(t + 1)
        rate = ok / episodes
        return rate - 1e-4 * float(np.mean(steps)), f"到达 {rate * 100:.1f}%（{episodes} 回合），平均 {np.mean(steps) / 10:.1f} s"
    return eval_fn


def make_manip_aurora_eval(env_kw, episodes):
    """manip_aurora 的定期确定性评估：固定的一组回合（三种模式各 1/3），分数 = 成功率 − 用时惩罚。"""
    import numpy as np
    from sim.manip_aurora_env import AuroraManipEnv
    env = AuroraManipEnv(seed=12345, **env_kw)

    def eval_fn(act):
        ok, steps, per = 0, [], {0: [0, 0], 1: [0, 0], 2: [0, 0]}
        for ep in range(episodes):
            o, info0 = env.reset(seed=12345000 + ep, options={"mode": ep % 3})
            for t in range(env.max_steps):
                o, r, term, trunc, info = env.step(act(o))
                if term or trunc:
                    break
            ok += info["success"]
            per[ep % 3][0] += info["success"]; per[ep % 3][1] += 1
            steps.append(t + 1)
        rate = ok / episodes
        msg = "，".join(f"{n} {per[k][0]}/{per[k][1]}" for k, n in ((0, "抓放"), (1, "抓起端持"), (2, "放置")))
        return rate - 1e-4 * float(np.mean(steps)), f"成功 {rate * 100:.1f}%（{msg}），平均 {np.mean(steps) * 0.05:.1f} s"
    return eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["nav_aurora", "manip_aurora"], required=True)
    ap.add_argument("--num-envs", type=int, default=os.cpu_count())
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None, help="学习率（微调时建议调小，如 1e-4）")
    ap.add_argument("--run-name", default=None, help="检查点子目录名（默认=任务名）")
    ap.add_argument("--debug", action="store_true", help="单进程运行，方便调试")
    ap.add_argument("--eval-every", type=int, default=10, help="每多少轮做一次确定性评估（nav_aurora / manip_aurora）")
    ap.add_argument("--eval-episodes", type=int, default=100, help="确定性评估的回合数")
    ap.add_argument("--fix-std", type=float, default=None,
                    help="把策略的探索标准差固定为该值（不再学习）。收尾微调用：std 小了，策略不能再靠探索噪声"
                         "“碰巧”越过 Aurora 的速度死区，确定性策略（部署时用的）才会自己发出够大的指令")
    ap.add_argument("--back-penalty", type=float, default=None,
                    help="仅 nav_aurora：倒退代价（每米），第一阶段 8、第二阶段 20（默认 20）")
    ap.add_argument("--freeze-obs-norm", action="store_true",
                    help="不再更新观测归一化（从行为克隆初始化微调时推荐，见 rl/ppo.py）")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="前 N 轮只训练价值网络、演员不动（从行为克隆初始化微调时推荐 20–50）")
    args = ap.parse_args()

    import torch
    from rl.ppo import PPO
    torch.set_num_threads(1)      # 仿真子进程占满 CPU，主进程的网络推理用 1 个线程即可
    if args.task == "manip_aurora":       # 先在主进程生成 GR1T2 场景文件，子进程直接加载（避免同时写文件）
        from sim import gr1t2_manip_scene
        gr1t2_manip_scene.build()
    cfg = PPOCfg(**TASK_CFG[args.task])
    cfg.num_envs, cfg.seed = args.num_envs, args.seed
    if args.steps:
        cfg.total_steps = args.steps
    if args.lr:
        cfg.lr = args.lr
    kw = {}
    if args.task == "nav_aurora" and args.back_penalty is not None:
        kw["back_penalty"] = args.back_penalty
    fns = [make_env(args.task, **kw) for _ in range(cfg.num_envs)]
    envs = (DummyVecEnv if args.debug else SubprocVecEnv)(fns, base_seed=args.seed * 1000)

    run_dir = os.path.join(CKPT_DIR, args.run_name or args.task)
    agent = PPO(envs, cfg, device=args.device, run_dir=run_dir)
    if args.resume:
        agent.load(args.resume)
        print("从检查点继续:", args.resume, "已训练步数", agent.global_step)
    agent.freeze_obs_rms = args.freeze_obs_norm
    agent.critic_warmup = args.critic_warmup
    if args.fix_std is not None:
        import math
        with torch.no_grad():
            agent.ac.log_std.fill_(math.log(args.fix_std))
        agent.ac.log_std.requires_grad_(False)
        print(f"探索标准差固定为 {args.fix_std}")
    print(f"任务={args.task} 观测维度={agent.obs_dim} 动作维度={agent.act_dim} "
          f"并行环境={cfg.num_envs} 设备={agent.device}")
    try:
        eval_fn = {"nav_aurora": make_nav_aurora_eval, "manip_aurora": make_manip_aurora_eval}.get(args.task)
        eval_fn = eval_fn(kw, args.eval_episodes) if eval_fn else None
        agent.learn(cfg.total_steps, extra_info_keys=EXTRA_KEYS[args.task], eval_fn=eval_fn,
                    eval_every=args.eval_every)
    finally:
        agent.save(os.path.join(run_dir, "last.pt"))
        envs.close()


if __name__ == "__main__":
    main()
