"""
rl/ppo.py —— 从零实现的 PPO (Proximal Policy Optimization)
===========================================================

PPO 一轮迭代 = 两个阶段：
  ① 收集 (rollout)：用当前策略在 N 个并行环境里各跑 T 步，存下 (s, a, r, logπ(a|s), V(s))
  ② 更新 (update) ：用这批数据做若干轮小批量梯度下降，目标函数：
        L = −E[min(ρ·Â, clip(ρ, 1−ε, 1+ε)·Â)]  + c_v·(V − R)²  − c_e·熵
        ρ = π_new(a|s) / π_old(a|s)   （新旧策略概率比）
        Â = GAE 优势估计：这个动作比“平均水平”好多少
     “clip” 限制每次更新别离旧策略太远 —— 这就是 PPO 稳定的关键。

实现中的实用技巧（每条都会显著影响训练成败）：
  * 观测归一化（运行均值/方差）       —— 不同量纲的输入（米、弧度、速度）拉到同一尺度
  * 奖励缩放（按回报标准差）           —— 让价值网络的目标数值稳定
  * 优势归一化                         —— 每个 minibatch 内零均值单位方差
  * 时间截断的价值自举 (bootstrap)     —— 回合是“被时间砍断”的，不是真的结束
  * 学习率线性衰减、梯度裁剪、KL 早停
"""
import csv
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from config import PPOCfg


# ----------------------------------------------------------------------------
class RunningMeanStd:
    """在线估计均值和方差（Welford/并行合并公式）。"""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


# ----------------------------------------------------------------------------
def mlp(inp, hidden, out, out_gain=1.0):
    layers, last = [], inp
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2)); nn.init.zeros_(lin.bias)
        layers += [lin, nn.ELU()]
        last = h
    head = nn.Linear(last, out)
    nn.init.orthogonal_(head.weight, out_gain); nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


class ActorCritic(nn.Module):
    """演员-评论家 (Actor-Critic)：
    * Actor（策略网络）：输入观测，输出高斯分布的均值 μ；标准差 σ 是可学习参数。
    * Critic（价值网络）：输入观测，输出 V(s) —— “从这里开始，预计还能拿多少回报”。
    两个网络分开（不共享层），对连续控制任务通常更稳。"""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256), init_log_std=-0.5):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, act_dim, out_gain=0.01)   # 初始输出≈0，动作温和
        self.critic = mlp(obs_dim, hidden, 1, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, obs):
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


# ----------------------------------------------------------------------------
class PPO:
    def __init__(self, envs, cfg: PPOCfg, device=None, run_dir="runs/ppo"):
        self.envs, self.cfg = envs, cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_dim = envs.observation_space.shape[0]
        self.act_dim = envs.action_space.shape[0]
        torch.manual_seed(cfg.seed)
        self.ac = ActorCritic(self.obs_dim, self.act_dim, cfg.hidden, cfg.init_log_std).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.ret_rms = RunningMeanStd(())
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.global_step = 0
        # 从行为克隆初始化微调时有用的两个开关（见 scripts/train.py --freeze-obs-norm / --critic-warmup）：
        #  * freeze_obs_rms：不再更新观测归一化。BC 的归一化统计来自老师数据，PPO 一边采样一边改它，
        #    等于悄悄改变了策略网络的输入 → 刚接手的策略会“变形”。
        #  * critic_warmup：前 N 轮只训练价值网络（BC 只训练了演员，评论家是随机的，优势估计全是噪声，
        #    这时候更新演员只会把 BC 学到的东西冲掉）。
        self.freeze_obs_rms = False
        self.critic_warmup = 0

    # ------------------------------------------------------------------
    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act_deterministic(self, obs):
        """当前网络的确定性动作（取高斯均值），给训练中的定期评估用。"""
        o = torch.as_tensor(self.obs_rms.normalize(obs).astype(np.float32), device=self.device)[None]
        return np.clip(self.ac.actor(o)[0].cpu().numpy(), -1, 1)

    def learn(self, total_steps, log_every=1, save_every=20, extra_info_keys=(), eval_fn=None, eval_every=10):
        """eval_fn(act_fn) -> (分数, 说明文字)：每 eval_every 轮用确定性策略评估一次，分数最高的存为 best.pt。
        没有 eval_fn 时（如 manip），best.pt 退回旧规则：训练日志 success（带探索噪声、最近 100 回合）最高的一版。
        旧规则的问题：success 一旦到 1.0 就再也超不过，best.pt 会停在第一次到 1.0 的那一版（通常很早），
        而且带噪声的成功率和部署时（确定性）的表现可能差很多。"""
        cfg, envs, n, T = self.cfg, self.envs, self.envs.n, self.cfg.rollout_steps
        obs_buf = np.zeros((T, n, self.obs_dim), np.float32)
        act_buf = np.zeros((T, n, self.act_dim), np.float32)
        logp_buf = np.zeros((T, n), np.float32)
        rew_buf = np.zeros((T, n), np.float32)
        done_buf = np.zeros((T, n), np.float32)
        val_buf = np.zeros((T, n), np.float32)

        obs = envs.reset()
        ep_ret, ep_len = np.zeros(n), np.zeros(n, int)
        disc_ret = np.zeros(n)
        recent = deque(maxlen=100)          # 最近 100 个回合的统计
        n_iters = int(np.ceil(total_steps / (n * T)))
        best_success = -1.0
        best_eval = -np.inf
        log_f = open(os.path.join(self.run_dir, "progress.csv"), "w", newline="")
        writer = csv.writer(log_f)
        writer.writerow(["iter", "steps", "fps", "ep_return", "ep_len", "success",
                         "pi_loss", "v_loss", "entropy", "kl", "std"] + list(extra_info_keys))
        t_start, step0 = time.time(), self.global_step

        for it in range(1, n_iters + 1):
            # 学习率线性衰减到 0
            frac = 1.0 - (it - 1) / n_iters
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr * frac

            # ============ ① 收集数据 ============
            t_roll = time.time()
            self.ac.eval()
            for t in range(T):
                if not self.freeze_obs_rms:
                    self.obs_rms.update(obs)
                o_n = self.obs_rms.normalize(obs).astype(np.float32)
                with torch.no_grad():
                    ot = self._t(o_n)
                    d = self.ac.dist(ot)
                    a = d.sample()
                    logp = d.log_prob(a).sum(-1)
                    v = self.ac.value(ot)
                a_np = a.cpu().numpy()
                next_obs, r, term, trunc, infos = envs.step(np.clip(a_np, -1, 1))

                # 奖励缩放：除以“折扣回报”的标准差
                disc_ret = disc_ret * cfg.gamma + r
                self.ret_rms.update(disc_ret)
                r_scaled = r / np.sqrt(self.ret_rms.var + 1e-8)

                # 时间截断：用 V(最后观测) 自举，相当于“假装回合还在继续”
                for i in np.where(trunc & ~term)[0]:
                    fo = self.obs_rms.normalize(infos[i]["final_obs"]).astype(np.float32)
                    with torch.no_grad():
                        r_scaled[i] += cfg.gamma * self.ac.value(self._t(fo[None]))[0].item()

                obs_buf[t], act_buf[t], logp_buf[t] = o_n, a_np, logp.cpu().numpy()
                val_buf[t], rew_buf[t] = v.cpu().numpy(), r_scaled
                done = term | trunc
                done_buf[t] = done

                ep_ret += r; ep_len += 1
                for i in np.where(done)[0]:
                    rec = {"ret": ep_ret[i], "len": ep_len[i],
                           "success": float(infos[i].get("success", False))}
                    for k in extra_info_keys:
                        rec[k] = float(infos[i].get(k, np.nan))
                    recent.append(rec)
                    ep_ret[i], ep_len[i], disc_ret[i] = 0, 0, 0
                obs = next_obs
            self.global_step += n * T

            # ============ GAE 优势估计 ============
            with torch.no_grad():
                last_v = self.ac.value(self._t(self.obs_rms.normalize(obs).astype(np.float32))).cpu().numpy()
            adv = np.zeros((T, n), np.float32)
            gae = np.zeros(n, np.float32)
            for t in reversed(range(T)):
                next_v = last_v if t == T - 1 else val_buf[t + 1]
                nonterminal = 1.0 - done_buf[t]
                delta = rew_buf[t] + cfg.gamma * next_v * nonterminal - val_buf[t]
                gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
                adv[t] = gae
            ret = adv + val_buf

            t_roll = time.time() - t_roll
            t_upd = time.time()
            # ============ ② 更新网络 ============
            self.ac.train()
            b_obs = self._t(obs_buf.reshape(-1, self.obs_dim))
            b_act = self._t(act_buf.reshape(-1, self.act_dim))
            b_logp = self._t(logp_buf.reshape(-1))
            b_adv = self._t(adv.reshape(-1))
            b_ret = self._t(ret.reshape(-1))
            batch = n * T
            mb = batch // cfg.minibatches
            stats = []
            stop = False
            for epoch in range(cfg.epochs):
                perm = torch.randperm(batch, device=self.device)
                for s in range(0, batch, mb):
                    idx = perm[s:s + mb]
                    d = self.ac.dist(b_obs[idx])
                    logp = d.log_prob(b_act[idx]).sum(-1)
                    ratio = (logp - b_logp[idx]).exp()
                    a_mb = b_adv[idx]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    pi_loss = -torch.min(ratio * a_mb,
                                         ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * a_mb).mean()
                    v_loss = 0.5 * (self.ac.value(b_obs[idx]) - b_ret[idx]).pow(2).mean()
                    ent = d.entropy().sum(-1).mean()
                    actor_on = 0.0 if it <= self.critic_warmup else 1.0
                    loss = actor_on * (pi_loss - cfg.ent_coef * ent) + cfg.vf_coef * v_loss
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                    self.opt.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - (logp - b_logp[idx])).mean().item()   # 近似 KL
                    stats.append((pi_loss.item(), v_loss.item(), ent.item(), kl))
                    if cfg.target_kl and kl > 1.5 * cfg.target_kl:
                        stop = True
                        break
                if stop:
                    break

            t_upd = time.time() - t_upd
            # ============ 日志与保存 ============
            fps = (self.global_step - step0) / (time.time() - t_start)
            st = np.mean(stats, 0)
            if recent:
                mret = np.mean([e["ret"] for e in recent])
                mlen = np.mean([e["len"] for e in recent])
                succ = np.mean([e["success"] for e in recent])
                extra = [np.nanmean([e[k] for e in recent]) for k in extra_info_keys]
            else:
                mret = mlen = succ = np.nan
                extra = [np.nan] * len(extra_info_keys)
            std = self.ac.log_std.exp().mean().item()
            writer.writerow([it, self.global_step, int(fps), mret, mlen, succ, *st, std, *extra])
            log_f.flush()
            if it % log_every == 0:
                print(f"[{it:4d}/{n_iters}] steps={self.global_step:>9d} fps={fps:6.0f} "
                      f"return={mret:7.2f} len={mlen:5.1f} success={succ:5.2f} "
                      f"kl={st[3]:.4f} std={std:.3f} (采样{t_roll:.1f}s/更新{t_upd:.1f}s)", flush=True)
            if it % save_every == 0 or it == n_iters:
                self.save(os.path.join(self.run_dir, "last.pt"))
            if eval_fn is None and len(recent) >= 50 and succ > best_success:
                best_success = succ
                self.save(os.path.join(self.run_dir, "best.pt"))
            if eval_fn is not None and (it % eval_every == 0 or it == n_iters):
                score, msg = eval_fn(self.act_deterministic)
                tag = ""
                if score > best_eval:
                    best_eval = score
                    self.save(os.path.join(self.run_dir, "best.pt"))
                    tag = "  → 存为 best.pt"
                print(f"       [确定性评估] {msg}{tag}", flush=True)
                with open(os.path.join(self.run_dir, "eval.csv"), "a") as ef:
                    ef.write(f"{it},{self.global_step},{score:.4f}\n")
        log_f.close()

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save({"model": self.ac.state_dict(), "obs_rms": self.obs_rms.state_dict(),
                    "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                    "hidden": self.cfg.hidden, "steps": self.global_step}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(ck["model"])
        self.obs_rms.load_state_dict(ck["obs_rms"])
        self.global_step = ck.get("steps", 0)


class Policy:
    """部署用：加载训练好的检查点，输入原始观测，输出动作（不再需要环境、优化器）。"""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.ac = ActorCritic(ck["obs_dim"], ck["act_dim"], tuple(ck["hidden"]))
        self.ac.load_state_dict(ck["model"])
        self.ac.eval()
        self.rms = RunningMeanStd((ck["obs_dim"],))
        self.rms.load_state_dict(ck["obs_rms"])

    @torch.no_grad()
    def __call__(self, obs, deterministic=True):
        o = torch.as_tensor(self.rms.normalize(obs).astype(np.float32))[None]
        d = self.ac.dist(o)
        a = d.mean if deterministic else d.sample()
        return np.clip(a[0].numpy(), -1, 1)
