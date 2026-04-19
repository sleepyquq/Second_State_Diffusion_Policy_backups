"""
Diffusion Policy 训练脚本（State 模式）
============================================================
功能：从 HDF5 演示数据训练 Diffusion Policy，输出 eval 推力模型。

任务：PickCube-v1 / PegInsertionSide-v1
观测：状态向量（obs_dim=42/43）
动作：8维连续动作（Box(-1,1)）

使用方法：
  python train_dp.py --data demos_PickCube-v1_state_random.h5 --env PickCube-v1 --epochs 100

输出：checkpoints/policy_epoch_*.pt
"""

import argparse
import csv
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from collect_demos import DemoDataset, obs_to_numpy

# ═══════════════════════════════════════════════════════════════
# 设备自动选择（无 GPU 时回退 CPU）
# ═══════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Diffusion Policy] 使用设备: {DEVICE}")
if DEVICE == "cpu":
    print("[提示] 运行在 CPU 上，建议使用 GPU 加速。Windows 无 GPU 时可考虑 Google Colab。")


# ═══════════════════════════════════════════════════════════════
# 1. 噪声调度器（DDPM + Cosine Schedule）
# ═══════════════════════════════════════════════════════════════

class DDPMScheduler:
    """DDPM 噪声调度器，支持 cosine 噪声调度。"""

    def __init__(self, num_timesteps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, schedule: str = "cosine"):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            # Cosine schedule from Improved DDPM
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            alphas = torch.cos((steps + 0.008) / 1.008 * (torch.pi / 2)) ** 2
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """轻量 buffer 注册；调度器不是 nn.Module，只需保存张量属性。"""
        setattr(self, name, tensor)

    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        """对 x_start 在时刻 t 添加噪声：x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_start)[t]
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod.to(x_start)[t]
        # broadcast
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, *([1] * (x_start.ndim - 1)))
        sqrt_one_minus_t = sqrt_one_minus_t.view(-1, *([1] * (x_start.ndim - 1)))
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_t * noise

    def get_losses(self, model, x_start: torch.Tensor, obs: torch.Tensor,
                   action_dim: int, noise: torch.Tensor = None):
        """计算 DDPM 损失：E[||ε - ε_θ(x_t, t, c)||²]"""
        B, Tp, Da = x_start.shape
        if noise is None:
            noise = torch.randn_like(x_start)

        t = torch.randint(0, self.num_timesteps, (B,), device=x_start.device)
        x_t = self.add_noise(x_start, noise, t)
        # obs: (B, To, Do) → 拼接为条件
        cond = obs.flatten(1)  # (B, To*Do)
        # 预测噪声
        noise_pred = model(x_t, t, cond)  # (B, Tp, Da)
        return F.mse_loss(noise_pred, noise), noise_pred

    @torch.no_grad()
    def sample(self, model, obs: torch.Tensor, pred_horizon: int,
               action_dim: int, num_inference_steps: int = None):
        """DDIM 反向采样：从纯噪声生成动作序列。

        训练仍使用 DDPM 噪声目标；评估时用确定性的 DDIM 更新可以减少
        推理步数，并避免手写 DDPM 后验方差在 cosine schedule 下出现负数开方。
        """
        steps = int(num_inference_steps or self.num_timesteps)
        steps = max(1, min(steps, self.num_timesteps))
        indices = torch.linspace(
            self.num_timesteps - 1,
            0,
            steps,
            dtype=torch.long,
        ).tolist()

        # 评估时传入的 obs 通常是 (B, To, obs_dim)，这里统一展平成条件向量。
        obs_flat = obs.flatten(1)
        B = obs_flat.shape[0]

        # 从纯噪声开始
        x_t = torch.randn(B, pred_horizon, action_dim, device=obs.device)

        for step_idx, i in enumerate(indices):
            prev_i = indices[step_idx + 1] if step_idx + 1 < len(indices) else -1
            t = torch.full((B,), i, device=obs.device, dtype=torch.long)
            noise_pred = model(x_t, t, obs_flat)
            alpha_bar = self.alphas_cumprod[i].to(obs.device)

            pred_x0 = (
                x_t - (1 - alpha_bar).sqrt() * noise_pred
            ) / alpha_bar.sqrt()
            # 动作在训练时已经做 Gaussian 标准化，不能按原始动作范围裁剪。
            # 用较宽的保护范围避免极端发散即可，最终动作会在反标准化后 clip 到环境范围。
            pred_x0 = pred_x0.clamp(-5, 5)

            if prev_i < 0:
                x_t = pred_x0
            else:
                alpha_bar_prev = self.alphas_cumprod[prev_i].to(obs.device)
                x_t = (
                    alpha_bar_prev.sqrt() * pred_x0
                    + (1 - alpha_bar_prev).sqrt() * noise_pred
                )

        return x_t.clamp(-5, 5)


# ═══════════════════════════════════════════════════════════════
# 2. Diffusion Policy 网络
# ═══════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码，将整数 t 映射到 d 维向量。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualMLP(nn.Module):
    """带残差连接的 MLP Block。"""

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ConditionalUnet1D(nn.Module):
    """
    1D Conditional U-Net，用于预测噪声。

    输入：
      - x: (B, Tp, Da)  含噪动作序列
      - t: (B,)  时间步
      - c: (B, Do)  观测条件

    输出：(B, Tp, Da)  预测的噪声
    """

    def __init__(self, action_dim: int, obs_dim: int, pred_horizon: int,
                 hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_dim = obs_dim

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 观测条件编码
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 输入投影：动作 + 时间 + 观测
        input_dim = action_dim + hidden_dim + hidden_dim  # x + t_emb + obs_emb
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, pred_horizon, hidden_dim))

        # 中间层（残差 MLP）
        self.layers = nn.ModuleList([
            ResidualMLP(hidden_dim, hidden_mult=4, dropout=0.1)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # 自适应层归一化（AdaLN）：每层根据条件调整
        self.ada_ln = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 2) for _ in range(num_layers)
        ])

    def _apply_cond(self, x: torch.Tensor, cond: torch.Tensor, layer_idx: int):
        """对特征 x 应用自适应条件调制（AdaLN）。"""
        scale, shift = self.ada_ln[layer_idx](cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # (B, 1, H)
        shift = shift.unsqueeze(1)
        return x * (scale + 1) + shift

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                obs_cond: torch.Tensor) -> torch.Tensor:
        B, Tp, Da = x.shape

        # 1. 时间嵌入
        t_emb = self.time_mlp(t)               # (B, H)

        # 2. 观测嵌入
        obs_emb = self.obs_mlp(obs_cond)        # (B, H)

        # 3. 条件组合
        cond = torch.cat([t_emb, obs_emb], dim=-1)  # (B, 2H)

        # 4. 输入投影并拼接
        t_emb_bt = t_emb.unsqueeze(1).expand(-1, Tp, -1)  # (B, Tp, H)
        obs_emb_bt = obs_emb.unsqueeze(1).expand(-1, Tp, -1)

        h = torch.cat([x, t_emb_bt, obs_emb_bt], dim=-1)  # (B, Tp, Da+2H)
        h = self.input_proj(h)                             # (B, Tp, H)
        h = h + self.pos_emb[:, :Tp]

        # 5. 通过残差层
        for i, layer in enumerate(self.layers):
            h = self._apply_cond(h, cond, i)
            h = layer(h)

        # 6. 输出噪声预测
        return self.output_proj(h)  # (B, Tp, Da)


class GaussianNormalizer:
    """对观测/动作做均值方差归一化（基于数据统计）。"""

    def __init__(self, obs_mean: np.ndarray, obs_std: np.ndarray,
                 action_mean: np.ndarray, action_std: np.ndarray):
        self.obs_mean = torch.from_numpy(obs_mean).float()
        self.obs_std = torch.from_numpy(obs_std).float().clamp(min=1e-8)
        self.action_mean = torch.from_numpy(action_mean).float()
        self.action_std = torch.from_numpy(action_std).float().clamp(min=1e-8)
        self.action_dim = int(self.action_mean.shape[-1])

    def normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.obs_mean.to(x)) / self.obs_std.to(x)

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.action_mean.to(x)) / self.action_std.to(x)

    def denormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.action_std.to(x) + self.action_mean.to(x)


# ═══════════════════════════════════════════════════════════════
# 3. EMA（指数移动平均）
# ═══════════════════════════════════════════════════════════════

class EMA:
    """指数移动平均，用于稳定 Diffusion Policy 训练。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """将 EMA 权重应用回模型（用于评估）。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """恢复原始权重。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()


# ═══════════════════════════════════════════════════════════════
# 4. 评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_policy(env_id: str, policy_net: nn.Module, scheduler: DDPMScheduler,
                    normalizer: GaussianNormalizer, obs_horizon: int,
                    pred_horizon: int, num_episodes: int = 10,
                    obs_mode: str = "state", device: str = "cpu",
                    render_backend: str = "gpu",
                    num_inference_steps: int = 20,
                    max_episode_steps: int | None = None,
                    action_horizon: int = 8) -> dict:
    """在真实环境中评估训练好的策略，返回成功率。"""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if obs_mode != "state":
        raise NotImplementedError("当前 train_dp.py 只支持 state 模式评估；RGBD 需先实现视觉 obs_encoder。")

    is_windows = platform.system() == "Windows"
    needs_render = obs_mode not in ("state", "state_dict", "none")
    rb = "sapien_cpu" if (is_windows and needs_render) else render_backend

    env_kwargs = {}
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_id, obs_mode=obs_mode, render_mode="rgb_array",
                   render_backend=rb, **env_kwargs)
    policy_net.eval()

    def _extract_state_obs(raw_obs) -> np.ndarray:
        """从 ManiSkill 返回值中取 state 观测，并转换为一维 numpy 数组。"""
        state_obs = raw_obs["state"] if isinstance(raw_obs, dict) else raw_obs
        state_obs = obs_to_numpy(state_obs)
        return np.asarray(state_obs, dtype=np.float32).reshape(-1)

    successes = []
    rewards = []
    steps_taken = []
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep)
            done = False
            step = 0

            # 收集 obs_horizon 帧历史
            first_obs = _extract_state_obs(obs)
            obs_history = [first_obs.copy() for _ in range(obs_horizon)]

            obs_seq = np.stack(obs_history[-obs_horizon:], axis=0)  # (To, obs_dim)
            done = False
            total_reward = 0.0

            rollout_limit = max_episode_steps or 500
            while not done and step < rollout_limit:
                # 标准化
                obs_t = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                obs_t = normalizer.normalize_obs(obs_t)

                # 预测动作序列
                action_seq = scheduler.sample(
                    policy_net, obs_t, pred_horizon, normalizer.action_dim,
                    num_inference_steps=num_inference_steps,
                )
                action_seq = normalizer.denormalize_action(action_seq)

                # 执行前 action_horizon 个动作，再基于新观测重规划。
                exec_horizon = min(action_horizon, pred_horizon)
                for action_i in range(exec_horizon):
                    action_np = action_seq[0, action_i].cpu().numpy()
                    action_np = np.clip(action_np, -1, 1)
                    obs, reward, terminated, truncated, info = env.step(action_np)
                    total_reward += float(reward)
                    done = terminated or truncated or info.get("success", False)
                    step += 1

                    obs_history.append(_extract_state_obs(obs))
                    obs_history.pop(0)
                    obs_seq = np.stack(obs_history, axis=0)

                    if done or step >= rollout_limit:
                        break

            successes.append(float(info.get("success", False)))
            rewards.append(total_reward)
            steps_taken.append(step)
            print(f"  Eval ep {ep+1}/{num_episodes}: "
                  f"{'成功' if successes[-1] else '失败'} | 步数={step} | 奖励={total_reward:.2f}")

    env.close()
    sr = np.mean(successes) * 100
    return {
        "success_rate": sr,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_steps": float(np.mean(steps_taken)) if steps_taken else 0.0,
        "n_episodes": num_episodes,
    }


def write_metrics_csv(metrics: list[dict], path: Path) -> None:
    """写出训练曲线原始数据，便于后续画图和报告引用。"""
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "global_step", "train_loss", "lr",
        "eval_success_rate", "eval_mean_reward", "eval_mean_steps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            writer.writerow({k: row.get(k, "") for k in fields})


def save_training_curves(metrics: list[dict], path: Path) -> None:
    """保存 loss / success / reward 曲线图。"""
    if not metrics:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[曲线] matplotlib 不可用，跳过绘图：{exc}")
        return

    epochs = [m["epoch"] for m in metrics]
    losses = [m["train_loss"] for m in metrics]
    success_epochs = [m["epoch"] for m in metrics if m.get("eval_success_rate") is not None]
    success = [m["eval_success_rate"] for m in metrics if m.get("eval_success_rate") is not None]
    rewards = [m["eval_mean_reward"] for m in metrics if m.get("eval_mean_reward") is not None]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, losses, marker="o", linewidth=1.5)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(success_epochs, success, marker="o", linewidth=1.5)
    axes[1].set_title("Eval Success Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(success_epochs, rewards, marker="o", linewidth=1.5)
    axes[2].set_title("Eval Mean Reward")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 5. 主训练循环
# ═══════════════════════════════════════════════════════════════

def train(args):
    # ── 加载数据 ──────────────────────────────────────────────
    dataset = DemoDataset(args.data, obs_horizon=args.obs_horizon,
                          pred_horizon=args.pred_horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE == "cuda"))

    if args.obs_dim is None:
        args.obs_dim = int(dataset.episodes[0]["obs"].shape[-1])

    print(f"[数据] 样本数={len(dataset)}, obs_dim={args.obs_dim}, action_dim={args.action_dim}")
    print(f"[数据] 轨迹数={len(dataset.episodes)}")

    # ── 统计归一化参数 ────────────────────────────────────────
    all_obs = np.concatenate([ep["obs"] for ep in dataset.episodes], axis=0)
    all_act = np.concatenate([ep["actions"] for ep in dataset.episodes], axis=0)

    obs_mean = all_obs.mean(axis=0)
    obs_std = all_obs.std(axis=0)
    action_mean = all_act.mean(axis=0)
    action_std = all_act.std(axis=0)

    normalizer = GaussianNormalizer(obs_mean, obs_std, action_mean, action_std)
    print(f"[归一化] 动作均值={action_mean[:3]}..., 标准差={action_std[:3]}...")

    # ── 模型 & 调度器 ──────────────────────────────────────────
    scheduler = DDPMScheduler(num_timesteps=args.num_timesteps, schedule=args.schedule)

    policy_net = ConditionalUnet1D(
        action_dim=args.action_dim,
        obs_dim=args.obs_dim * args.obs_horizon,
        pred_horizon=args.pred_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(DEVICE)

    ema = EMA(policy_net, decay=args.ema_decay)
    print(f"[模型] 参数总数={sum(p.numel() for p in policy_net.parameters()):,}")

    # ── 优化器 & 学习率调度 ──────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy_net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler_lr = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * len(loader),
        pct_start=0.1,
    )

    # ── 训练循环 ─────────────────────────────────────────────
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_dir) / "metrics.csv"
    curves_path = Path(args.output_dir) / "training_curves.png"

    global_step = 0
    best_sr = 0.0
    metrics = []

    for epoch in range(args.epochs):
        policy_net.train()
        epoch_losses = []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for obs_seq, action_seq in pbar:
            obs_seq = obs_seq.to(DEVICE)    # (B, To, obs_dim)
            action_seq = action_seq.to(DEVICE)  # (B, Tp, action_dim)

            # 归一化
            obs_seq = normalizer.normalize_obs(obs_seq)
            action_seq_norm = normalizer.normalize_action(action_seq)

            # 展平观测为条件向量
            obs_cond = obs_seq.flatten(1)    # (B, To*obs_dim)

            # DDPM 损失
            loss, _ = scheduler.get_losses(policy_net, action_seq_norm, obs_cond,
                                            args.action_dim)
            # action_dim for normalizer fixup

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            scheduler_lr.step()
            ema.update()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        avg_loss = np.mean(epoch_losses)
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": float(avg_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "eval_success_rate": None,
            "eval_mean_reward": None,
            "eval_mean_steps": None,
        }

        # ── 每 N 轮评估一次 ─────────────────────────────────
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            ema.apply_shadow()  # 使用 EMA 权重评估
            policy_net.eval()

            print(f"\n[Epoch {epoch+1}] loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
            print(f"[评估] 在 {args.env} 上运行...")

            eval_result = evaluate_policy(
                env_id=args.env,
                policy_net=policy_net,
                scheduler=scheduler,
                normalizer=normalizer,
                obs_horizon=args.obs_horizon,
                pred_horizon=args.pred_horizon,
                num_episodes=args.eval_episodes,
                obs_mode=args.obs_mode,
                device=DEVICE,
                num_inference_steps=args.inference_steps,
                max_episode_steps=args.eval_max_steps,
                action_horizon=args.action_horizon,
            )

            sr = eval_result["success_rate"]
            row["eval_success_rate"] = float(sr)
            row["eval_mean_reward"] = float(eval_result["mean_reward"])
            row["eval_mean_steps"] = float(eval_result["mean_steps"])
            print(
                f"[Epoch {epoch+1}] 成功率: {sr:.1f}% | "
                f"平均奖励: {eval_result['mean_reward']:.2f} | "
                f"平均步数: {eval_result['mean_steps']:.1f}"
            )

            # 保存最佳模型
            if sr >= best_sr:
                best_sr = sr
                best_path = ckpt_dir / "policy_best.pt"
                torch.save({
                    "epoch": epoch,
                    "policy_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "obs_mean": obs_mean,
                        "obs_std": obs_std,
                        "action_mean": action_mean,
                        "action_std": action_std,
                    },
                    "args": vars(args),
                    "success_rate": sr,
                }, best_path)
                print(f"[保存] 最佳模型 ({best_sr:.1f}%) → {best_path}")

            ema.restore()  # 恢复用于继续训练

        metrics.append(row)
        write_metrics_csv(metrics, metrics_path)
        save_training_curves(metrics, curves_path)

        last_path = ckpt_dir / "policy_last.pt"
        torch.save({
            "epoch": epoch,
            "policy_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer": {
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "action_mean": action_mean,
                "action_std": action_std,
            },
            "args": vars(args),
            "metrics": metrics,
        }, last_path)

    print(f"\n训练完成！最佳成功率: {best_sr:.1f}%")
    print(f"[曲线] 指标 CSV → {metrics_path}")
    print(f"[曲线] 训练曲线 → {curves_path}")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy 训练")
    # 数据
    parser.add_argument("--data", type=str, default="demos_PickCube-v1_state_random.h5",
                        help="HDF5 演示数据路径")
    # 任务
    parser.add_argument("--env", type=str, default="PickCube-v1",
                        choices=["PickCube-v1", "PegInsertionSide-v1"],
                        help="任务环境 ID")
    parser.add_argument("--obs-mode", type=str, default="state",
                        choices=["state", "rgbd"])
    parser.add_argument("--obs-dim", type=int, default=None,
                        help="观测维度（state 模式: PickCube=42, PegInsertion=43）")
    parser.add_argument("--action-dim", type=int, default=8,
                        help="动作维度（两任务均为 8）")
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="批次大小（无 GPU 建议 32~64）")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-timesteps", type=int, default=100,
                        help="DDPM 噪声步数")
    parser.add_argument("--inference-steps", type=int, default=20,
                        help="评估/推理时 DDIM 去噪步数")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["linear", "cosine"])
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    # 模型
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="残差 MLP 层数")
    # 时序
    parser.add_argument("--obs-horizon", type=int, default=2,
                        help="历史观测帧数 To")
    parser.add_argument("--pred-horizon", type=int, default=16,
                        help="预测动作序列长度 Tp")
    parser.add_argument("--action-horizon", type=int, default=8,
                        help="每次采样后连续执行的动作步数 Ta")
    # 评估
    parser.add_argument("--eval-every", type=int, default=5,
                        help="每多少轮评估一次")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=None,
                        help="评估 episode 最大步数；默认使用环境内置上限")
    # 输出
    parser.add_argument("--output-dir", type=str, default="output",
                        help="输出目录")

    args = parser.parse_args()

    # 自动设置 obs_dim
    if args.obs_dim is None:
        args.obs_dim = 42 if "PickCube" in args.env else 43

    print("=" * 55)
    print("Diffusion Policy 训练")
    print("=" * 55)
    print(f"  数据文件 : {args.data}")
    print(f"  任务环境 : {args.env}")
    print(f"  观测模式 : {args.obs_mode}  (obs_dim={args.obs_dim})")
    print(f"  动作维度 : {args.action_dim}")
    print(f"  训练轮数 : {args.epochs}")
    print(f"  批次大小 : {args.batch_size}")
    print(f"  学习率   : {args.lr}")
    print(f"  训练噪声步数 : {args.num_timesteps} ({args.schedule})")
    print(f"  推理去噪步数 : {args.inference_steps}")
    print(f"  To / Tp  : {args.obs_horizon} / {args.pred_horizon}")
    print(f"  设备     : {DEVICE}")
    print("=" * 55)

    train(args)


if __name__ == "__main__":
    main()
