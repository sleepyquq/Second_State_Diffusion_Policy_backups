# 第二阶段交付说明 — Diffusion Policy 实现

> **前置工作（第一阶段）已完成**：ManiSkill3 环境搭建、两个任务跑通验证  
> **本阶段目标**：实现 Diffusion Policy 主方法，调训练参数，保证模型收敛  
> **时间**：2026-04-19  

---

## 一、你接手时的现状

### 1.1 已完成的工作

| 项 | 状态 | 说明 |
|----|:----:|------|
| ManiSkill3 安装 | ✅ | mani-skill 3.0.0b22，Python 3.13 + CUDA 11.8 |
| 任务一 PickCube-v1 | ✅ | state / rgbd 两种观测模式均跑通 |
| 任务二 PegInsertionSide-v1 | ✅ | state / rgbd 两种观测模式均跑通 |
| 任务配置说明文档 | ✅ | `ManiSkill3_任务配置说明.md` |
| 两任务跑通脚本 | ✅ | `run_tasks.py`（含 Windows 适配） |

### 1.2 运行环境

| 项 | 版本 |
|----|------|
| 操作系统 | Windows 11，64-bit |
| Python | 3.13.5（路径：`C:\Python\python.exe`） |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8，RTX 4090 Laptop GPU |
| ManiSkill | 3.0.0b22 |
| Gymnasium | 0.29.1 |

---

## 二、核心接口规格（你的代码需对接这些）

### 2.1 观测空间

**State 模式**（快速验证 Diffusion Policy 可训练性，推荐优先用此模式）

| 任务 | obs tensor shape | dtype | 说明 |
|------|-----------------|-------|------|
| PickCube-v1 | `(1, 42)` | float32 | 单环境时 batch=1 |
| PegInsertionSide-v1 | `(1, 43)` | float32 | 多 1 个孔洞半径维度 |

**RGBD 模式**（视觉策略，Diffusion Policy 最终目标）

| 任务 | 字段 | shape | dtype |
|------|------|-------|-------|
| PickCube-v1 | `agent/qpos` | (1, 9) | float32 |
| | `agent/qvel` | (1, 9) | float32 |
| | `extra/tcp_pose` | (1, 7) | float32 |
| | `base_camera/rgb` | (1, 128, 128, 3) | uint8 |
| | `base_camera/depth` | (1, 128, 128, 1) | int16 |
| PegInsertionSide-v1 | 以上所有字段 + | | |
| | `hand_camera/rgb` | (1, 128, 128, 3) | uint8 ← 手腕摄像头 |
| | `hand_camera/depth` | (1, 128, 128, 1) | int16 |

### 2.2 动作空间

两个任务**完全相同**，可统一设计网络输出头：

```
类型：Box（连续）
维度：(8,)
范围：[-1.0, 1.0]（每个维度）
dtype：float32
控制模式：pd_ee_delta_pose（末端执行器增量位姿 PD 控制）

动作含义（8维）：
  [0:3]  = Δ末端位置    (Δx, Δy, Δz)
  [3:6]  = Δ末端姿态    (Δroll, Δpitch, Δyaw) 或轴角
  [6]    = 夹爪开合度   +1=全开 / -1=全闭
  [7]    = （保留/未使用）
```

---

## 三、Diffusion Policy 实现指南

### 3.1 推荐技术路线

```
推荐参考实现：
  官方仓库：https://github.com/real-stanford/diffusion_policy
  ManiSkill 示例：https://github.com/haosulab/ManiSkill/tree/main/examples/baselines

噪声预测网络选型（二选一）：
  ① CNN-based（UNet1D）   — 计算快，收敛稳定，推荐先跑通
  ② Transformer-based    — 性能更强，参数量更大，调参复杂

扩散调度器：
  ① DDPM（训练用）
  ② DDIM（推理时加速，减少去噪步数 100→10）
```

### 3.2 关键超参数参考（可收敛起点）

| 超参数 | 推荐值（State模式） | 推荐值（RGBD模式） | 说明 |
|--------|-------------------|------------------|------|
| `obs_horizon` (To) | 2 | 2 | 每次输入观测帧数 |
| `action_horizon` (Ta) | 8 | 8 | 单次执行步数 |
| `pred_horizon` (Tp) | 16 | 16 | 预测动作序列长度 |
| `num_diffusion_iters`（训练） | 100 | 100 | DDPM 加噪步数 |
| `num_diffusion_iters`（推理） | 10 | 10 | DDIM 去噪步数 |
| `batch_size` | 256 | 64 | RGBD 显存占用大，需减小 |
| `learning_rate` | 1e-4 | 1e-4 | AdamW |
| `weight_decay` | 1e-6 | 1e-6 | AdamW |
| `lr_scheduler` | cosine | cosine | 带 warmup |
| `ema_decay` | 0.9999 | 0.9999 | EMA 指数平均 |
| 训练步数 | 60k–100k steps | 200k+ steps | 视收敛曲线而定 |

> **收敛判断基准**：  
> - PickCube-v1：State 模式 10 万步内，成功率 > 70% 是可行的  
> - PegInsertionSide-v1：难度更高，视觉模式建议先跑 State 验证网络正确性

### 3.3 演示数据规格

Diffusion Policy 是模仿学习方法，**必须先有专家演示数据**。

```python
# 演示数据格式（每条轨迹）：
{
    "obs": {
        "state": np.ndarray,   # shape (T, obs_dim)  —— state 模式
        # 或
        "rgb":   np.ndarray,   # shape (T, H, W, 3)  —— rgbd 模式
        "depth": np.ndarray,   # shape (T, H, W, 1)  —— rgbd 模式
        "proprio": np.ndarray, # shape (T, 9)        —— 关节状态（rgbd模式补充）
    },
    "actions": np.ndarray,     # shape (T, 8)
}

# 推荐收集数量（确保策略收敛）：
#   PickCube-v1：        ≥ 100 条演示（state 模式），≥ 300 条（rgbd 模式）
#   PegInsertionSide-v1：≥ 200 条演示（state 模式），≥ 500 条（rgbd 模式）
```

**收集方式**（见 `collect_demos.py`）：
1. **内置专家策略**：ManiSkill3 自带 CPU 专家控制器（见下文），可批量生成高质量演示
2. **脚本自动收集**：批量运行 `python -m mani_skill.examples.demo_random_action`（随机，效果差），推荐用内置专家

### 3.4 Gym 交互接口（数据收集循环模板）

```python
import gymnasium as gym
import mani_skill.envs  # 必须导入完成注册
import numpy as np

# Windows 须加 render_backend='sapien_cpu'（RGBD 模式）
env = gym.make(
    "PickCube-v1",
    obs_mode="rgbd",           # 或 "state"
    render_mode="rgb_array",
    render_backend="sapien_cpu",  # Windows 必须！
)

trajectories = []
for traj_id in range(num_trajectories):
    obs, info = env.reset(seed=traj_id)
    obs_list, action_list = [], []

    done = False
    while not done:
        # ← 这里替换为你的专家策略或 teleoperation
        action = get_expert_action(obs, info)

        obs_list.append(obs)
        action_list.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated or info.get("success", False)

    if info.get("success", False):      # 只保存成功轨迹
        trajectories.append({
            "obs": obs_list,
            "actions": np.array(action_list),
        })

env.close()
```

---

## 四、已知坑与注意事项

### 4.1 Windows 渲染崩溃（最重要！）

```
问题：RGBD 模式下，Intel 核显 + RTX 4090 双显卡笔记本
     使用默认 render_backend='gpu' 会触发 0xC0000005 崩溃（无 Python 异常）

解决：所有 RGBD/sensor_data 观测模式必须加：
     render_backend='sapien_cpu'

代码（已在 run_tasks.py 中实现）：
     is_windows = platform.system() == "Windows"
     render_backend = "sapien_cpu" if is_windows else "gpu"
```

### 4.2 观测维度注意

```
State 模式返回的 obs 是 torch.Tensor（不是 np.ndarray），shape 含 batch 维度：
  (1, 42) 而不是 (42,)

在 Diffusion Policy 的 obs_encoder 里需要注意 squeeze/reshape：
  obs = obs.squeeze(0)  # → (42,)
  或者统一用 obs.reshape(-1, obs_dim)
```

### 4.3 PegInsertionSide 尺寸随机化

```
peg_half_size（插销尺寸）每回合都会随机化，策略必须将其作为观测输入！
state 向量已包含此信息（第 43 维），但 rgbd 模式下需手动从 info 中取：
  peg_half_size = info.get("peg_half_size")  # (3,)
并拼接到本体感知向量中。
```

### 4.4 成功率衡量

```
随机策略两个任务成功率均为 0（符合预期）。
实测随机奖励：
  PickCube-v1:          2.02（50步）
  PegInsertionSide-v1:  1.72（50步）

Diffusion Policy 训练时请使用 eval_success_rate（每 N 步 rollout 评估）
作为主收敛指标，而不是 loss 曲线。
```

### 4.5 并行采样 num_envs

```
ManiSkill3 支持 GPU 并行化（num_envs > 1），可大幅加速数据收集：
  CPU state 模式：num_envs ≤ 256（推荐 64）
  GPU rgbd 模式：num_envs ≤ 32（RTX 4090 Laptop 16GB 限制）

注意：并行采样时 obs shape 变为 (num_envs, obs_dim)，需相应调整批处理逻辑。
```

---

## 五、文件清单

收到的文件：

| 文件 | 说明 |
|------|------|
| `ManiSkill3_任务配置说明.md` | 两个任务完整配置文档（观测/动作/随机化/奖励函数） |
| `run_tasks.py` | 两任务跑通脚本（含 Windows 渲染适配） |
| `env_probe_results.json` | 实测观测/动作空间数据（JSON 格式） |
| `collect_demos.py` | 专家演示数据收集脚本模板 |
| 本文档 | 第二阶段接口说明 |

---

## 六、推荐实施步骤

```
Step 1：先用 State 模式验证 Diffusion Policy 网络可训练
   - obs_dim=42（PickCube），action_dim=8
   - 用内置专家收集 100 条演示
   - 训练 6 万步，验证 success_rate > 60%

Step 2：迁移到 RGBD 模式
   - 替换 obs_encoder：Linear → ResNet/CNN + 本体感知拼接
   - 增大演示数量（≥ 300 条）
   - 调整 batch_size（64 避免 OOM）

Step 3：在 PegInsertionSide-v1 上验证
   - 注意 peg_half_size 必须包含在观测中
   - 难度更高，可先 state 模式，成功率 > 30% 为基线

Step 4：调参收敛优化
   - 学习率：1e-4（稳定）→ 根据 loss 平台期调整
   - EMA：0.9999（必须，DP 核心技巧）
   - 数据增强（RGBD）：随机颜色抖动 + 随机裁剪
```

---

> 有问题联系第一阶段负责人（本文档随环境一并交付）。
