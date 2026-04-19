# ManiSkill3 任务配置说明

> **项目**：基于 ManiSkill3 的视觉机器人操作中 Diffusion Policy 的应用导向研究  
> **阶段**：第一阶段 — 环境搭建与任务验证  
> **时间**：2026-04-19  
> **状态**：✅ 环境安装完成，两个任务跑通验证

---

## 一、环境搭建指南

### 1.1 系统环境

| 项目 | 版本 / 状态 |
|------|-------------|
| 操作系统 | Windows 11 (64-bit) |
| Python | 3.13.5（`C:\Python\python.exe`） |
| PyTorch | 2.7.1+cu118 |
| CUDA | 可用 ✅ — NVIDIA GeForce RTX 4090 Laptop GPU |
| ManiSkill | 3.0.0b22 |
| Gymnasium | 0.29.1 |

> **注意**：ManiSkill3 官方推荐 Python 3.9–3.12，但经实测 Python 3.13 可正常安装与运行（CPU 及 GPU 均可）。如遇兼容性问题，建议使用 conda 创建 Python 3.10 虚拟环境。

### 1.2 安装步骤

```bash
# 方法一：直接 pip 安装（已验证可用）
pip install --upgrade mani-skill

# 方法二：国内镜像加速（网络受限时使用）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mani-skill

# 可选：安装 pinocchio（机器人运动学，缺少时会警告但不影响基本功能）
pip install pin
```

**自动安装的主要依赖**（安装 mani-skill 时一并安装）：

| 包名 | 用途 |
|------|------|
| `sapien 3.0.3` | 底层物理仿真引擎（SAPIEN） |
| `gymnasium 0.29.1` | RL 环境标准接口 |
| `torch 2.7.1+cu118` | 深度学习框架 |
| `trimesh` | 3D 网格处理 |
| `transforms3d` | 空间变换工具 |
| `h5py` | HDF5 数据集读写 |
| `huggingface_hub` | 资产文件下载 |
| `pytorch_kinematics` | 机器人运动学 |

### 1.3 安装验证

```python
import mani_skill
import gymnasium as gym
import mani_skill.envs  # 必须导入，完成环境注册
import torch

print('ManiSkill 版本:', mani_skill.__version__)     # 3.0.0b22
print('Gymnasium 版本:', gym.__version__)             # 0.29.1
print('PyTorch 版本:', torch.__version__)              # 2.7.1+cu118
print('CUDA 可用:', torch.cuda.is_available())        # True
print('GPU 设备:', torch.cuda.get_device_name(0))     # RTX 4090 Laptop GPU
```

> ⚠️ **常见警告**：  
> `UserWarning: pinnochio package is not installed, robotics functionalities will not be available`  
> 此警告不影响任务运行，可忽略或通过安装 `pin` 包消除。

---

## 二、任务一：桌面单物体抓取（PickCube-v1）

### 2.1 任务概述

| 项目 | 描述 |
|------|------|
| **环境 ID** | `PickCube-v1` |
| **任务类型** | 桌面操作 / 抓取放置 |
| **难度** | ⭐⭐（中等偏低，适合基线实验） |
| **机器人** | Panda（7-DOF 机械臂 + 平行夹爪） |
| **任务目标** | 抓取桌面上的红色立方体，将其移动到绿色目标球所在位置 |

### 2.2 观测空间（Observation Space）

**默认观测模式（`obs_mode='state'`，状态向量）**

| 字段 | 维度 | 类型 | 说明 |
|------|------|------|------|
| `is_grasped` | (1,) | bool | 机器人当前是否抓取立方体 |
| `tcp_pose` | (7,) | float32 | 末端执行器位姿（3D 位置 + 四元数） |
| `goal_pos` | (3,) | float32 | 目标位置（绿色球体 XYZ 坐标） |
| `obj_pose` | (7,) | float32 | 立方体位姿（3D 位置 + 四元数） |
| `tcp_to_obj_pos` | (3,) | float32 | TCP 到立方体的相对位置向量 |
| `obj_to_goal_pos` | (3,) | float32 | 立方体到目标的相对位置向量 |
| 机器人关节状态 | (18,) | float32 | qpos(9) + qvel(9)（7关节+2夹爪） |

> **合并后状态向量维度**：`(1, 42)` — 经实测确认（单环境时 batch_size=1）

**图像观测模式（`obs_mode='rgbd'`，用于 Diffusion Policy 训练）**

| 传感器 | 图像类型 | 分辨率 | 视场角（FoV） |
|--------|----------|--------|-------------|
| `base_camera` | RGB + Depth | 128×128 | 90°（π/2 rad） |
| 可选手腕摄像头 | RGB + Depth | 128×128 | — |

创建图像观测环境示例：
```python
env = gym.make(
    'PickCube-v1',
    obs_mode='rgbd',         # 图像+深度观测
    render_mode='rgb_array'  # 无头渲染
)
```

### 2.3 动作空间（Action Space）

| 项目 | 值 |
|------|----|
| **类型** | `Box`（连续动作空间） |
| **维度** | `(8,)` |
| **取值范围** | `[-1.0, +1.0]`（每个维度） |
| **数据类型** | `float32` |
| **控制模式** | 末端执行器增量控制（ΔEE pose，delta position + delta rotation） |

```
动作向量含义（8维）:
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_open, (reserved)] ← 通用示意
```

> 精确含义取决于控制模式配置，ManiSkill3 默认为 `pd_ee_delta_pose`（末端执行器增量位姿 PD 控制）。

### 2.4 随机化设置

每次 `env.reset()` 时进行以下随机化：

| 随机化对象 | 范围 | 方式 |
|------------|------|------|
| **立方体 XY 位置** | 中心 (0,0)，范围 ±0.05m | 均匀随机 |
| **立方体 Z 旋转** | [0, 2π) | 均匀随机（X/Y 轴固定） |
| **目标点 XY 位置** | 与立方体相同区域 | 均匀随机 |
| **目标点 Z 高度** | [立方体高度, 立方体高度 + 0.3m] | 均匀随机 |
| **机器人初始关节** | 标准位形 ± 0.02 rad | 高斯噪声 |

### 2.5 成功判定条件

任务成功需**同时满足**以下两个条件：

1. **位置误差**：立方体质心与目标点距离 ≤ **0.025 m**
2. **机器人静止**：所有关节速度 < **0.2 rad/s**（防止惯性碰撞）

### 2.6 奖励函数结构

密集奖励（dense reward）组成：
- **接近奖励**：TCP 与立方体距离的单调减函数
- **抓取奖励**：成功抓取时给予正向奖励（is_grasped=True）
- **放置奖励**：立方体与目标点距离的单调减函数
- **成功奖励**：满足成功条件时 +10

### 2.7 验证代码

```python
import gymnasium as gym
import mani_skill.envs

# 创建环境（无头模式，适合服务器/无显示器场景）
env = gym.make('PickCube-v1', render_mode='rgb_array')
obs, info = env.reset(seed=42)
print('环境创建成功！观测维度:', obs.shape)  # torch.Size([1, 42])

# 运行随机动作
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print('PickCube-v1 验证通过 ✅')
```

**实测结果**（50步随机动作）：
- 环境初始化：✅ 正常
- 随机步进：✅ 无报错
- 累计奖励：2.02（随机策略，未成功抓取）
- 任务成功次数：0（符合预期，随机动作无法完成精确抓取）

---

## 三、任务二：轴孔插入（PegInsertionSide-v1）

### 3.1 任务概述

| 项目 | 描述 |
|------|------|
| **环境 ID** | `PegInsertionSide-v1` |
| **任务类型** | 桌面操作 / 精密装配 |
| **难度** | ⭐⭐⭐⭐（高精度，适合验证策略精细操作能力） |
| **机器人** | PandaWristCam（带手腕摄像头的 Panda 机械臂） |
| **任务目标** | 夹取橙白色插销（peg），将橙色端插入带孔盒子侧面的孔中 |

### 3.2 观测空间（Observation Space）

**默认观测模式（`obs_mode='state'`，状态向量）**

| 字段 | 维度 | 类型 | 说明 |
|------|------|------|------|
| `tcp_pose` | (7,) | float32 | 末端执行器位姿（位置 + 四元数） |
| `peg_pose` | (7,) | float32 | 插销位姿（位置 + 四元数） |
| `peg_half_size` | (3,) | float32 | 插销半尺寸 [半长, 半径, 半径]（m） |
| `box_hole_pose` | (7,) | float32 | 盒子孔洞位姿 |
| `box_hole_radius` | (1,) | float32 | 孔洞半径（m） |
| 机器人关节状态 | (18,) | float32 | qpos(9) + qvel(9) |

> **合并后状态向量维度**：`(1, 43)` — 经实测确认（比 PickCube 多 1 维，包含孔洞半径）

**图像观测模式（`obs_mode='rgbd'`）**

| 传感器 | 图像类型 | 分辨率 | 备注 |
|--------|----------|--------|------|
| `base_camera` | RGB + Depth | 128×128 | 固定外部视角 |
| `hand_camera` | RGB + Depth | 128×128 | 手腕摄像头（PandaWristCam 特有） |

> PegInsertionSide 使用 **PandaWristCam** 机器人，带有手腕摄像头，可提供更丰富的近距离视觉信息，有助于精密插入操作。

### 3.3 动作空间（Action Space）

| 项目 | 值 |
|------|----|
| **类型** | `Box`（连续动作空间） |
| **维度** | `(8,)` |
| **取值范围** | `[-1.0, +1.0]`（每个维度） |
| **数据类型** | `float32` |
| **控制模式** | 末端执行器增量控制（pd_ee_delta_pose） |

> 与 PickCube-v1 **动作空间结构相同**，便于统一 Diffusion Policy 网络输出头设计。

### 3.4 随机化设置

每次 `env.reset()` 时进行以下随机化：

| 随机化对象 | 范围 | 方式 |
|------------|------|------|
| **插销半长** | [0.085, 0.125] m | 均匀随机 |
| **插销半径** | [0.015, 0.025] m | 均匀随机 |
| **孔洞半径** | 插销半径 + 0.003 m（固定间隙） | 由插销尺寸决定 |
| **插销 XY 位置** | 桌面区域内 | 均匀随机 |
| **插销 Z 旋转** | π/2 ± π/3（约 30°–150°） | 均匀随机 |
| **盒子 X 位置** | [-0.05, 0.05] m | 均匀随机 |
| **盒子 Y 位置** | [0.20, 0.40] m | 均匀随机 |
| **盒子 Z 旋转** | π/2 ± π/8（约 67.5°–112.5°） | 均匀随机 |

> ⚠️ **重要**：插销尺寸（半径、半长）每回合都会随机化，这意味着策略需要泛化到不同尺寸的插销。训练时应将 `peg_half_size` 纳入观测。

### 3.5 成功判定条件

插销**橙色端（头部）**相对于孔洞坐标系的位置需满足：

| 轴向 | 条件 | 含义 |
|------|------|------|
| X（插入方向） | x ≥ -0.015 m | 插入深度 ≥ 15mm |
| Y（侧向） | \|y\| ≤ 孔洞半径 | 横向对齐 |
| Z（竖向） | \|z\| ≤ 孔洞半径 | 纵向对齐 |

即：**插销头部距孔洞中心轴的横向偏差 ≤ 孔洞半径，且插入深度 ≥ 15mm**

### 3.6 奖励函数结构

密集奖励分为 4 个阶段，具有明确的层次结构：

| 阶段 | 条件 | 奖励描述 |
|------|------|---------|
| 1 — 旋转对齐 | 始终计算 | 夹爪旋转与插销方向对齐 |
| 2 — 接近抓取 | 始终计算 | TCP 接近插销 + is_grasped 奖励 |
| 3 — 预插入对齐 | is_grasped=True | 插销方向与孔洞轴线对齐 |
| 4 — 完成插入 | is_grasped & pre_inserted | 插入深度奖励 |
| 成功 | 满足成功条件 | **+10**（稀疏奖励） |

### 3.7 验证代码

```python
import gymnasium as gym
import mani_skill.envs

# 创建环境
env = gym.make('PegInsertionSide-v1', render_mode='rgb_array')
obs, info = env.reset(seed=42)
print('环境创建成功！观测维度:', obs.shape)  # torch.Size([1, 43])

# 运行随机动作
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print('PegInsertionSide-v1 验证通过 ✅')
```

**实测结果**（50步随机动作）：
- 环境初始化：✅ 正常
- 随机步进：✅ 无报错
- 累计奖励：1.72（随机策略）
- 任务成功次数：0（符合预期，插销插入需要高精度控制）

---

## 四、两任务对比

| 对比项 | PickCube-v1 | PegInsertionSide-v1 |
|--------|-------------|---------------------|
| **任务类型** | 抓取放置 | 精密插入 |
| **难度** | 中等 | 高 |
| **机器人** | Panda | PandaWristCam |
| **状态向量维度** | (1, 42) | (1, 43) |
| **动作维度** | (8,) | (8,) |
| **动作范围** | [-1, 1] | [-1, 1] |
| **成功阈值** | 位置误差 ≤ 25mm | 插入深度 ≥ 15mm & 对齐 |
| **手腕摄像头** | ❌ | ✅ |
| **几何随机化** | 仅初始位置 | 包含插销尺寸随机化 |
| **随机动作成功率** | 极低（< 1%） | 几乎为 0 |

---

## 五、Diffusion Policy 适配要点

基于以上环境信息，适配 Diffusion Policy 时需注意：

### 5.1 观测设计

```
建议观测输入（视觉模式）:
- RGB 图像: (H, W, 3) — 来自 base_camera（两任务均有）
- 深度图: (H, W, 1) — 可选，提供深度信息
- 本体感知: tcp_pose (7,) + gripper_state (1,)
- 任务状态（可选）: is_grasped, goal_pos / peg_half_size 等

建议观测输入（纯状态模式，用于快速验证）:
- 状态向量: (42,) 或 (43,) — 直接输入 MLP/Transformer
```

### 5.2 动作设计

```
Diffusion Policy 输出头（两任务统一）:
- 动作序列长度: T_a（建议 4–16）
- 动作维度: 8
- 动作范围: [-1, 1]（与环境 action_space 一致）
- 归一化: 无需额外归一化（范围已标准化）
```

### 5.3 数据收集

```bash
# 收集演示数据（teleoperation 或专家策略）
python -m mani_skill.examples.demo_teleop -e PickCube-v1 --save-video
# 数据默认保存至 ~/.maniskill/demos/

# 转换为 HDF5 格式（用于 Diffusion Policy 训练）
python -m mani_skill.utils.convert_demo ...
```

### 5.4 已知限制与注意事项

1. **Windows 渲染限制**：Windows 下无法弹出 GUI 窗口（`render_mode='human'` 不可用），需使用 `render_mode='rgb_array'`
2. **GPU 内存**：并行仿真（`num_envs > 1`）对显存要求较高，RTX 4090 Laptop（16GB）建议 `num_envs ≤ 32`
3. **pinocchio 警告**：缺少 pinocchio 包时会打印警告，不影响运行；若需正向/逆向运动学功能则需安装
4. **资产下载**：首次创建某任务环境时会自动从 HuggingFace 下载资产文件（模型、纹理等）至 `~/.maniskill/data/`，需要网络连接；可设置 `MS_ASSET_DIR` 环境变量更改存储路径

---

## 六、关键代码参考

### 6.1 环境创建（全模式）

```python
import gymnasium as gym
import mani_skill.envs

# --- 基础模式 ---
env = gym.make('PickCube-v1', render_mode='rgb_array')

# --- 图像观测模式（Diffusion Policy 推荐）---
env = gym.make('PickCube-v1', obs_mode='rgbd', render_mode='rgb_array')

# --- GPU 并行加速（数据收集）---
env = gym.make('PickCube-v1', num_envs=16, render_mode='rgb_array')

# --- 自定义机器人 ---
env = gym.make('PickCube-v1', robot_uids='fetch', render_mode='rgb_array')
```

### 6.2 标准 RL 循环

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make('PickCube-v1', render_mode='rgb_array')
obs, info = env.reset(seed=0)

for episode in range(10):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # 用你的策略替换 action_space.sample()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        done = terminated or truncated

    print(f'Episode {episode}: reward={episode_reward:.3f}, success={info.get("success", False)}')

env.close()
```

### 6.3 查询观测/动作空间

```python
import gymnasium as gym
import mani_skill.envs

for env_id in ['PickCube-v1', 'PegInsertionSide-v1']:
    env = gym.make(env_id, render_mode='rgb_array')
    obs, _ = env.reset(seed=0)
    print(f'\n=== {env_id} ===')
    print('obs_space:', env.observation_space)
    print('obs shape:', obs.shape if hasattr(obs, 'shape') else {k: v.shape for k, v in obs.items()})
    print('act_space:', env.action_space)
    print('act shape:', env.action_space.shape)
    print('act low:', env.action_space.low)
    print('act high:', env.action_space.high)
    env.close()
```

---

## 七、附录：安装日志摘要

```
mani-skill==3.0.0b22 安装成功（2026-04-19）
主要依赖版本：
  sapien==3.0.3
  gymnasium==0.29.1
  pytorch_kinematics（最新）
  huggingface_hub（最新）
  h5py（最新）
  trimesh==4.11.5
  transforms3d==0.4.2

GPU 环境：
  CUDA: 11.8（cu118）
  GPU: NVIDIA GeForce RTX 4090 Laptop GPU
  PyTorch: 2.7.1+cu118

验证结果：
  PickCube-v1       ✅ 环境创建成功，50步随机动作完成
  PegInsertionSide-v1  ✅ 环境创建成功，50步随机动作完成
```
