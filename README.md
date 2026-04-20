# Second State Diffusion Policy

本项目在 ManiSkill3 标准任务上训练并评估 Diffusion Policy，当前阶段重点是任务2：

1. `PickCube-v1` 抓取任务标准条件下跑通并收敛。
2. `PegInsertionSide-v1` 插入任务标准条件下跑通，并留下可复现 baseline。
3. 输出训练曲线、评估 CSV、checkpoint、日志和 OpenCV 基准视频，供后续任务3/4/5继续使用。

当前结论：

- `PickCube-v1` 已稳定收敛，最终 10 episode 评估成功率为 100%。
- `PegInsertionSide-v1` 已跑通并有成功 rollout，但尚未稳定收敛。当前 30 episode 确认评估成功率为 30%，适合作为后续优化 baseline，不应当被写成高成功率最终模型。

## 1. 环境

当前项目运行环境为 Windows 云主机，本地目录即云端训练目录：

```text
D:\Second_State_Diffusion_Policy
```

已验证配置：

| 项目 | 当前值 |
|---|---|
| OS | Windows |
| Python | 3.11 |
| 环境目录 | `.mambaenv` |
| GPU | NVIDIA GeForce RTX 3080 |
| PyTorch | 2.5.1+cu121 |
| OpenCV | 4.13.0 |
| ManiSkill | 已安装，可运行 `PickCube-v1` 和 `PegInsertionSide-v1` |

运行命令时建议统一使用 `.mambaenv`：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:GIT_PYTHON_REFRESH='quiet'
& 'C:\micromamba\Library\bin\micromamba.exe' run -p 'D:\Second_State_Diffusion_Policy\.mambaenv' python <script.py> <args>
```

说明：

- `KMP_DUPLICATE_LIB_OK=TRUE` 用于避免 Windows 上 OpenMP 重复加载导致的崩溃。
- `GIT_PYTHON_REFRESH=quiet` 用于避免 GitPython 在没有 git.exe 时打断训练。
- 任务管理器如果看不到 GPU 占用，需要切换 GPU 图表为 CUDA/Compute；训练日志中会打印 `使用设备: cuda`。

## 2. 项目文件说明

### 2.1 核心脚本

| 文件 | 作用 |
|---|---|
| `train_dp.py` | Diffusion Policy 主训练脚本，支持 state 观测、DDIM 推理、EMA、训练曲线、resume、receding horizon 评估 |
| `collect_demos.py` | demo 数据读取与转换，已支持 ManiSkill 官方 `traj_*` HDF5 格式 |
| `evaluate_checkpoint.py` | 加载 checkpoint 单独评估，支持比较 `action_horizon`、`inference_steps`、`max_steps`、`sample_init` |
| `record_policy_video.py` | 使用 OpenCV 录制策略 rollout 视频，支持多 seed 尝试并只保存成功视频 |
| `run_tasks.py` | 早期任务运行辅助脚本，当前最终训练主要使用上面三个脚本 |

### 2.2 文档

| 文件 | 说明 |
|---|---|
| `README.md` | 当前文件，说明如何复现、项目结构、数据与结果对应关系 |
| `任务2_Diffusion_Policy训练技术记录.md` | 详细训练技术记录，包括障碍、分析、解决方法和阶段结果 |
| `任务3_4_5成员分工与指导.md` | 面向后续成员的任务三、四、五分工与执行建议 |
| `ManiSkill3_任务配置说明.md` | 原始 ManiSkill 任务配置说明 |
| `交付_第二阶段_Diffusion_Policy实现说明.md` | 第二阶段实现说明 |
| `提案.pdf` | 原提案材料 |

### 2.3 数据目录

```text
official_demos/
  PickCube-v1/
    motionplanning/
    rl/
    teleop/
  PegInsertionSide-v1/
    motionplanning/
    rl/
    combined/
```

当前最终采用的数据如下：

| 任务 | 最终采用数据 | 说明 |
|---|---|---|
| PickCube | `official_demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | 100 条 motionplanning replay 成功轨迹 |
| PegInsertionSide | `official_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | 165 条 motionplanning replay 成功轨迹，当前 Peg baseline 使用它 |

已尝试但未作为最终结果的数据：

| 数据 | 位置 | 结果 |
|---|---|---|
| Peg RL replay | `official_demos/PegInsertionSide-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | 单独训练效果差，成功率 0% |
| Peg combined | `official_demos/PegInsertionSide-v1/combined/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | motionplanning + RL 共 360 条，训练后最佳仅 10%，未采用 |
| PickCube RL replay | `official_demos/PickCube-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | replay 成功轨迹过少，未作为最终训练集 |

### 2.4 输出目录

| 目录 | 状态 |
|---|---|
| `output_pickcube_state_final/` | PickCube 最终收敛结果 |
| `output_peg_state_motionplanning/` | Peg motionplanning 第一轮训练结果 |
| `output_peg_state_motionplanning_resume/` | Peg 当前最终 baseline，基于第一轮继续训练 |
| `output_peg_state_combined/` | Peg combined 数据尝试，结果较差，仅作对照 |
| `logs/` | 所有训练、评估、录制日志 |

## 3. 如何复现当前结果

以下命令均在项目根目录运行：

```powershell
cd D:\Second_State_Diffusion_Policy
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:GIT_PYTHON_REFRESH='quiet'
```

为便于阅读，下面命令中的 Python 前缀统一为：

```powershell
$PY = "C:\micromamba\Library\bin\micromamba.exe"
```

实际执行时使用：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python ...
```

### 3.1 复现 PickCube 最终训练

训练命令：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python train_dp.py `
  --data official_demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 `
  --env PickCube-v1 `
  --obs-mode state `
  --epochs 300 `
  --batch-size 256 `
  --lr 1e-4 `
  --weight-decay 1e-6 `
  --ema-decay 0.995 `
  --eval-every 25 `
  --eval-episodes 10 `
  --eval-max-steps 100 `
  --action-horizon 8 `
  --output-dir output_pickcube_state_final `
  --num-timesteps 100 `
  --inference-steps 20
```

最终结果文件：

| 文件 | 说明 |
|---|---|
| `output_pickcube_state_final/checkpoints/policy_best.pt` | 最佳模型 |
| `output_pickcube_state_final/checkpoints/policy_last.pt` | 最后一轮模型 |
| `output_pickcube_state_final/metrics.csv` | 每 epoch loss 与评估指标 |
| `output_pickcube_state_final/training_curves.png` | loss / success / reward 曲线 |
| `logs/pickcube_state_final.log` | 训练日志 |

最终指标：

| 指标 | 数值 |
|---|---:|
| final eval success | 100.0% |
| eval episodes | 10 |
| mean reward | 20.41 |
| mean steps | 66.2 |
| final loss | 0.0504 |

单独确认 checkpoint：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python evaluate_checkpoint.py `
  --checkpoint output_pickcube_state_final/checkpoints/policy_best.pt `
  --env PickCube-v1 `
  --episodes 10 `
  --max-steps 100 `
  --action-horizon 8 `
  --inference-steps 20
```

录制 PickCube 基准视频：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python record_policy_video.py `
  --checkpoint output_pickcube_state_final/checkpoints/policy_best.pt `
  --env PickCube-v1 `
  --out output_pickcube_state_final/videos/pickcube_baseline.mp4 `
  --attempts 10 `
  --seed-start 0 `
  --require-success `
  --max-steps 100 `
  --action-horizon 8 `
  --inference-steps 20
```

当前视频结果：

| 文件 | seed | 成功 | 步数 | reward |
|---|---:|---|---:|---:|
| `output_pickcube_state_final/videos/pickcube_baseline.mp4` | 0 | 是 | 69 | 21.37 |

## 4. PegInsertionSide 当前 baseline 复现

### 4.1 第一轮 motionplanning 训练

第一轮输出目录：

```text
output_peg_state_motionplanning/
```

训练命令：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python train_dp.py `
  --data official_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 `
  --env PegInsertionSide-v1 `
  --obs-mode state `
  --epochs 200 `
  --batch-size 256 `
  --lr 1e-4 `
  --weight-decay 1e-6 `
  --ema-decay 0.995 `
  --eval-every 20 `
  --eval-episodes 10 `
  --eval-max-steps 200 `
  --action-horizon 8 `
  --output-dir output_peg_state_motionplanning `
  --num-timesteps 100 `
  --inference-steps 20
```

结果摘要：

| 项目 | 结果 |
|---|---:|
| 最佳训练内成功率 | 10.0% |
| final eval success | 0.0% |
| final mean reward | 51.54 |
| final mean steps | 200.0 |

第一轮说明：reward 明显高于 RL 数据方案，说明策略学到了接近目标的方向，但插入末端不稳定。

### 4.2 续训得到当前 Peg baseline

当前采用输出目录：

```text
output_peg_state_motionplanning_resume/
```

训练命令：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python train_dp.py `
  --data official_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 `
  --env PegInsertionSide-v1 `
  --obs-mode state `
  --epochs 250 `
  --batch-size 256 `
  --lr 5e-5 `
  --weight-decay 1e-6 `
  --ema-decay 0.995 `
  --eval-every 25 `
  --eval-episodes 10 `
  --eval-max-steps 250 `
  --action-horizon 12 `
  --output-dir output_peg_state_motionplanning_resume `
  --num-timesteps 100 `
  --inference-steps 20 `
  --resume output_peg_state_motionplanning/checkpoints/policy_last.pt
```

训练内评估摘要：

| epoch | train loss | eval success | mean reward | mean steps |
|---:|---:|---:|---:|---:|
| 25 | 0.0465 | 10.0% | 83.51 | 243.0 |
| 75 | 0.0405 | 20.0% | 83.41 | 226.7 |
| 100 | 0.0399 | 30.0% | 84.86 | 223.3 |
| 125 | 0.0368 | 50.0% | 69.36 | 195.6 |
| 250 | 0.0308 | 40.0% | 76.75 | 210.3 |

当前 Peg baseline 文件：

| 文件 | 说明 |
|---|---|
| `output_peg_state_motionplanning_resume/checkpoints/policy_best.pt` | 当前 Peg 最佳模型，来自训练 epoch 124/125 附近 |
| `output_peg_state_motionplanning_resume/checkpoints/policy_last.pt` | 续训最后一轮模型 |
| `output_peg_state_motionplanning_resume/metrics.csv` | 续训曲线数据 |
| `output_peg_state_motionplanning_resume/training_curves.png` | 续训曲线图 |
| `logs/peg_state_motionplanning_resume.log` | 续训日志 |

### 4.3 Peg 确认评估

30 episode 确认评估：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python evaluate_checkpoint.py `
  --checkpoint output_peg_state_motionplanning_resume/checkpoints/policy_best.pt `
  --env PegInsertionSide-v1 `
  --episodes 30 `
  --max-steps 350 `
  --action-horizon 10 `
  --inference-steps 20 `
  --csv-out output_peg_state_motionplanning_resume/eval_confirm_best_ah10_max350.csv
```

确认结果：

| 配置 | episodes | success | mean reward | mean steps |
|---|---:|---:|---:|---:|
| `max_steps=350, action_horizon=10, inference_steps=20` | 30 | 30.0% | 99.19 | 290.4 |
| `max_steps=250, action_horizon=12, inference_steps=20` | 30 | 30.0% | 80.40 | 220.27 |

结论：Peg 当前是可运行、有成功 rollout、有评估证据的 baseline，但尚未稳定收敛。

### 4.4 Peg 基准视频

录制命令：

```powershell
& $PY run -p "D:\Second_State_Diffusion_Policy\.mambaenv" python record_policy_video.py `
  --checkpoint output_peg_state_motionplanning_resume/checkpoints/policy_best.pt `
  --env PegInsertionSide-v1 `
  --out output_peg_state_motionplanning_resume/videos/peg_baseline.mp4 `
  --attempts 40 `
  --seed-start 0 `
  --require-success `
  --max-steps 350 `
  --action-horizon 10 `
  --inference-steps 20
```

当前视频结果：

| 文件 | seed | 成功 | 步数 | reward |
|---|---:|---|---:|---:|
| `output_peg_state_motionplanning_resume/videos/peg_baseline.mp4` | 9 | 是 | 191 | 68.75 |

## 5. 其他训练尝试与对应结果

### 5.1 Peg RL 数据训练

输出目录：

```text
output_peg_state_final/
```

结果：

- 最终成功率：0%
- 平均 reward：约 4.66
- 结论：当前 RL replay 数据不适合作为 Peg 主训练集。

### 5.2 Peg combined 数据训练

数据：

```text
official_demos/PegInsertionSide-v1/combined/trajectory.state.pd_joint_delta_pos.physx_cpu.h5
```

输出目录：

```text
output_peg_state_combined/
```

结果：

| epoch | success | mean reward | mean steps |
|---:|---:|---:|---:|
| 60 | 10.0% | 21.07 | 323.9 |
| 180 | 10.0% | 59.39 | 333.0 |
| 270 | 10.0% | 37.25 | 332.4 |
| 300 | 10.0% | 40.64 | 335.6 |

结论：motionplanning 与 RL replay 混合后分布更杂，模型表现比 motionplanning-only 续训差，未采用。

### 5.3 Peg 推理参数搜索

输出文件：

| 文件 | 说明 |
|---|---|
| `output_peg_state_motionplanning/eval_matrix.csv` | 第一轮模型的 `action_horizon` / `inference_steps` 小矩阵 |
| `output_peg_state_motionplanning_resume/eval_refine.csv` | 续训最佳模型的 `max_steps` / `action_horizon` 搜索 |
| `output_peg_state_motionplanning_resume/eval_sample_init.csv` | `random` 与 `zero` 初始噪声对比 |

结论：

- `inference_steps=20` 保持为默认。
- `action_horizon=10` 或 `12` 都可用，但 30 episode 确认后成功率均为 30%。
- `sample_init=zero` 没有稳定提升，默认仍使用 `random`。

## 6. 数据与结果对应关系总表

| 任务 | 数据 | 输出目录 | checkpoint | 确认结果 |
|---|---|---|---|---|
| PickCube final | `official_demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | `output_pickcube_state_final/` | `checkpoints/policy_best.pt` | 10 ep, 100% |
| Peg first round | `official_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | `output_peg_state_motionplanning/` | `checkpoints/policy_best.pt` | 最佳训练内 10% |
| Peg current baseline | `official_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | `output_peg_state_motionplanning_resume/` | `checkpoints/policy_best.pt` | 30 ep, 30% |
| Peg combined failed attempt | `official_demos/PegInsertionSide-v1/combined/trajectory.state.pd_joint_delta_pos.physx_cpu.h5` | `output_peg_state_combined/` | `checkpoints/policy_best.pt` | 最佳 10% |

## 7. 视频与可视化证据

| 任务 | 视频 | OpenCV 校验 |
|---|---|---|
| PickCube | `output_pickcube_state_final/videos/pickcube_baseline.mp4` | 70 frames, 512x512, 30 fps |
| PegInsertionSide | `output_peg_state_motionplanning_resume/videos/peg_baseline.mp4` | 192 frames, 512x512, 30 fps |

训练曲线：

| 任务 | 曲线 |
|---|---|
| PickCube | `output_pickcube_state_final/training_curves.png` |
| Peg current baseline | `output_peg_state_motionplanning_resume/training_curves.png` |
| Peg combined attempt | `output_peg_state_combined/training_curves.png` |

## 8. 备份

按要求，阶段性成果已备份到：

```text
D:\备份
```

最近一次完整备份：

```text
D:\备份\Second_State_Diffusion_Policy_20260420_082822
```

说明：备份时排除了 `.git`、`.venv`、`.mambaenv`、`__pycache__`、`*.pyc`。这些环境目录可以重建，训练数据、日志、checkpoint、曲线和视频均已保留。

## 9. 给后续研究人员的注意事项

1. 不要把 Peg 写成稳定收敛。当前更准确的表述是：`PegInsertionSide-v1` 已形成可运行 baseline，30 episode 成功率约 30%，仍需任务3/4继续优化。
2. 后续比较模型时，请固定至少一组确认评估：
   - `episodes=30`
   - `max_steps=350`
   - `action_horizon=10`
   - `inference_steps=20`
   - checkpoint 使用 `policy_best.pt`
3. PickCube 可作为 sanity check。如果新改动导致 PickCube 低于 90%，需要先排查评估或训练逻辑。
4. Peg 失败 episode 经常 reward 很高，说明策略接近插入但末端不稳。后续优化应重点关注末端对准和插入阶段，而不是只看总 loss。
5. 所有新实验都应至少保存：
   - `metrics.csv`
   - `training_curves.png`
   - `policy_best.pt`
   - `policy_last.pt`
   - 评估 CSV
   - 一段成功或失败代表性视频
