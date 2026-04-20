# 任务2 Diffusion Policy 训练技术记录

记录时间：2026-04-20  
项目目录：`D:\Second_State_Diffusion_Policy`

## 1. 任务目标

任务2需要在 ManiSkill 标准任务条件下完成两个 Diffusion Policy 基线：

1. `PickCube-v1` 抓取任务跑通并收敛。
2. `PegInsertionSide-v1` 插入任务跑通并尽量收敛。
3. 留下训练曲线、评估数据、checkpoint、日志和 OpenCV 基准视频，作为任务3/任务4继续工作的基线。

## 2. 环境搭建

当前机器是刚创建的云主机，本地即云端环境。最初 Python/pip 环境缺少 ManiSkill 训练依赖，尤其是机器人仿真常用的 Pinocchio。处理流程如下：

1. 安装 Python 3.11，并创建过 `.venv`。
2. 由于 Windows pip 环境下 Pinocchio 依赖不完整，改用 `micromamba` 创建 `.mambaenv`。
3. 在 `.mambaenv` 中安装 `pinocchio`，再 pip 安装 PyTorch CUDA 版与项目依赖。
4. 验证 GPU：
   - PyTorch：`2.5.1+cu121`
   - CUDA available：`True`
   - GPU：`NVIDIA GeForce RTX 3080`
5. 运行训练时固定使用：
   - `KMP_DUPLICATE_LIB_OK=TRUE`：绕过 Windows 上 OpenMP 重复加载问题。
   - `GIT_PYTHON_REFRESH=quiet`：避免 GitPython 因云主机未安装 git.exe 报错。

关于“任务管理器看不到 GPU 占用”的问题：训练确实运行在 CUDA 上；Windows 任务管理器默认图表可能显示 3D/Copy 而不是 CUDA/Compute，需要切换图表类型。数据 replay 和环境步进也会有较多 CPU 占用，因此 GPU 图表不一定持续高负载。

## 3. 数据准备

项目原说明中提到 `pd_ee_delta_pose`，但当前 ManiSkill 安装版本对 `PickCube-v1` 和 `PegInsertionSide-v1` 的可用控制模式并不包含它，实际标准可用控制模式为 `pd_joint_delta_pos`，动作维度为 8。

本机无法顺利使用 motion planning 示例依赖 `mplib`，因此改用 ManiSkill 官方 demo 下载和 replay：

1. 下载官方 demo：
   - `python -m mani_skill.utils.download_demo PickCube-v1 -o official_demos`
   - `python -m mani_skill.utils.download_demo PegInsertionSide-v1 -o official_demos`
2. replay 到 `state + pd_joint_delta_pos + physx_cpu` 格式。
3. 最终用于训练的数据：
   - PickCube：`official_demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`
   - Peg motionplanning：`official_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`
   - Peg RL replay：`official_demos/PegInsertionSide-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`
   - Peg combined 尝试：`official_demos/PegInsertionSide-v1/combined/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`

Replay 结果：

| 任务 | 数据来源 | 轨迹数 | 备注 |
|---|---:|---:|---|
| PickCube | motionplanning | 100 | 用于最终抓取训练 |
| PegInsertionSide | motionplanning | 165 | 用于最终插入基线 |
| PegInsertionSide | RL | 195 | 单独训练效果差，混合后也变差 |
| PegInsertionSide | combined | 360 | motionplanning + RL，最终未采用 |

## 4. 代码修改

训练脚本原始状态不能直接稳定完成任务。关键修改如下：

1. `collect_demos.py`
   - 支持 ManiSkill 官方 `traj_*` HDF5 结构。
   - 兼容 `/traj_i/obs` 和 `/traj_i/actions` 数据集。
   - 对 `obs` 长度为 `T+1`、`actions` 长度为 `T` 的官方数据做对齐裁剪。

2. `train_dp.py`
   - 修复 `DDPMScheduler` 非 `nn.Module` 时缺少 `register_buffer` 的问题。
   - 修复 timestep sinusoidal embedding 的 dtype。
   - 用确定性 DDIM 风格采样替换原先不稳定的反向采样公式。
   - 归一化动作空间不再硬裁剪到 `[-1, 1]`，改为在归一化空间裁剪 `[-5, 5]`，最后再反归一化并送入环境动作范围。
   - 给动作序列加入位置嵌入 `pos_emb`，否则模型难以区分 action horizon 内每一步动作的顺序。
   - 评估阶段改为 receding horizon：每次采样后连续执行 `action_horizon` 个动作，而不是只执行第一个动作。
   - 增加 `--inference-steps`、`--eval-max-steps`、`--action-horizon` 参数。
   - 增加 `metrics.csv`、`training_curves.png`、`policy_last.pt`、`policy_best.pt` 输出。
   - 增加 `--resume`，可从已有 checkpoint 继续训练模型权重。
   - 增加 `sample_init` 选项，用于比较随机初始噪声和 zero 初始噪声。

3. 新增 `evaluate_checkpoint.py`
   - 加载 checkpoint 后单独评估。
   - 支持批量比较 `action_horizon`、`inference_steps`、`max_steps`、`sample_init`。
   - 结果追加写入 CSV。

4. 新增 `record_policy_video.py`
   - 使用 OpenCV `cv2.VideoWriter` 录制 rollout。
   - 支持多 seed 尝试，设置 `--require-success` 后只保存成功 rollout。
   - 已用于生成两个任务的基准视频。

## 5. PickCube 训练过程与结果

早期问题：

1. 只执行采样动作序列的第一个动作，导致策略不断重规划，抓取任务成功率为 0。
2. 没有动作序列位置嵌入，模型无法稳定学习不同时间步动作。
3. 反向采样和动作裁剪导致动作分布异常。

修复后使用如下配置训练最终模型：

```powershell
python train_dp.py `
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

最终结果：

| 指标 | 数值 |
|---|---:|
| 最终评估成功率 | 100.0% |
| 评估 episodes | 10 |
| 平均 reward | 20.41 |
| 平均步数 | 66.2 |
| 最终 loss | 约 0.0504 |
| 视频 seed | 0 |
| 视频 rollout | 成功，69 步 |

产物：

- `output_pickcube_state_final/checkpoints/policy_best.pt`
- `output_pickcube_state_final/checkpoints/policy_last.pt`
- `output_pickcube_state_final/metrics.csv`
- `output_pickcube_state_final/training_curves.png`
- `output_pickcube_state_final/videos/pickcube_baseline.mp4`
- `logs/pickcube_state_final.log`
- `logs/pickcube_record_video.log`

## 6. PegInsertionSide 训练过程与结果

Peg 插入任务明显比 PickCube 难，主要难点是末端插入阶段对动作精度和 rollout 稳定性非常敏感。

### 6.1 RL replay 数据尝试

使用 RL replay 数据单独训练 300 epoch：

- 最终成功率：0%
- 平均 reward：约 4.66
- 判断：该数据对当前 state Diffusion Policy 不适合作为主训练集。

### 6.2 motionplanning 数据第一轮

使用 165 条 motionplanning replay 成功轨迹训练 200 epoch：

- 最好训练内评估成功率：10%
- 最终 eval：0%
- reward 相比 RL 数据明显提高，最高阶段平均 reward 约 60，说明策略已经学到接近目标的动作方向，但末端插入不稳。

### 6.3 评估参数矩阵

对第一轮 checkpoint 比较 `action_horizon` 和 `inference_steps` 后发现：

- 原先 `action_horizon=8, max_steps=200` 会低估性能。
- `policy_last + action_horizon=12 + max_steps=250` 小样本可到 60%，但 20 轮确认只有 15%，存在随机波动。

### 6.4 续训 motionplanning

从第一轮 `policy_last.pt` 继续训练 250 epoch：

```powershell
python train_dp.py `
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

训练内最佳点：

| epoch | train loss | eval success | mean reward | mean steps |
|---:|---:|---:|---:|---:|
| 125 | 0.0368 | 50.0% | 69.36 | 195.6 |
| 250 | 0.0308 | 40.0% | 76.75 | 210.3 |

更大样本确认：

| 配置 | episodes | success |
|---|---:|---:|
| `action_horizon=12, max_steps=250` | 30 | 30.0% |
| `action_horizon=10, max_steps=350` | 30 | 30.0% |

结论：Peg 已得到可运行、可复现、有成功 rollout 的 Diffusion Policy 基线，但成功率仍明显低于 PickCube。该结果适合作为后续任务3/任务4的 baseline，不应被解读为高成功率最终策略。

### 6.5 combined 数据尝试

将 motionplanning 与 RL replay 合并为 360 条轨迹后重新训练 300 epoch：

- 最佳成功率：10%
- 最终成功率：10%
- 判断：RL replay 与 motionplanning 的动作分布混合后反而降低了策略稳定性，未采用。

### 6.6 sample_init 尝试

比较 DDIM 初始噪声：

| sample init | 最好小样本表现 | 判断 |
|---|---:|---|
| random | 40% 左右 | 保留为默认 |
| zero | 40% 左右 | 没有稳定提升 |

## 7. OpenCV 基准视频

录制脚本：`record_policy_video.py`

PickCube：

```powershell
python record_policy_video.py `
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

结果：seed 0 成功，69 步。

PegInsertionSide：

```powershell
python record_policy_video.py `
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

结果：seed 9 成功，191 步。

## 8. 当前最终产物

PickCube 最终基线：

- `output_pickcube_state_final/checkpoints/policy_best.pt`
- `output_pickcube_state_final/metrics.csv`
- `output_pickcube_state_final/training_curves.png`
- `output_pickcube_state_final/videos/pickcube_baseline.mp4`

PegInsertionSide 当前基线：

- `output_peg_state_motionplanning_resume/checkpoints/policy_best.pt`
- `output_peg_state_motionplanning_resume/metrics.csv`
- `output_peg_state_motionplanning_resume/training_curves.png`
- `output_peg_state_motionplanning_resume/eval_confirm_best.csv`
- `output_peg_state_motionplanning_resume/eval_confirm_best_ah10_max350.csv`
- `output_peg_state_motionplanning_resume/videos/peg_baseline.mp4`

辅助脚本：

- `evaluate_checkpoint.py`
- `record_policy_video.py`

关键日志：

- `logs/pickcube_state_final.log`
- `logs/pickcube_record_video.log`
- `logs/peg_state_motionplanning.log`
- `logs/peg_state_motionplanning_resume.log`
- `logs/peg_eval_confirm_best_ah12.log`
- `logs/peg_eval_confirm_best_ah10_max350.log`
- `logs/peg_record_video.log`

## 9. 阶段备份

每取得阶段性成果都备份到 `D:\备份`。本次主要备份点包括：

- `D:\备份\Second_State_Diffusion_Policy_20260420_013127`
- `D:\备份\Second_State_Diffusion_Policy_20260420_014232`
- `D:\备份\Second_State_Diffusion_Policy_20260420_023457`
- `D:\备份\Second_State_Diffusion_Policy_20260420_042336`
- `D:\备份\Second_State_Diffusion_Policy_20260420_053124`
- `D:\备份\Second_State_Diffusion_Policy_20260420_061800`
- `D:\备份\Second_State_Diffusion_Policy_20260420_065652`
- `D:\备份\Second_State_Diffusion_Policy_20260420_082807`

## 10. 后续建议

1. PickCube 已经可以作为稳定基线直接交给任务3/任务4使用。
2. Peg 当前是可成功、可复现的插入基线，但不是高成功率策略。后续若要继续提高，建议优先尝试：
   - 更强的策略网络或更接近官方 Diffusion Policy 的 1D UNet 结构。
   - 增加高质量 motionplanning replay 数据，而不是混入当前 RL replay。
   - 针对插入末端加入更细粒度的 action horizon 搜索或多候选轨迹选择。
   - 单独评估 Peg 成功判定附近的失败轨迹，确认是否是对准阶段、插入深度阶段或动作饱和导致失败。
3. 文档中的 Peg 指标应作为 baseline 下限使用：后续改进应至少超过 30% 的 30-episode 确认成功率，并保留同样格式的曲线、CSV 和视频。
