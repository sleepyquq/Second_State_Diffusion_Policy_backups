# Windows 云主机 Codex 交接说明

这份说明给 **Windows 云主机上的 Codex** 使用。当前项目目标是完成成员分工第二项：

> 策略实现与训练：负责 Diffusion Policy 的适配与训练。

当前本地机器不能承担正式训练，因此本仓库已整理为“上传到 Windows CUDA 云主机后先做最小闭环，再扩大训练”的形态。

## 1. 当前项目状态

已完成：

- 第一阶段材料：ManiSkill3 环境与任务配置说明、任务探测结果、任务跑通脚本。
- 第二阶段初始脚本：
  - `collect_demos.py`：收集专家演示数据，保存为 HDF5。
  - `train_dp.py`：state 模式 Diffusion Policy 训练与评估脚本。
  - `requirements_cloud.txt`：云端依赖清单。
  - `云端训练说明.md`：云端训练操作步骤。

`train_dp.py` 当前只支持 **state 模式训练与评估**。RGBD 视觉训练还没有实现视觉编码器，不要直接把正式实验切到 RGBD。

## 2. Windows 云主机环境优先级

请优先确认 Windows 云主机有 NVIDIA CUDA GPU。推荐：

- State 模式调试：T4 / RTX 3060 / RTX 4060 起步。
- State 模式正式训练：RTX 3090 / RTX 4090 / A10。
- RGBD 后续训练：建议 24GB 显存。

建议 Python 版本：3.10 或 3.11。

Windows 云主机需要确认：

- NVIDIA 驱动可用，`nvidia-smi` 能运行。
- Python 是 64-bit。
- 项目路径不要放在包含特殊权限限制的位置，建议放到 `C:\workspace\Second_State_Diffusion_Policy` 或 `D:\workspace\Second_State_Diffusion_Policy`。
- 如果通过 VS Code Remote SSH 操作，确认 VS Code 左下角已经显示连接到 Windows 云主机。

## 3. Windows 云端环境配置

进入仓库后，在 PowerShell 中执行：

```powershell
cd C:\workspace\Second_State_Diffusion_Policy
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 禁止激活脚本，可在当前会话临时放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

先安装匹配云主机 CUDA 的 PyTorch。CUDA 12.1 示例：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_cloud.txt
```

如果云主机是 CUDA 11.8，把 `cu121` 改为 `cu118`。

## 4. 接手后第一批验证命令

先确认 CUDA 和 ManiSkill：

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import gymnasium, mani_skill; print(gymnasium.__version__); print(mani_skill.__version__)"
nvidia-smi
```

必须看到 `torch.cuda.is_available()` 为 `True`，否则不要做正式训练。

然后做脚本语法和环境 step 验证：

```powershell
python -m py_compile run_tasks.py collect_demos.py train_dp.py
python run_tasks.py --worker PickCube-v1 state 1 2
python run_tasks.py --worker PegInsertionSide-v1 state 1 2
```

## 5. 最小训练闭环

第一目标不是成功率，而是确认“数据收集 -> 训练 -> 评估 -> 保存 checkpoint”整条链路不崩。

先收集 3 条 PickCube state 专家轨迹：

```powershell
python collect_demos.py \
  --env PickCube-v1 \
  --obs-mode state \
  --n-demos 3 \
  --max-steps 500 \
  --output demos_PickCube-v1_state_smoke.h5
```

PowerShell 不支持 Bash 风格的反斜杠换行时，使用单行命令：

```powershell
python collect_demos.py --env PickCube-v1 --obs-mode state --n-demos 3 --max-steps 500 --output demos_PickCube-v1_state_smoke.h5
```

注意日志里必须出现类似：

```text
[专家策略] 使用 PickCube 内置运动规划专家
```

如果出现“内置专家不可用，使用随机动作”，不要继续训练，需要先修 `collect_demos.py` 的专家策略导入或改用 ManiSkill 官方演示收集流程。

查看数据：

```powershell
python collect_demos.py --info --dataset demos_PickCube-v1_state_smoke.h5
```

跑 1 epoch 训练 smoke test：

```powershell
python train_dp.py \
  --data demos_PickCube-v1_state_smoke.h5 \
  --env PickCube-v1 \
  --obs-mode state \
  --epochs 1 \
  --batch-size 16 \
  --eval-every 1 \
  --eval-episodes 1 \
  --output-dir output_smoke
```

PowerShell 单行版：

```powershell
python train_dp.py --data demos_PickCube-v1_state_smoke.h5 --env PickCube-v1 --obs-mode state --epochs 1 --batch-size 16 --eval-every 1 --eval-episodes 1 --output-dir output_smoke
```

成功标准：

- 训练不报错。
- 评估能跑完。
- 生成 `output_smoke/checkpoints/policy_best.pt`。

## 6. 正式训练建议

最小闭环通过后，再扩大到 100 条 PickCube state 演示：

```powershell
python collect_demos.py \
  --env PickCube-v1 \
  --obs-mode state \
  --n-demos 100 \
  --max-steps 500 \
  --output demos_PickCube-v1_state_100.h5
```

训练：

```powershell
New-Item -ItemType Directory -Force logs
python train_dp.py \
  --data demos_PickCube-v1_state_100.h5 \
  --env PickCube-v1 \
  --obs-mode state \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-4 \
  --weight-decay 1e-6 \
  --ema-decay 0.9999 \
  --eval-every 5 \
  --eval-episodes 10 \
  --output-dir output_pickcube_state \
  *> logs\pickcube_state_train.log
```

PowerShell 单行版：

```powershell
New-Item -ItemType Directory -Force logs
python train_dp.py --data demos_PickCube-v1_state_100.h5 --env PickCube-v1 --obs-mode state --epochs 100 --batch-size 256 --lr 1e-4 --weight-decay 1e-6 --ema-decay 0.9999 --eval-every 5 --eval-episodes 10 --output-dir output_pickcube_state *> logs\pickcube_state_train.log
```

训练过程中也可以不重定向日志，直接在 VS Code 远程终端观察输出。长训练建议保持远程会话稳定，或使用 Windows Terminal / PowerShell 后台任务。

PegInsertionSide 更难，建议等 PickCube state 训练闭环稳定后再做。

## 7. 已知注意事项

- `train_dp.py` 当前不支持 RGBD 评估，误传 `--obs-mode rgbd` 会主动报错。
- `train_dp.py` 会从 HDF5 数据自动推断 state 观测维度，也可以手动传：
  - PickCube-v1：`--obs-dim 42`
  - PegInsertionSide-v1：`--obs-dim 43`
- Windows 下 RGBD 渲染需要 `render_backend='sapien_cpu'`，但云端 Linux/CUDA 通常优先用 GPU 渲染。
- 如果 ManiSkill 首次运行下载资产失败，请先处理云主机网络或手动配置资产缓存。

## 8. 报错回传格式

如果训练或验证失败，请把以下内容发回给本地 Codex：

1. 运行的完整命令。
2. 最后 80 行日志或完整 traceback。
3. 环境确认输出：
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   python -c "import gymnasium, mani_skill; print(gymnasium.__version__, mani_skill.__version__)"
   ```
4. 当前目录下生成的 `.h5`、`output_*`、`logs/*` 文件名。

## 9. 当前优先级

请按顺序推进：

```text
CUDA 环境确认
-> PickCube state 2 步验证
-> 3 条专家轨迹
-> 1 epoch smoke train
-> 100 条 PickCube state 正式训练
-> PegInsertionSide state
-> RGBD 视觉编码器与视觉训练
```

不要跳过 smoke test 直接长训。
