
```markdown
# Task 2: Diffusion Policy Training Technical Record

**Record Time:** April 20, 2026  
**Project Directory:** `D:\Second_State_Diffusion_Policy`

## 1. Task Objectives

Task 2 requires completing two Diffusion Policy baselines under standard ManiSkill task conditions:

1. `PickCube-v1` grabbing task: run and converge.
2. `PegInsertionSide-v1` insertion task: run and converge as much as possible.
3. Preserve training curves, evaluation data, checkpoints, logs, and OpenCV benchmark videos as baselines for continued work in Task 3/Task 4.

## 2. Environment Setup

The current machine is a newly created cloud host; the local environment is the cloud environment. Initially, the Python/pip environment lacked ManiSkill training dependencies, especially Pinocchio, commonly used in robot simulation. The processing flow was as follows:

1. Installed Python 3.11 and created `.venv`.
2. Due to incomplete Pinocchio dependencies in the Windows pip environment, switched to `micromamba` to create `.mambaenv`.
3. Installed `pinocchio` in `.mambaenv`, then pip-installed PyTorch CUDA version and project dependencies.
4. GPU Verification:
   - PyTorch: `2.5.1+cu121`
   - CUDA available: `True`
   - GPU: `NVIDIA GeForce RTX 3080`
5. Training execution constants:
   - `KMP_DUPLICATE_LIB_OK=TRUE`: Bypasses OpenMP duplicate loading issues on Windows.
   - `GIT_PYTHON_REFRESH=quiet`: Avoids GitPython errors due to git.exe not being installed on the cloud host.

Regarding the issue of "GPU usage not visible in Task Manager": Training indeed runs on CUDA; the Windows Task Manager default chart may show 3D/Copy instead of CUDA/Compute (requires switching chart types). Data replay and environment stepping also incur high CPU usage, so GPU charts may not show continuous high load.

## 3. Data Preparation

The original project description mentioned `pd_ee_delta_pose`, but the current installed version of ManiSkill does not include this for `PickCube-v1` and `PegInsertionSide-v1`. The actual standard available control mode is `pd_joint_delta_pos`, with an action dimension of 8.

The machine could not smoothly use the motion planning example dependency `mplib`, so ManiSkill official demo download and replay were used instead:

1. Download official demos:
   - `python -m mani_skill.utils.download_demo PickCube-v1 -o official_demos`
   - `python -m mani_skill.utils.download_demo PegInsertionSide-v1 -o official_demos`
2. Replay to `state + pd_joint_delta_pos + physx_cpu` format.
3. Final data used for training:
   - PickCube: `official_demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`
   - Peg motionplanning: `official_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`
   - Peg RL replay: `official_demos/PegInsertionSide-v1/rl/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`
   - Peg combined attempt: `official_demos/PegInsertionSide-v1/combined/trajectory.state.pd_joint_delta_pos.physx_cpu.h5`

Replay Results:

| Task | Data Source | Trajectories | Remarks |
|---|---:|---:|---|
| PickCube | motionplanning | 100 | Used for final grabbing training |
| PegInsertionSide | motionplanning | 165 | Used for final insertion baseline |
| PegInsertionSide | RL | 195 | Poor individual performance, worsened after mixing |
| PegInsertionSide | combined | 360 | motionplanning + RL, ultimately not adopted |

## 4. Code Modifications

The original training script could not stably complete tasks. Key modifications are as follows:

1. `collect_demos.py`
   - Supports ManiSkill official `traj_*` HDF5 structures.
   - Compatible with `/traj_i/obs` and `/traj_i/actions` datasets.
   - Alignment and cropping for official data where `obs` length is `T+1` and `actions` length is `T`.

2. `train_dp.py`
   - Fixed the missing `register_buffer` issue when `DDPMScheduler` is not an `nn.Module`.
   - Fixed the `dtype` of timestep sinusoidal embedding.
   - Replaced unstable reverse sampling formulas with deterministic DDIM-style sampling.
   - Normalized action space is no longer hard-clipped to `[-1, 1]`; changed to clipping `[-5, 5]` in normalized space before de-normalizing into the environment's action range.
   - Added positional embeddings `pos_emb` to the action sequence; otherwise, the model struggles to distinguish the order of steps within the action horizon.
   - Changed evaluation phase to receding horizon: executing `action_horizon` consecutive actions after each sampling instead of only the first action.
   - Added `--inference-steps`, `--eval-max-steps`, and `--action-horizon` parameters.
   - Added output for `metrics.csv`, `training_curves.png`, `policy_last.pt`, and `policy_best.pt`.
   - Added `--resume` to continue training from existing checkpoint weights.
   - Added `sample_init` option to compare random initial noise vs. zero initial noise.

3. Added `evaluate_checkpoint.py`
   - Load checkpoints for standalone evaluation.
   - Supports batch comparison of `action_horizon`, `inference_steps`, `max_steps`, and `sample_init`.
   - Results appended to CSV.

4. Added `record_policy_video.py`
   - Uses OpenCV `cv2.VideoWriter` to record rollouts.
   - Supports multi-seed attempts; setting `--require-success` saves only successful rollouts.
   - Used to generate baseline videos for both tasks.

## 5. PickCube Training Process and Results

Early issues:
1. Only executing the first action of the sampled sequence caused continuous re-planning, resulting in a 0% success rate.
2. Lack of action sequence positional embeddings meant the model could not stably learn actions at different timesteps.
3. Reverse sampling and action clipping caused abnormal action distributions.

Fixed configuration for the final model:

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
Final Results:
| Metric | Value |
|---|---|
| Final Eval Success Rate | 100.0% |
| Evaluation Episodes | 10 |
| Average Reward | 20.41 |
| Average Steps | 66.2 |
| Final Loss | ~0.0504 |
| Video Seed | 0 |
| Video Rollout | Success, 69 steps |
Artifacts:
 * output_pickcube_state_final/checkpoints/policy_best.pt
 * output_pickcube_state_final/videos/pickcube_baseline.mp4
 * (Includes metrics.csv, training_curves.png, and logs)
## 6. PegInsertionSide Training Process and Results
The Peg insertion task is significantly more difficult than PickCube, primarily due to sensitivity to action precision and rollout stability during the final insertion phase.
### 6.1 RL Replay Data Attempt
Trained for 300 epochs using RL replay data alone:
 * Final success rate: 0%
 * Average reward: ~4.66
 * Conclusion: This data is unsuitable as the primary training set for the current state Diffusion Policy.
### 6.2 Motionplanning Data Round 1
Trained for 200 epochs using 165 motionplanning replay success trajectories:
 * Best in-training eval success: 10%
 * Final eval: 0%
 * Reward significantly improved over RL data (peak avg reward ~60), showing the policy learned the direction toward the target, but the terminal insertion was unstable.
### 6.3 Evaluation Parameter Matrix
Comparing action_horizon and inference_steps on Round 1 checkpoint:
 * Original action_horizon=8, max_steps=200 underestimated performance.
 * policy_last + action_horizon=12 + max_steps=250 reached 60% in small samples, but only 15% in 20-round confirmation; random fluctuations exist.
### 6.4 Resumed Motionplanning Training
Continued training for 250 epochs from Round 1 policy_last.pt:
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
Best training points:
| epoch | train loss | eval success | mean reward | mean steps |
|---|---|---|---|---|
| 125 | 0.0368 | 50.0% | 69.36 | 195.6 |
| 250 | 0.0308 | 40.0% | 76.75 | 210.3 |
Larger sample confirmation:
| Configuration | episodes | success |
|---|---|---|
| action_horizon=12, max_steps=250 | 30 | 30.0% |
| action_horizon=10, max_steps=350 | 30 | 30.0% |
Conclusion: Peg has achieved a runnable, reproducible baseline with successful rollouts, though the success rate remains significantly lower than PickCube. This result is suitable as a baseline for Tasks 3/4.
### 6.5 Combined Data Attempt
Merged motionplanning and RL replay (360 trajectories), trained for 300 epochs:
 * Best success rate: 10%
 * Final success rate: 10%
 * Conclusion: Mixing RL replay and motionplanning action distributions decreased stability; not adopted.
### 6.6 Sample_init Attempt
Comparing DDIM initial noise:
 * random: ~40% (kept as default).
 * zero: ~40%; no stable improvement.
## 7. OpenCV Benchmark Videos
Recording script: record_policy_video.py
PickCube:
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
Result: Seed 0 success, 69 steps.
PegInsertionSide:
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
Result: Seed 9 success, 191 steps.
## 8. Current Final Artifacts
PickCube Baseline:
 * output_pickcube_state_final/checkpoints/policy_best.pt
 * output_pickcube_state_final/metrics.csv
 * output_pickcube_state_final/training_curves.png
 * output_pickcube_state_final/videos/pickcube_baseline.mp4
PegInsertionSide Baseline:
 * output_peg_state_motionplanning_resume/checkpoints/policy_best.pt
 * output_peg_state_motionplanning_resume/metrics.csv
 * output_peg_state_motionplanning_resume/training_curves.png
 * output_peg_state_motionplanning_resume/eval_confirm_best.csv
 * output_peg_state_motionplanning_resume/videos/peg_baseline.mp4
(Includes auxiliary scripts and key logs)
## 9. Phase Backups
Phase results backed up to D:\备份. Key points include:
 * D:\备份\Second_State_Diffusion_Policy_20260420_013127
 * ... through ...
 * D:\备份\Second_State_Diffusion_Policy_20260420_082807
## 10. Future Recommendations
 1. PickCube can be used as a stable baseline for Tasks 3/4.
 2. Peg is currently a reproducible baseline but not high-success. Future improvements should prioritize:
   * Stronger policy networks (e.g., 1D UNet).
   * Increasing high-quality motionplanning data rather than mixing current RL replay.
   * Fine-grained action horizon search or multi-candidate trajectory selection for the terminal insertion phase.
   * Analyzing failed trajectories near success to confirm if issues are alignment, depth, or action saturation.
 3. Peg metrics should serve as a baseline floor; future improvements should exceed 30% confirmation success rate.
```

```
