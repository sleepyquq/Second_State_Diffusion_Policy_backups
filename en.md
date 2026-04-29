# Task 2: Diffusion Policy Training Technical Record

**Date:** April 20, 2026  
**Project Directory:** `D:\Second_State_Diffusion_Policy`

---

## 1. Task Objectives

The goal of Task 2 is to complete two Diffusion Policy baselines under standard ManiSkill task conditions:

1. **PickCube-v1**: Run the picking task and achieve convergence.
2. **PegInsertionSide-v1**: Run the insertion task and achieve convergence as much as possible.
3. **Deliverables**: Preserve training curves, evaluation data, checkpoints, logs, and OpenCV benchmark videos as a baseline for Tasks 3 and 4.

---

## 2. Environment Setup

The current machine is a newly created cloud instance. Initially, the Python/pip environment lacked ManiSkill training dependencies, specifically `Pinocchio`. The resolution was as follows:

1. Installed **Python 3.11** and created a `.venv`.
2. Due to incomplete `Pinocchio` dependencies in Windows pip, switched to `micromamba` to create a `.mambaenv`.
3. Installed `pinocchio` via `.mambaenv`, then used pip for PyTorch CUDA and project dependencies.
4. **GPU Verification**:
    - PyTorch: `2.5.1+cu121`
    - CUDA available: `True`
    - GPU: `NVIDIA GeForce RTX 3080`
5. **Runtime Constants**:
    - `KMP_DUPLICATE_LIB_OK=TRUE`: Bypasses OpenMP duplicate loading issues on Windows.
    - `GIT_PYTHON_REFRESH=quiet`: Avoids GitPython errors caused by the absence of `git.exe`.

> [!NOTE]
> **GPU Utilization:** Training runs on CUDA; however, Windows Task Manager may default to 3D/Copy graphs rather than CUDA/Compute. Data replay and environment stepping also cause significant CPU usage.

---

## 3. Data Preparation

While the original instructions mentioned `pd_ee_delta_pose`, the current ManiSkill version uses `pd_joint_delta_pos` (Action Dimension: 8). Official ManiSkill demos were replayed:

| Task | Data Source | Trajectories | Remarks |
| :--- | :--- | :--- | :--- |
| PickCube | motionplanning | 100 | Used for final picking training |
| PegInsertionSide | motionplanning | 165 | Final insertion baseline |
| PegInsertionSide | RL | 195 | Poor individual/combined performance |
| PegInsertionSide | combined | 360 | MP + RL; ultimately discarded |

---

## 4. Code Modifications

Key modifications to the training scripts:

* **`collect_demos.py`**: Added support for ManiSkill official `traj_*` HDF5 structures and handled alignment for `obs` (T+1) and `actions` (T).
* **`train_dp.py`**:
    * Fixed `DDPMScheduler` `register_buffer` issues.
    * Implemented deterministic **DDIM-style sampling** to replace unstable reverse sampling.
    * Added **Positional Embeddings (`pos_emb`)** to the action sequence.
    * Implemented **Receding Horizon** during evaluation (executing `action_horizon` steps per sampling).
* **New Utility Scripts**:
    - `evaluate_checkpoint.py`: For batch comparison of hyperparameters.
    - `record_policy_video.py`: Rollout recording using OpenCV.

---

## 5. PickCube: Training & Results

Final model performance after fixing planning stability:

| Metric | Value |
| :--- | :--- |
| **Final Success Rate** | **100.0%** |
| Eval Episodes | 10 |
| Avg Reward | 20.41 |
| Avg Steps | 66.2 |
| Final Loss | ~0.0504 |

---

## 6. PegInsertionSide: Training & Results

Peg insertion is significantly harder, particularly in the final precision phase.

### 6.1 Resumed Training Results
Resuming training for an additional 250 epochs with a lower learning rate (`5e-5`):

| Epoch | Train Loss | Eval Success | Mean Reward | Mean Steps |
| :--- | :--- | :--- | :--- | :--- |
| 125 | 0.0368 | 50.0% | 69.36 | 195.6 |
| 250 | 0.0308 | 40.0% | 76.75 | 210.3 |

**Large Sample Confirmation (30 episodes)**:
- `action_horizon=12, max_steps=250`: **30.0% Success**
- `action_horizon=10, max_steps=350`: **30.0% Success**

---

## 7. OpenCV Benchmark Videos

Videos generated with `--require-success`:
- **PickCube**: Seed 0, Success in 69 steps.
- **PegInsertionSide**: Seed 9, Success in 191 steps.

---

## 8. Current Final Artifacts

* **Checkpoints**: `policy_best.pt` for both tasks.
* **Logs**: Stored in `logs/`.
* **Visuals**: `metrics.csv`, `training_curves.png`, and `.mp4` baseline videos.

---

## 9. Backup History

Phase backups are stored in `D:\备份` with timestamps from `20260420_013127` through `20260420_082807`.

---

## 10. Future Recommendations

1. **PickCube**: Use as a stable baseline for Task 3/4.
2. **Peg**: Success rates can be improved by:
    - Transitioning to a **1D UNet** policy network.
    - Increasing high-quality motion planning data.
    - Analyzing failure cases to see if errors occur during alignment or insertion depth phases.
