"""
ManiSkill3 专家演示数据收集脚本
================================================
用途：为 Diffusion Policy 收集高质量专家演示轨迹，保存为 HDF5 格式。

支持任务：
  - PickCube-v1         (obs_dim=42, action_dim=8)
  - PegInsertionSide-v1 (obs_dim=43, action_dim=8)

支持观测模式：
  - state  : 状态向量（推荐先用此模式验证 DP 可训练性）
  - rgbd   : RGB + 深度图（视觉策略最终目标）

运行示例：
  # 收集 PickCube state 模式演示（100 条）
  python collect_demos.py --env PickCube-v1 --obs-mode state --n-demos 100

  # 收集 PegInsertionSide rgbd 模式演示（200 条）
  python collect_demos.py --env PegInsertionSide-v1 --obs-mode rgbd --n-demos 200

  # 查看已有数据集信息
  python collect_demos.py --info --dataset demos_PickCube-v1_state.h5

输出文件：demos_{env_id}_{obs_mode}.h5

Windows 注意：RGBD 模式已自动设置 render_backend='sapien_cpu'
"""

import argparse
import platform
import time
import warnings
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401  必须导入完成环境注册
import numpy as np
import torch
import h5py


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────

def obs_to_numpy(obs):
    """将观测从 Tensor/dict 转换为 numpy（去掉 batch 维度）。"""
    if isinstance(obs, torch.Tensor):
        arr = obs.cpu().numpy()
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]          # (1, dim) → (dim,)
        return arr
    elif isinstance(obs, dict):
        return {k: obs_to_numpy(v) for k, v in obs.items()}
    elif isinstance(obs, np.ndarray):
        if obs.ndim > 1 and obs.shape[0] == 1:
            return obs[0]
        return obs
    return obs


def action_to_numpy(action):
    """将动作转换为 numpy。"""
    if isinstance(action, torch.Tensor):
        return action.cpu().numpy()
    return np.array(action, dtype=np.float32)


def flatten_obs_dict(obs_dict: dict) -> dict:
    """
    将嵌套 dict 展平为 key=path 的一级 dict。
    例如 {"base_camera": {"rgb": ...}} → {"base_camera/rgb": ...}
    """
    result = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key + "/")
            else:
                result[key] = v
    _flatten(obs_dict)
    return result


# ─────────────────────────────────────────────────────────────────
# 专家策略
# ─────────────────────────────────────────────────────────────────

class ExpertPolicy:
    """
    包装 ManiSkill3 内置专家控制器（CPU-based Motion Planning）。

    如果内置专家不可用，回退到随机动作（仅用于测试管道，不能用于训练）。
    """

    def __init__(self, env_id: str, env):
        self.env_id = env_id
        self.env = env
        self._expert = None
        self._use_random = False
        self._init_expert()

    def _init_expert(self):
        """尝试初始化内置专家。"""
        try:
            # ManiSkill3 部分任务提供内置 solution
            from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
            from mani_skill.examples.motionplanning.panda.solutions.pick_cube import (
                SolutionPolicy as PickCubeSolution
            )
            if "PickCube" in self.env_id:
                self._expert = PickCubeSolution(self.env)
                print("  [专家策略] 使用 PickCube 内置运动规划专家")
                return
        except Exception:
            pass

        try:
            from mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side import (
                SolutionPolicy as PegSolution
            )
            if "PegInsertion" in self.env_id:
                self._expert = PegSolution(self.env)
                print("  [专家策略] 使用 PegInsertionSide 内置运动规划专家")
                return
        except Exception:
            pass

        # 兜底：用 ManiSkill3 通用运动规划接口
        try:
            from mani_skill.utils.wrappers.gymnasium import ManiSkillWrapper
            print("  [专家策略] 尝试通用运动规划接口...")
        except Exception:
            pass

        # 最终兜底：随机动作（仅用于调试管道）
        print("  [警告] 内置专家不可用，使用随机动作（仅供管道调试，不可用于训练！）")
        self._use_random = True

    def reset(self, obs, info):
        """每次 episode reset 后调用，让专家初始化内部状态。"""
        if self._expert is not None:
            try:
                self._expert.reset(obs, info)
            except Exception:
                pass

    def get_action(self, obs, info) -> np.ndarray:
        """根据观测返回专家动作。"""
        if self._use_random:
            return self.env.action_space.sample()
        try:
            action = self._expert.get_action(obs, info)
            return action_to_numpy(action)
        except Exception as e:
            # 专家推理失败时回退随机
            warnings.warn(f"专家推理异常，使用随机动作：{e}")
            return self.env.action_space.sample()


# ─────────────────────────────────────────────────────────────────
# 数据收集主函数
# ─────────────────────────────────────────────────────────────────

def collect_demonstrations(
    env_id: str,
    obs_mode: str,
    n_demos: int,
    max_steps_per_episode: int,
    output_path: str,
    seed_offset: int = 0,
    max_attempts: int | None = None,
    allow_random_expert: bool = False,
) -> None:
    """
    收集专家演示轨迹并保存到 HDF5 文件。

    HDF5 文件结构：
      /episode_0/
        obs/          (轨迹长度 T, ...) 各观测字段
        actions       (T, 8)
        rewards       (T,)
        success       标量 bool
        env_id        字符串
        obs_mode      字符串
      /episode_1/
        ...
      /metadata/
        n_demos       成功收集的条数
        obs_mode      字符串
        env_id        字符串
        action_dim    8
        obs_dim       42 / 43 / ...
    """
    warnings.filterwarnings("ignore")

    # ── 决定渲染后端 ───────────────────────────────────────────────
    is_windows = platform.system() == "Windows"
    needs_render = obs_mode not in ("state", "state_dict", "none")
    render_backend = "sapien_cpu" if (is_windows and needs_render) else "gpu"

    if is_windows and needs_render:
        print(f"  [Windows] RGBD 渲染使用 render_backend='sapien_cpu'")

    # ── 创建环境 ──────────────────────────────────────────────────
    print(f"\n正在创建环境 {env_id}（obs_mode={obs_mode}）...")
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        render_backend=render_backend,
    )
    print(f"  环境创建完成。动作空间：{env.action_space}")

    # ── 初始化专家 ────────────────────────────────────────────────
    expert = ExpertPolicy(env_id, env)
    if expert._use_random and not allow_random_expert:
        env.close()
        raise RuntimeError(
            "未找到 ManiSkill 内置专家策略，已停止收集，避免用随机动作生成训练数据。"
            "如果只想测试 HDF5 管道，请显式添加 --allow-random-expert。"
        )

    # ── 收集轨迹 ──────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = max_attempts or max(n_demos * 20, n_demos + 10)

    success_count = 0
    attempt_count = 0
    state_obs_dim = None
    t_start = time.time()

    with h5py.File(str(out_path), "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["env_id"]   = env_id
        meta.attrs["obs_mode"] = obs_mode

        print(f"\n开始收集演示（目标：{n_demos} 条成功轨迹）...")
        print(f"  {'进度':>6}  {'尝试':>6}  {'成功率':>8}  {'耗时':>8}")
        print(f"  {'-'*40}")

        while success_count < n_demos and attempt_count < max_attempts:
            seed = seed_offset + attempt_count
            obs, info = env.reset(seed=seed)
            expert.reset(obs, info)

            obs_list    = []
            action_list = []
            reward_list = []
            done        = False
            step        = 0

            while not done and step < max_steps_per_episode:
                action = expert.get_action(obs, info)

                obs_np = obs_to_numpy(obs)
                obs_list.append(obs_np)
                action_list.append(action_to_numpy(action))

                obs, reward, terminated, truncated, info = env.step(action)
                reward_list.append(float(reward))
                done  = terminated or truncated or info.get("success", False)
                step += 1

            success = bool(info.get("success", False))
            attempt_count += 1

            if success:
                # ── 保存这条轨迹 ───────────────────────────────────
                ep_grp = f.create_group(f"episode_{success_count}")
                ep_grp.attrs["success"]  = True
                ep_grp.attrs["env_id"]   = env_id
                ep_grp.attrs["obs_mode"] = obs_mode
                ep_grp.attrs["length"]   = len(action_list)

                obs_grp = ep_grp.create_group("obs")
                if isinstance(obs_list[0], dict):
                    # RGBD 模式：dict of arrays
                    flat_list = [flatten_obs_dict(o) for o in obs_list]
                    keys = flat_list[0].keys()
                    for key in keys:
                        arr = np.stack([o[key] for o in flat_list], axis=0)
                        obs_grp.create_dataset(
                            key.replace("/", "_"),  # h5py 不支持 "/" 作为 key 内字符
                            data=arr,
                            compression="gzip", compression_opts=4,
                        )
                else:
                    # State 模式：ndarray
                    obs_arr = np.stack(obs_list, axis=0)   # (T, obs_dim)
                    state_obs_dim = int(obs_arr.shape[-1])
                    obs_grp.create_dataset(
                        "state", data=obs_arr,
                        compression="gzip", compression_opts=4,
                    )

                ep_grp.create_dataset(
                    "actions",
                    data=np.stack(action_list, axis=0),    # (T, 8)
                    compression="gzip", compression_opts=4,
                )
                ep_grp.create_dataset(
                    "rewards",
                    data=np.array(reward_list),             # (T,)
                )

                success_count += 1
                elapsed = time.time() - t_start
                rate    = success_count / attempt_count * 100
                print(f"  {success_count:>6}/{n_demos:<6}  {attempt_count:>5}次  "
                      f"{rate:>7.1f}%  {elapsed:>6.1f}s")

        # 写入元数据
        meta.attrs["n_demos"]    = success_count
        meta.attrs["action_dim"] = 8
        if obs_mode == "state" and state_obs_dim is not None:
            meta.attrs["obs_dim"] = state_obs_dim

    env.close()
    total_time = time.time() - t_start
    if success_count < n_demos:
        raise RuntimeError(
            f"只收集到 {success_count}/{n_demos} 条成功轨迹，"
            f"已达到最大尝试次数 {attempt_count}/{max_attempts}。"
        )

    print(f"\n收集完成！")
    print(f"  成功轨迹：{success_count} 条")
    print(f"  总尝试次数：{attempt_count}")
    print(f"  专家成功率：{success_count/attempt_count*100:.1f}%")
    print(f"  总耗时：{total_time:.1f}s")
    print(f"  输出文件：{out_path}")


# ─────────────────────────────────────────────────────────────────
# 数据集信息查看
# ─────────────────────────────────────────────────────────────────

def print_dataset_info(dataset_path: str):
    """打印 HDF5 数据集的详细信息。"""
    path = Path(dataset_path)
    if not path.exists():
        print(f"文件不存在：{path}")
        return

    with h5py.File(str(path), "r") as f:
        print(f"\n数据集信息：{path}")
        print(f"{'='*50}")

        if "metadata" in f:
            meta = f["metadata"]
            print(f"  env_id   : {meta.attrs.get('env_id', 'N/A')}")
            print(f"  obs_mode : {meta.attrs.get('obs_mode', 'N/A')}")
            print(f"  n_demos  : {meta.attrs.get('n_demos', 'N/A')}")
            print(f"  action_dim: {meta.attrs.get('action_dim', 'N/A')}")

        ep_keys = [k for k in f.keys() if k.startswith("episode_") or k.startswith("traj_")]
        print(f"\n  轨迹数量：{len(ep_keys)}")

        if ep_keys:
            ep0 = f[ep_keys[0]]
            print(f"\n  第一条轨迹（{ep_keys[0]}）：")
            length = ep0.attrs.get("length", ep0["actions"].shape[0] if "actions" in ep0 else "?")
            print(f"    长度：{length} 步")
            print(f"    观测字段：")
            if "obs" in ep0 and isinstance(ep0["obs"], h5py.Dataset):
                ds = ep0["obs"]
                print(f"      obs: shape={ds.shape}, dtype={ds.dtype}")
            elif "obs" in ep0:
                for k in ep0["obs"].keys():
                    ds = ep0["obs"][k]
                    print(f"      {k}: shape={ds.shape}, dtype={ds.dtype}")
            print(f"    actions: shape={ep0['actions'].shape}")
            if "rewards" in ep0:
                print(f"    rewards: shape={ep0['rewards'].shape}")

        # 统计所有轨迹长度
        lengths = []
        for k in ep_keys:
            try:
                lengths.append(f[k].attrs.get("length", 0))
            except Exception:
                pass
        if lengths:
            print(f"\n  轨迹长度统计：")
            print(f"    平均：{np.mean(lengths):.1f} 步")
            print(f"    最短：{np.min(lengths)} 步")
            print(f"    最长：{np.max(lengths)} 步")


# ─────────────────────────────────────────────────────────────────
# 数据加载工具（供 Diffusion Policy 训练时使用）
# ─────────────────────────────────────────────────────────────────

class DemoDataset(torch.utils.data.Dataset):
    """
    读取收集的 HDF5 演示数据，输出 (obs_seq, action_seq) 供 Diffusion Policy 训练。

    使用示例：
        dataset = DemoDataset(
            "demos_PickCube-v1_state.h5",
            obs_horizon=2,    # To：每个样本使用的历史观测帧数
            pred_horizon=16,  # Tp：预测动作序列长度
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        for obs_seq, action_seq in loader:
            # obs_seq:    (B, To, obs_dim)
            # action_seq: (B, Tp, action_dim)
            ...
    """

    def __init__(
        self,
        h5_path: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        pad_before: int = 1,     # 轨迹开头用第一帧 obs 填充
        pad_after: int = 7,      # 轨迹结尾用最后一帧 action 填充
    ):
        super().__init__()
        self.obs_horizon  = obs_horizon
        self.pred_horizon = pred_horizon
        self.pad_before   = pad_before
        self.pad_after    = pad_after

        self._load_data(h5_path)

    def _load_data(self, h5_path: str):
        """加载所有轨迹到内存（数据量小时推荐；大数据集改为懒加载）。"""
        self.episodes = []
        with h5py.File(h5_path, "r") as f:
            ep_keys = sorted(
                [k for k in f.keys() if k.startswith("episode_") or k.startswith("traj_")],
                key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x,
            )
            for k in ep_keys:
                ep = f[k]
                if "obs/state" in ep:
                    obs = ep["obs/state"][:].astype(np.float32)
                elif "obs" in ep and isinstance(ep["obs"], h5py.Dataset):
                    # ManiSkill 官方 replay_trajectory 的 state 轨迹格式：
                    # /traj_i/obs 是 (T+1, obs_dim)，/traj_i/actions 是 (T, action_dim)。
                    obs = ep["obs"][:].astype(np.float32)
                elif "obs" in ep:
                    # rgbd 模式：简单拼接 qpos + qvel
                    obs_key_list = ["obs/agent_qpos", "obs/agent_qvel",
                                    "obs/extra_tcp_pose"]
                    obs_parts = [ep[k][:] for k in obs_key_list if k in ep]
                    obs = np.concatenate(obs_parts, axis=-1).astype(np.float32)
                else:
                    raise KeyError(f"{h5_path}:{k} 中没有可用 obs 字段")

                actions = ep["actions"][:].astype(np.float32)  # (T, 8)
                if len(obs) == len(actions) + 1:
                    obs = obs[:-1]
                elif len(obs) != len(actions):
                    min_len = min(len(obs), len(actions))
                    obs = obs[:min_len]
                    actions = actions[:min_len]
                self.episodes.append({"obs": obs, "actions": actions})

        # 构建样本索引 (ep_idx, t_idx)
        self.indices = []
        for ep_idx, ep in enumerate(self.episodes):
            T = len(ep["actions"])
            for t in range(T):
                self.indices.append((ep_idx, t))

        print(f"  [DemoDataset] 加载 {len(self.episodes)} 条轨迹，"
              f"共 {len(self.indices)} 个样本")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        ep = self.episodes[ep_idx]
        obs_arr    = ep["obs"]      # (T, obs_dim)
        action_arr = ep["actions"]  # (T, 8)
        T = len(action_arr)

        # ── 观测序列 (To 帧) ──────────────────────────────
        obs_seq = []
        for i in range(self.obs_horizon):
            t_obs = t - (self.obs_horizon - 1 - i)
            t_obs = max(0, min(t_obs, T - 1))   # 边界 pad
            obs_seq.append(obs_arr[t_obs])
        obs_seq = np.stack(obs_seq, axis=0)      # (To, obs_dim)

        # ── 动作序列 (Tp 帧) ──────────────────────────────
        action_seq = []
        for i in range(self.pred_horizon):
            t_act = t + i
            t_act = min(t_act, T - 1)            # 结尾 pad
            action_seq.append(action_arr[t_act])
        action_seq = np.stack(action_seq, axis=0) # (Tp, 8)

        return (
            torch.from_numpy(obs_seq),            # (To, obs_dim)
            torch.from_numpy(action_seq),          # (Tp, 8)
        )


# ─────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ManiSkill3 专家演示数据收集工具（Diffusion Policy 专用）"
    )
    parser.add_argument(
        "--env", default="PickCube-v1",
        choices=["PickCube-v1", "PegInsertionSide-v1"],
        help="任务环境 ID"
    )
    parser.add_argument(
        "--obs-mode", default="state",
        choices=["state", "rgbd"],
        help="观测模式（state=状态向量，rgbd=图像+深度）"
    )
    parser.add_argument(
        "--n-demos", type=int, default=100,
        help="收集的成功演示条数"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="每条轨迹的最大步数"
    )
    parser.add_argument(
        "--output", default=None,
        help="输出 HDF5 文件路径（默认：demos_{env}_{obs_mode}.h5）"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="随机种子偏移（避免多次收集数据重复）"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help="最大尝试 episode 数（默认 max(n_demos*20, n_demos+10)）"
    )
    parser.add_argument(
        "--allow-random-expert", action="store_true",
        help="仅用于测试数据管道：允许内置专家缺失时使用随机动作"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="查看已有数据集信息（需配合 --dataset 使用）"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="配合 --info 使用，指定数据集路径"
    )

    args = parser.parse_args()

    if args.info:
        ds_path = args.dataset or f"demos_{args.env}_{args.obs_mode}.h5"
        print_dataset_info(ds_path)
        return

    output_path = args.output or f"demos_{args.env}_{args.obs_mode}.h5"

    print(f"收集参数：")
    print(f"  任务：{args.env}")
    print(f"  观测模式：{args.obs_mode}")
    print(f"  目标演示条数：{args.n_demos}")
    print(f"  每集最大步数：{args.max_steps}")
    print(f"  输出文件：{output_path}")
    print(f"  系统：{platform.system()}")

    collect_demonstrations(
        env_id=args.env,
        obs_mode=args.obs_mode,
        n_demos=args.n_demos,
        max_steps_per_episode=args.max_steps,
        output_path=output_path,
        seed_offset=args.seed_offset,
        max_attempts=args.max_attempts,
        allow_random_expert=args.allow_random_expert,
    )

    # 收集完成后打印数据集信息
    print_dataset_info(output_path)

    print(f"""
下一步（Diffusion Policy 训练）：
  from collect_demos import DemoDataset
  dataset = DemoDataset('{output_path}', obs_horizon=2, pred_horizon=16)
  loader  = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
  # → 接入你的 Diffusion Policy 训练循环
""")


if __name__ == "__main__":
    main()
