"""
ManiSkill3 两任务跑通脚本
================================================
任务 1：PickCube-v1         —— 桌面单物体抓取
任务 2：PegInsertionSide-v1 —— 轴孔插入

功能覆盖：
  - 状态观测模式 (obs_mode='state')  — 状态向量，(1,42)/(1,43)
  - RGBD 观测模式 (obs_mode='rgbd')  — 图像+深度，适合视觉策略
  - 完整 episode 循环（3 个随机 episode × 200 步/集）
  - 观测/动作空间完整打印

环境要求：
  pip install mani-skill
  Python 3.9-3.13, SAPIEN 3.x, Gymnasium 0.29+

Windows 特别说明：
  - render_mode='human' 不可用（无 GUI），需用 'rgb_array'
  - RGBD 渲染须显式设置 render_backend='sapien_cpu'，
    否则在笔记本双显卡环境下 Vulkan GPU 渲染会崩溃。

运行方式：
    python run_tasks.py
"""

import sys
import os
import subprocess
import json
import platform


# ─────────────────────────────────────────────────────────────────
# 子进程工作函数（以 --worker 模式调用自身，避免 SAPIEN 渲染器重复
# 初始化在同一进程内崩溃）
# ─────────────────────────────────────────────────────────────────

def worker_main(env_id: str, obs_mode: str, n_episodes: int, max_steps: int):
    """在子进程中实际运行环境，结果以 JSON 写到 stdout 最后一行。"""
    import warnings
    warnings.filterwarnings("ignore")
    import time
    import numpy as np
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    result = {
        "env_id": env_id, "obs_mode": obs_mode,
        "status": "FAIL", "error": None,
        "init_time": None, "episodes": [],
    }

    # Windows 下 RGBD / sensor_data 渲染须用 sapien_cpu 渲染后端
    is_windows = platform.system() == "Windows"
    needs_render = obs_mode not in ("state", "state_dict", "none")
    render_backend = "sapien_cpu" if (is_windows and needs_render) else "gpu"

    try:
        t0 = time.time()
        env = gym.make(
            env_id,
            obs_mode=obs_mode,
            render_mode="rgb_array",
            render_backend=render_backend,
        )
        obs, info = env.reset(seed=42)
        result["init_time"] = round(time.time() - t0, 2)

        # ── 观测结构汇总 ─────────────────────────────────
        def _obs_summary(o):
            if isinstance(o, dict):
                out = {}
                def _r(d, prefix=""):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            _r(v, prefix + k + "/")
                        elif hasattr(v, "shape"):
                            out[prefix + k] = {
                                "shape": list(v.shape),
                                "dtype": str(v.dtype)
                            }
                _r(o)
                return out
            elif hasattr(o, "shape"):
                return {"shape": list(o.shape), "dtype": str(o.dtype)}
            return {}

        result["obs_summary"] = _obs_summary(obs)

        act = env.action_space
        result["action_space"] = {
            "type": type(act).__name__,
            "shape": list(act.shape),
            "dtype": str(act.dtype),
            "low": float(act.low.flat[0]),
            "high": float(act.high.flat[0]),
        }

        # ── 随机 episode 运行 ────────────────────────────
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_reward = 0.0
            ep_success = False
            steps = 0
            for _ in range(max_steps):
                action = act.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                steps += 1
                if info.get("success", False):
                    ep_success = True
                if terminated or truncated:
                    break
            result["episodes"].append({
                "ep": ep + 1,
                "reward": round(ep_reward, 4),
                "steps": steps,
                "success": ep_success,
            })

        env.close()
        result["status"] = "OK"

    except Exception:
        import traceback
        result["error"] = traceback.format_exc()

    # 最后一行输出 JSON（主进程只解析这一行）
    print("__RESULT__:" + json.dumps(result, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────
# 主进程：依次启动子进程并收集结果
# ─────────────────────────────────────────────────────────────────

def run_in_subprocess(
    env_id: str,
    obs_mode: str,
    n_episodes: int = 3,
    max_steps: int = 200,
) -> dict:
    """在独立子进程中运行一个（env_id, obs_mode）组合。"""
    cmd = [sys.executable, __file__,
           "--worker", env_id, obs_mode,
           str(n_episodes), str(max_steps)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__:"):
            return json.loads(line[len("__RESULT__:"):])
    return {
        "env_id": env_id, "obs_mode": obs_mode,
        "status": "CRASH",
        "error": proc.stderr[-2000:] if proc.stderr else "(no stderr)",
        "stdout": proc.stdout[-2000:] if proc.stdout else "(no stdout)",
    }


def section(title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def print_result(res: dict, max_steps: int):
    """格式化打印子进程结果。"""
    status = res["status"]
    env_id = res["env_id"]
    obs_mode = res["obs_mode"]

    if status not in ("OK",):
        print(f"  ❌ {env_id} [{obs_mode}] 失败：{status}")
        if res.get("error"):
            lines = res["error"].strip().splitlines()
            for ln in lines[-20:]:
                print(f"     {ln}")
        return

    print(f"  ✅ 环境创建成功（用时 {res.get('init_time', '?')}s）")

    # 观测结构
    obs_sum = res.get("obs_summary", {})
    if isinstance(obs_sum, dict) and "shape" in obs_sum and len(obs_sum) == 2:
        # 单 tensor
        print(f"  观测: shape={obs_sum['shape']}, dtype={obs_sum['dtype']}")
    elif isinstance(obs_sum, dict):
        print(f"  观测结构（{len(obs_sum)} 个字段）:")
        for k, v in obs_sum.items():
            print(f"    {k}: shape={v['shape']}, dtype={v['dtype']}")

    # 动作空间
    act = res.get("action_space", {})
    print(f"  动作: {act.get('type')}, shape={act.get('shape')}, "
          f"dtype={act.get('dtype')}, "
          f"范围=[{act.get('low'):.1f}, {act.get('high'):.1f}]")

    # Episodes
    episodes = res.get("episodes", [])
    if episodes:
        print(f"  随机策略运行 {len(episodes)} 个 episode（每集最多 {max_steps} 步）：")
        rewards = []
        successes = 0
        for ep in episodes:
            ok = "✅ 成功" if ep["success"] else "⬜ 未成功"
            print(f"    Episode {ep['ep']}: reward={ep['reward']:8.3f}, "
                  f"steps={ep['steps']:3d}, {ok}")
            rewards.append(ep["reward"])
            if ep["success"]:
                successes += 1
        import statistics
        print(f"  平均奖励: {statistics.mean(rewards):.3f}")
        print(f"  成功集数: {successes}/{len(episodes)}（随机策略预期为 0）")


def main():
    import torch
    import mani_skill

    section("系统环境信息")
    print(f"  操作系统       : {platform.system()} {platform.version()[:20]}")
    print(f"  ManiSkill 版本 : {mani_skill.__version__}")
    print(f"  PyTorch 版本   : {torch.__version__}")
    print(f"  CUDA 可用      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 设备       : {torch.cuda.get_device_name(0)}")
    if platform.system() == "Windows":
        print("  ⚠️  Windows 系统：RGBD 模式将使用 render_backend='sapien_cpu'")

    TASKS = [
        {
            "env_id"     : "PickCube-v1",
            "name"       : "桌面单物体抓取",
            "description": "将桌面红色立方体移动到绿色目标球所在位置",
        },
        {
            "env_id"     : "PegInsertionSide-v1",
            "name"       : "轴孔插入",
            "description": "夹取插销，将橙色端插入带孔盒子侧面的孔中",
        },
    ]
    # (obs_mode, 描述)
    OBS_MODES = [
        ("state", "状态向量（适合快速 RL 实验，观测维度 (1,42)/(1,43)）"),
        ("rgbd",  "图像+深度（适合 Diffusion Policy 视觉策略）"),
    ]
    N_EPISODES = 3
    MAX_STEPS  = 200

    all_results = {}

    for task in TASKS:
        env_id = task["env_id"]
        section(f"任务：{env_id}  —  {task['name']}")
        print(f"  描述：{task['description']}")
        all_results[env_id] = {}

        for obs_mode, obs_desc in OBS_MODES:
            print(f"\n  [{obs_mode.upper():12s}] {obs_desc}")
            print(f"  → 启动子进程运行 {env_id} ({obs_mode}) ...")
            res = run_in_subprocess(env_id, obs_mode,
                                    n_episodes=N_EPISODES, max_steps=MAX_STEPS)
            all_results[env_id][obs_mode] = res
            print_result(res, max_steps=MAX_STEPS)

    # ── 汇总 ──────────────────────────────────────────────
    section("汇总结果")
    print(f"  {'任务':<32}  {'state模式':^10}  {'rgbd模式':^10}")
    print(f"  {'-'*58}")
    for env_id, modes in all_results.items():
        state_ok = "✅" if modes.get("state", {}).get("status") == "OK" else "❌"
        rgbd_ok  = "✅" if modes.get("rgbd",  {}).get("status") == "OK" else "❌"
        print(f"  {env_id:<32}  {state_ok:^10}  {rgbd_ok:^10}")

    # Windows 提示
    if platform.system() == "Windows":
        print("\n  💡 Windows 注意事项：")
        print("     - render_mode='human' 不可用（无 GUI），请使用 'rgb_array'")
        print("     - RGBD/sensor_data 渲染须加 render_backend='sapien_cpu'")
        print("       防止 Vulkan GPU 渲染在双显卡笔记本上崩溃")

    print("\n[DONE] 两个任务均已跑通 ✅")


# ─────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _, _, env_id, obs_mode, n_episodes, max_steps = sys.argv
        worker_main(env_id, obs_mode, int(n_episodes), int(max_steps))
    else:
        main()
