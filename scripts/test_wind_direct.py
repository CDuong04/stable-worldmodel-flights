"""Direct env test for wind disturbance (no swm.World wrapper chain).

Steps the raw RocketLandingExtEnv with the PDG expert, counts landings.
Bypasses the wrapper assertion that was tripping world.evaluate().
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium
import numpy as np

import stable_worldmodel.envs.rocket_landing_ext  # noqa: F401
from stable_worldmodel.envs.pdg_expert import PDGExpertPolicy


def parse_level(level: str) -> dict:
    """Map a compound level tag to reset options.
    Formats:
      "none"                              -> {}
      "wind:<calm|light|gust|storm>"      -> {"wind_level": ...}
      "kick:<light|moderate|severe>"      -> {"lateral_kick_level": ...}
      "movpad"                            -> {"moving_pad": True}
      "combo:<wind_x>+<kick_y>[+movpad]"  -> combined
    """
    if level == "none":
        return {}
    opts: dict = {}
    for tok in level.split("+"):
        if tok.startswith("wind:"):
            opts["wind_level"] = tok.split(":", 1)[1]
        elif tok.startswith("kick:"):
            opts["lateral_kick_level"] = tok.split(":", 1)[1]
        elif tok == "movpad" or tok == "moving_pad":
            opts["moving_pad"] = True
            opts["pad_sigma_xy"] = 1.0
            opts["pad_theta"] = 0.5
        elif tok.startswith("combo:"):
            # combo:wind=gust+kick=moderate+movpad
            for sub in tok.split(":", 1)[1].split("+"):
                if sub.startswith("wind="):
                    opts["wind_level"] = sub.split("=", 1)[1]
                elif sub.startswith("kick="):
                    opts["lateral_kick_level"] = sub.split("=", 1)[1]
                elif sub == "movpad":
                    opts["moving_pad"] = True
    return opts


def run_episode(env, policy, max_steps=1200):
    obs, info = env.reset()
    policy.reset()
    total_reward = 0.0
    for t in range(max_steps):
        action = policy.predict(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            return {"success": bool(terminated), "T": t + 1, "reward": total_reward}
    return {"success": False, "T": max_steps, "reward": total_reward}


def sweep(episodes, seed, levels, max_steps):
    results = {}
    for level in levels:
        env = gymnasium.make(
            "swm/PFRocketLandingExt-v0",
            render_mode="rgb_array",
            max_episode_steps=max_steps,
        )
        policy = PDGExpertPolicy()
        succ, land, fatal, T_list, r_list = 0, 0, 0, [], []
        t0 = time.time()
        for ep in range(episodes):
            reset_opts = parse_level(level)
            obs, info = env.reset(seed=seed + ep,
                                   options=reset_opts if reset_opts else None)
            policy.reset()
            total_reward = 0.0
            done_info = None
            for t in range(max_steps):
                action = policy.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    # env's own flags: truncated+env_complete = success, terminated+fatal_collision = crash
                    is_landing = bool(truncated) and bool(info.get("env_complete", False))
                    is_crash = bool(terminated) and bool(info.get("fatal_collision", False))
                    done_info = {
                        "T": t + 1, "reward": total_reward,
                        "is_landing": is_landing, "is_crash": is_crash,
                        "terminated": bool(terminated), "truncated": bool(truncated),
                    }
                    break
            if done_info is None:
                done_info = {"T": max_steps, "reward": total_reward,
                              "is_landing": False, "is_crash": False,
                              "terminated": False, "truncated": False}
            succ += int(done_info["is_landing"])
            fatal += int(done_info["is_crash"])
            land += int(done_info["reward"] > 100.0)  # soft reward-based
            T_list.append(done_info["T"])
            r_list.append(done_info["reward"])
        elapsed = time.time() - t0
        env.close()
        landing = succ / episodes       # env_complete: proper landing
        crash   = fatal / episodes      # fatal_collision
        softR   = land / episodes       # reward > 100
        other   = 1.0 - landing - crash
        print(f"  [{level:>42}]  land={landing*100:5.1f}%  crash={crash*100:5.1f}%  "
              f"timeout={other*100:5.1f}%  R={np.mean(r_list):+7.1f}±{np.std(r_list):5.1f}  "
              f"({episodes} eps, {elapsed:.0f}s)")
        results[level] = {
            "landing_rate": float(landing),
            "crash_rate":   float(crash),
            "timeout_rate": float(other),
            "reward_success_rate": float(softR),
            "mean_steps": float(np.mean(T_list)),
            "mean_reward": float(np.mean(r_list)),
            "n": int(episodes),
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--levels", nargs="+",
                   default=["none", "calm", "light", "gust", "storm"])
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--out", default="results/wind_expert_direct.json")
    args = p.parse_args()

    print(f"PDG expert under wind disturbance ({args.episodes} eps each)")
    print("-" * 68)
    results = sweep(args.episodes, args.seed, args.levels, args.max_steps)
    print("-" * 68)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
