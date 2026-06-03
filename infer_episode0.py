"""Run inference with the trained x1_pingpong_lora checkpoint on episode 0
and produce a trajectory comparison plot (predicted vs. ground-truth).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import vlash.policies.pi05  # noqa: F401  (registers PI05Config as a PreTrainedConfig subclass)
from vlash.policies.factory import make_policy

CHECKPOINT = Path(
    "/mnt/workspace/outputs/train/x1_pingpong_lora/checkpoints/014000/pretrained_model"
)
DATA_ROOT = "/mnt/workspace/data/20260226"
REPO_ID = "local/x1_pingpong"
N_ACTION_STEPS = int(os.environ.get("N_ACTION_STEPS", "32"))
EPISODE = int(os.environ.get("EPISODE", "0"))
OUTPUT_PNG = Path(f"/mnt/workspace/outputs/episode{EPISODE}_traj_compare_n{N_ACTION_STEPS}.png")
OUTPUT_NPZ = Path(f"/mnt/workspace/outputs/episode{EPISODE}_traj_compare_n{N_ACTION_STEPS}.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_policy(ds_meta: LeRobotDatasetMetadata):
    """Bypass PreTrainedConfig.from_pretrained (which resolves to lerobot's PI05Config)
    and construct vlash's PI05Config directly from config.json."""
    import dataclasses
    import json

    from lerobot.configs.types import FeatureType, PolicyFeature

    from vlash.policies.pi05.configuration_pi05 import PI05Config

    # Override the lerobot pi05 registration so cfg.type returns 'pi05' for vlash's class
    # and the factory dispatches to vlash's PI05Policy.
    from lerobot.configs.policies import PreTrainedConfig as _PreTrainedConfig

    _PreTrainedConfig._choice_registry["pi05"] = PI05Config

    with open(CHECKPOINT / "config.json") as f:
        raw = json.load(f)
    raw.pop("type", None)
    raw.pop("vlm_config", None)
    raw.pop("action_expert_config", None)
    # Reconstruct features (PolicyFeature dataclass)
    def to_feats(d):
        return {
            k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
            for k, v in d.items()
        }

    raw["input_features"] = to_feats(raw["input_features"])
    raw["output_features"] = to_feats(raw["output_features"])

    if "normalization_mapping" in raw:
        from lerobot.configs.types import NormalizationMode
        raw["normalization_mapping"] = {
            k: NormalizationMode[v] for k, v in raw["normalization_mapping"].items()
        }

    field_names = {f.name for f in dataclasses.fields(PI05Config)}
    raw = {k: v for k, v in raw.items() if k in field_names}

    cfg = PI05Config(**raw)
    cfg.pretrained_path = str(CHECKPOINT)
    cfg.device = DEVICE
    cfg.dtype = "bfloat16"
    cfg.compile_model = False
    cfg.fuse_qkv = True
    cfg.fuse_gate_up = True

    policy = make_policy(cfg, ds_meta)
    policy.eval()
    return policy


def get_episode_frames(ds: LeRobotDataset, ep_idx: int) -> list[int]:
    """Return global indices that belong to a given episode."""
    eidx = ds.hf_dataset["episode_index"]
    if hasattr(eidx, "numpy"):
        eidx = eidx.numpy()
    eidx = np.asarray(eidx)
    return np.where(eidx == ep_idx)[0].tolist()


def collate_sample(sample: dict, device: str) -> dict:
    batch: dict = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif k == "task":
            batch["task"] = [v]
    return batch


@torch.no_grad()
def main():
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[device] {DEVICE}")
    print(f"[checkpoint] {CHECKPOINT}")

    ds_meta = LeRobotDatasetMetadata(REPO_ID, root=DATA_ROOT)
    print(f"[dataset] episodes={ds_meta.total_episodes}, frames={ds_meta.total_frames}")

    ds = LeRobotDataset(REPO_ID, root=DATA_ROOT)
    frames = get_episode_frames(ds, ep_idx=EPISODE)
    print(f"[episode {EPISODE}] {len(frames)} frames (indices {frames[0]}..{frames[-1]})")

    policy = build_policy(ds_meta)

    # Collect ground truth
    gt = np.stack([ds[i]["action"].numpy() for i in frames])  # [T, 7]
    state_traj = np.stack([ds[i]["observation.state"].numpy() for i in frames])

    # Run sequential inference: re-plan every n_action_steps frames (real-robot style)
    n_action_steps = N_ACTION_STEPS
    chunk_size = policy.config.chunk_size
    print(f"[policy] chunk_size={chunk_size}, n_action_steps={n_action_steps}")

    pred = np.full_like(gt, np.nan)
    t = 0
    while t < len(frames):
        sample = ds[frames[t]]
        batch = collate_sample(sample, DEVICE)
        action_chunk = policy.predict_action_chunk(batch)  # [1, T, 7]
        action_chunk = action_chunk[0].float().cpu().numpy()  # [T, 7]
        end = min(t + n_action_steps, len(frames))
        pred[t:end] = action_chunk[: end - t]
        print(f"  step {t:3d}/{len(frames)} -> chunk filled [{t}, {end})")
        t = end

    # Save raw arrays for reuse
    np.savez(OUTPUT_NPZ, gt=gt, pred=pred, state=state_traj)
    print(f"[saved] {OUTPUT_NPZ}")

    # Plot 7 joints
    joint_names = [f"right_joint_{i+1}.pos" for i in range(7)]
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    x = np.arange(len(frames))
    for i, ax in enumerate(axes):
        ax.plot(x, gt[:, i], label="ground truth", color="tab:blue", linewidth=1.6)
        ax.plot(x, pred[:, i], label="predicted", color="tab:red", linewidth=1.2, linestyle="--")
        ax.plot(x, state_traj[:, i], label="state", color="tab:gray", linewidth=0.8, alpha=0.5)
        mae = np.nanmean(np.abs(gt[:, i] - pred[:, i]))
        ax.set_ylabel(joint_names[i], fontsize=9)
        ax.set_title(f"{joint_names[i]}  |  MAE={mae:.4f}", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(f"frame index (episode {EPISODE})")
    overall_mae = np.nanmean(np.abs(gt - pred))
    fig.suptitle(
        f"Episode {EPISODE}  predicted vs ground-truth action  (overall MAE={overall_mae:.4f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUTPUT_PNG, dpi=130)
    print(f"[saved] {OUTPUT_PNG}")
    print(f"[overall MAE] {overall_mae:.4f}")


if __name__ == "__main__":
    sys.exit(main())
