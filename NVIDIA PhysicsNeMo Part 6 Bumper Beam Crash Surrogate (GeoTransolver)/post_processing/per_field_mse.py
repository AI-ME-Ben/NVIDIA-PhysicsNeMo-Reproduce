"""Compute separate MSE for position / effective_plastic_strain / stress_vm
from an inference run's exact_*.vtp files (which carry exact_/pred_/diff_
per-field point data when dynamic_targets is non-empty).

Usage:
    python per_field_mse.py --exact_dir <dir with frame_*_exact.vtp>
"""
import argparse
import glob
import os
import re

import numpy as np
import pyvista as pv


def sorted_frames(directory):
    files = glob.glob(os.path.join(directory, "*_exact.vtp"))

    def idx(p):
        return int(re.search(r"(\d+)_exact\.vtp$", os.path.basename(p)).group(1))

    return sorted(files, key=idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_dir", required=True)
    args = parser.parse_args()

    frames = sorted_frames(args.exact_dir)
    if not frames:
        raise FileNotFoundError(f"no *_exact.vtp under {args.exact_dir}")

    pos_sq = []
    field_sq = {}
    for f in frames:
        m = pv.read(f)
        pd = m.point_data
        diff = np.asarray(pd["difference"])  # [N,3] pred-exact position error
        pos_sq.append(np.mean(diff**2))
        for key in pd.keys():
            if key.startswith("diff_"):
                name = key[len("diff_"):]
                field_sq.setdefault(name, []).append(np.mean(np.asarray(pd[key]) ** 2))

    print(f"frames: {len(frames)}")
    print(f"position MSE (mean over all timesteps): {np.mean(pos_sq):.6f}")
    for name, vals in field_sq.items():
        print(f"{name} MSE (mean over all timesteps): {np.mean(vals):.6f}  "
              f"(last-frame: {vals[-1]:.6f})")


if __name__ == "__main__":
    main()
