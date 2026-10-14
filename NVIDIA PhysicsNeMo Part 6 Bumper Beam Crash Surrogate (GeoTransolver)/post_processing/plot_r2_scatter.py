# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Per-node True vs Predicted plot with R^2 for one split (default: test), from
the VTU files in data_full/compare_vtu/:

    displacement magnitude |u| (mm)   predict vs openradioss, relative to t0
    effective plastic strain (eps)    predict vs predict - error (node values)
    von Mises stress                  predict vs predict - error (node values)

Every node of every predicted frame (001-050) of every run in the split is
used; R^2 = 1 - SS_res / SS_tot over all of them. The plot is a 2D histogram
(log color scale), since there are millions of points per field.

With --last_frame only the final frame (050, t = 100 ms) is used: the state
that final-state surrogates (e.g. DI-GNN, PAMM 2026, doi:10.1002/pamm.70233)
are evaluated on. Early frames have small deformations, so all-frame R^2 is
not comparable to final-state R^2.

Displacement is relative to t0, not the absolute node position: the beam is
2 m long, so R^2 on absolute positions is ~0.9999 even for a model that
predicts no deformation at all.

Usage:
    python plot_r2_scatter.py                       # test split
    python plot_r2_scatter.py --split val --output ../data_full/compare_vtu/r2_scatter_val.png
    # position-only model: predictions exported elsewhere, displacement only
    python plot_r2_scatter.py --predict_dir <dir from export_pred_vtu.py --out_dir> --fields disp         --output ../data_full/compare_vtu/r2_scatter_per_node_disponly.png
    # final state only
    python plot_r2_scatter.py --last_frame --output ../data_full/compare_vtu/r2_scatter_per_node_last.png
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE_DIR = os.path.join(HERE, "..", "data_full", "compare_vtu")
FIELDS = {
    "disp": "Displacement |u| (mm)",
    "effective_plastic_strain": "Effective plastic strain (eps)",
    "stress_vm": "von Mises stress",
}


def collect(compare_dir: str, predict_dir: str, split: str, fields: list[str], last_frame: bool = False):
    """Return ({field: (pred, true)} over all nodes/frames/runs, run names)."""
    runs = sorted(os.listdir(os.path.join(predict_dir, split)), key=lambda r: int(r[3:]))
    pred = {k: [] for k in fields}
    true = {k: [] for k in fields}
    for run in runs:
        frames = sorted(glob.glob(os.path.join(predict_dir, split, run, "vtu", f"{run}_*.vtu")))
        frames = frames[-1:] if last_frame else frames[1:]  # frame 000 is the undeformed input, not a prediction
        for f in frames:
            p = pv.read(f)
            g = pv.read(os.path.join(compare_dir, "openradioss", split, run, "vtu", os.path.basename(f)))
            if "disp" in fields:
                pred["disp"].append(np.linalg.norm(p["displacement"], axis=1))
                true["disp"].append(np.linalg.norm(g["displacement"], axis=1))
            for k in set(fields) & {"effective_plastic_strain", "stress_vm"}:
                # node-level truth the model was trained on (cell data averaged to nodes)
                pred[k].append(p[k])
                true[k].append(p[k] - p[f"error_{k}"])
    return {k: (np.concatenate(pred[k]), np.concatenate(true[k])) for k in fields}, runs


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    return float(1.0 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare_dir", default=COMPARE_DIR)
    parser.add_argument("--predict_dir", default=None, help="Default: <compare_dir>/predict")
    parser.add_argument("--split", default="test")
    parser.add_argument("--fields", default=",".join(FIELDS), help="Subset of: " + ", ".join(FIELDS))
    parser.add_argument("--last_frame", action="store_true", help="Use only the final frame (050)")
    parser.add_argument("--output", default=None, help="Default: <compare_dir>/r2_scatter_per_node.png")
    args = parser.parse_args()
    compare_dir = os.path.abspath(args.compare_dir)
    output = args.output or os.path.join(compare_dir, "r2_scatter_per_node.png")

    predict_dir = os.path.abspath(args.predict_dir or os.path.join(compare_dir, "predict"))
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    data, runs = collect(compare_dir, predict_dir, args.split, fields, args.last_frame)
    ckpt_file = os.path.join(predict_dir, "checkpoint.txt")
    ckpt = open(ckpt_file).read().strip() if os.path.exists(ckpt_file) else "unknown checkpoint"
    n = len(next(iter(data.values()))[0])

    fig, axes = plt.subplots(1, len(fields), figsize=(5.7 * len(fields) + 0.5, 5.6), squeeze=False)
    for ax, k in zip(axes[0], fields):
        label = FIELDS[k]
        pred, true = data[k]
        r2 = r2_score(pred, true)
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        h = ax.hist2d(true, pred, bins=300, range=[[lo, hi], [lo, hi]], norm=LogNorm(), cmap="viridis")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"True {label}"); ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}\nR$^2$ = {r2:.4f}")
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04, label="points per bin")
        print(f"{k:<26} R2 = {r2:.4f}")
    sep = "\n" if len(fields) == 1 else " "  # narrow single-panel figure: wrap the title
    fig.suptitle(
        f"{args.split.capitalize()} set: per-node true vs predicted{sep}({len(runs)} runs x {'final step (t = 100 ms)' if args.last_frame else '50 steps'} x all nodes, "
        f"{n:,} points per field)\nGeoTransolver, {ckpt}",
        fontsize=12 if len(fields) > 1 else 10,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
