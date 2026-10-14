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
Compare the position-only one-shot models (GeoTransolver, MeshGraphNet, DI-GNN)
on the test set, from the outputs of export_pred_vtu.py --run_dir:

    figures/model_comparison_error.png      relative L2 displacement error (%), all frames vs final frame
    figures/model_comparison_r2.png         pooled R^2 of |u|, all frames vs final frame
    figures/model_comparison_r2_final.png   per-node true vs predicted |u| at the final frame, one panel per model

Reads data_full/compare_vtu/summary_<tag>.json and predict_<tag>/test/.

Usage:
    python plot_model_comparison.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from plot_r2_scatter import collect, r2_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE_DIR = os.path.join(HERE, "..", "data_full", "compare_vtu")
FIG_DIR = os.path.join(HERE, "..", "figures")
MODELS = [("geotransolver", "GeoTransolver"), ("mgn", "MeshGraphNet"), ("dignn", "DI-GNN (re-impl.)")]
ALL, FINAL = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def bars(ax, labels, series, fmt, ylabel):
    x = np.arange(len(labels))
    w = 0.36
    for j, (name, vals, color) in enumerate(series):
        xs = x + (j - 0.5) * (w + 0.02)
        ax.bar(xs, vals, w, color=color, label=name, zorder=3)
        for xv, v in zip(xs, vals):
            ax.text(xv, v, fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", ncol=2, bbox_to_anchor=(0, -0.1))


def main() -> None:
    models = [(t, n) for t, n in MODELS if os.path.exists(os.path.join(COMPARE_DIR, f"summary_{t}.json"))]
    test = {t: json.load(open(os.path.join(COMPARE_DIR, f"summary_{t}.json")))["splits"]["test"] for t, _ in models}
    labels = [n for _, n in models]

    for key, title, ylabel, series, fmt in [
        ("error", "Displacement error, test set (lower is better)", "Relative L2 error (%)",
         [("all 50 frames", [test[t]["disp_rel_l2_pct"]["mean"] for t, _ in models], ALL),
          ("final frame (t = 100 ms)", [test[t]["disp_rel_l2_pct_final"]["mean"] for t, _ in models], FINAL)], "{:.1f}"),
        ("r2", "Displacement R², test set (higher is better)", "R² of |u|, all nodes",
         [("all 50 frames", [test[t]["disp_r2"] for t, _ in models], ALL),
          ("final frame (t = 100 ms)", [test[t]["disp_r2_final"] for t, _ in models], FINAL)], "{:.3f}"),
    ]:
        # Calibri for the bar charts only: it has no glyphs for the log-scale colorbar ticks (10^n) below
        plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "axes.edgecolor": GRID,
                             "xtick.color": INK_2, "ytick.color": INK_2})
        fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        bars(ax, labels, series, fmt, ylabel)
        top = max(max(v) for _, v, _ in series)
        ax.set_ylim(0, (1.0 if key == "r2" else top) * 1.12)
        ax.set_title(title, fontsize=14, color=INK, loc="left", pad=26)
        ax.text(0, 1.03, "Position-only one-shot models, bumper beam, 9 test runs; error = mean over runs",
                transform=ax.transAxes, fontsize=10, color=INK_2)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"model_comparison_{key}.png")
        fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # final-frame scatter, one panel per model, shared axes
    plt.rcdefaults()
    matplotlib.use("Agg")
    data = {}
    for t, _ in models:
        d, runs = collect(COMPARE_DIR, os.path.join(COMPARE_DIR, f"predict_{t}"), "test", ["disp"], last_frame=True)
        data[t] = d["disp"]
    hi = max(max(p.max(), y.max()) for p, y in data.values())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.0), squeeze=False)
    for ax, (t, name) in zip(axes[0], models):
        pred, true = data[t]
        h = ax.hist2d(true, pred, bins=250, range=[[0, hi], [0, hi]], norm=LogNorm(), cmap="viridis")
        ax.plot([0, hi], [0, hi], "r--", lw=1)
        ax.set_aspect("equal")
        ax.set_xlabel("True |u| (mm)")
        ax.set_ylabel("Predicted |u| (mm)")
        ax.set_title(f"{name}\nR$^2$ = {r2_score(pred, true):.4f}, RMSE = {test[t]['disp_rmse_mm_final']:.1f} mm")
        fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Test set, final frame (t = 100 ms): per-node true vs predicted displacement "
                 f"({len(runs)} runs x all nodes)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "model_comparison_r2_final.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
