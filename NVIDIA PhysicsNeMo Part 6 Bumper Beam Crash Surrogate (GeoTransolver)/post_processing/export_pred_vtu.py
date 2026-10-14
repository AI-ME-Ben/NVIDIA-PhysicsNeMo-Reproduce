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
Run a trained GeoTransolver checkpoint on every run in data_full/vtk/ and
export the predictions as per-timestep VTU files (one time series per run):

    data_full/compare_vtu/predict/<split>/runN/runN.pvd          <- open in ParaView
    data_full/compare_vtu/predict/<split>/runN/vtu/runN_000.vtu ... runN_050.vtu
    data_full/compare_vtu/metrics.csv                    per-run error (%), plus split averages
    data_full/compare_vtu/predict/checkpoint.txt                 which checkpoint made them

The layout, frame numbering, and timesteps match export_vtu.py (ground truth
in data_full/compare_vtu/openradioss/), so the two .pvd files of a run can be opened
side by side in ParaView. Frame 000 is the undeformed input geometry (the
model predicts frames 001-050).

metrics.csv columns: disp, ux, uy, uz, effective_plastic_strain, stress_vm
error in percent, each the relative L2 error over all nodes and all 50
predicted frames of the run:

    error_% = 100 * ||pred - true||_2 / ||true||_2

disp is the displacement vector (relative to t0, all 3 components together),
ux/uy/uz its components. effective_plastic_strain (OpenRadioss /ANIM/SHELL/EPSP)
and stress_vm (von Mises, /ANIM/SHELL/VONM) use the same names as the
dataset's dynamic_targets and the VTU fields (stress models only). The
last rows (run = "mean") are the average over the runs of each split
(train / val / test) and over all runs. Metrics are recomputed for every run,
also when its VTU files already exist and are skipped.

Fields per frame (all point data):
    displacement                    predicted displacement relative to t0 (vector)
    error_displacement              |predicted - true| position error (mm)
    stress_vm, effective_plastic_strain              predicted values (stress models only)
    error_stress_vm, error_effective_plastic_strain  predicted - true (stress models only)

The model type (position-only vs. position + stress/strain) is read from the
checkpoint's args.json, so the same script works for checkpoints_full/ and
checkpoints_stress/.

Any one-shot model (GeoTransolver, MeshGraphNet, DI-GNN) can be evaluated from its
training folder with --run_dir: the model class comes from the checkpoint, the
dataset type (point cloud / graph), features and stats from <run_dir>/.hydra and
<run_dir>/stats, the checkpoint is the latest epoch in <run_dir>/checkpoints.
Outputs then go to predict_<tag>/, metrics_<tag>.csv and summary_<tag>.json.

summary_<tag>.json (per split): displacement relative L2 error (mean / median over
the runs) and pooled R^2 of |u| over all nodes, for all 50 frames and for the final
frame only (t = 100 ms), plus the final-frame RMSE of the displacement vector (mm).

Usage:
    python export_pred_vtu.py                     # default: checkpoints_stress epoch 1500
    python export_pred_vtu.py --splits test --runs run91,run92 --overwrite
    python export_pred_vtu.py --ckpt ../outputs/checkpoints_full/GeoTransolverOneShot.0.200.mdlus \
        --out_dir ../data_full/compare_vtu/predict_full
    # any model, from its training folder
    python export_pred_vtu.py --run_dir ../outputs_mgn --tag mgn
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import tempfile
import zipfile

import numpy as np
import pyvista as pv
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local physicsnemo source (see BUMPER_BEAM_REPRODUCTION.md 1.4)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, HERE)

import datapipe  # noqa: E402
import rollout  # noqa: E402
from datapipe import CrashPointCloudDataset  # noqa: E402
from export_vtu import run_paths, write_pvd  # noqa: E402
from vtp_reader import Reader, find_run_folders  # noqa: E402

SPLITS = ("train", "val", "test")
DYNAMIC_TARGETS = ["effective_plastic_strain", "stress_vm"]
GLOBAL_FEATURES = ["velocity_x", "thickness_scale", "rwall_origin_y"]
DT = 0.002  # /ANIM/DT TFREQ = 2 ms
# metrics.csv error columns and the target channel they come from
ERROR_COLUMNS = {
    "disp_error_pct": "disp",
    "ux_error_pct": "ux",
    "uy_error_pct": "uy",
    "uz_error_pct": "uz",
    "effective_plastic_strain_error_pct": "effective_plastic_strain",
    "stress_vm_error_pct": "stress_vm",
}


def rel_l2_pct(pred: np.ndarray, true: np.ndarray) -> float:
    """100 * ||pred - true||_2 / ||true||_2 over all entries."""
    return float(100.0 * np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def latest_checkpoint(run_dir: str) -> str:
    """Latest epoch .mdlus in <run_dir>/checkpoints."""
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "*.mdlus"))
    if not ckpts:
        raise FileNotFoundError(f"no .mdlus checkpoint in {run_dir}/checkpoints")
    return max(ckpts, key=lambda p: int(re.search(r"\.(\d+)\.mdlus$", p).group(1)))


def load_model(ckpt: str, device: str):
    """Rebuild the model class named in the .mdlus args.json and load its weights."""
    with zipfile.ZipFile(ckpt) as z:
        meta = json.loads(z.read("args.json"))
    kwargs = {k: v for k, v in meta["__args__"].items() if k != "args"}
    model = getattr(rollout, meta["__name__"])(**kwargs)
    model.load(ckpt)
    return model.to(device).eval(), kwargs


@torch.no_grad()
def export_split(model, split, data_dir, args, out_root, dynamic_targets, device, rows, acc=None):
    vtp_paths = find_run_folders(os.path.join(data_dir, split))
    if args.runs:
        wanted = set(args.runs.split(","))
        vtp_paths = [p for p in vtp_paths if os.path.splitext(os.path.basename(p))[0] in wanted]
    if not vtp_paths:
        return
    run_names = [os.path.splitext(os.path.basename(p))[0] for p in vtp_paths]

    # split="validation": loads the training stats from stats_dir instead of recomputing
    # (and overwriting) them, and stops the reader writing per-frame VTPs.
    # The reader still creates an empty ./output_<run> folder per run, so build the
    # dataset inside a temporary cwd.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="export_pred_vtu_") as tmp:
        os.chdir(tmp)
        try:
            ds = args.dataset_cls(
                name=f"predict_{split}",
                reader=Reader(),
                data_dir=os.path.join(data_dir, split),
                global_features_filepath=args.global_features,
                global_features=GLOBAL_FEATURES,
                split="validation",
                num_samples=len(find_run_folders(os.path.join(data_dir, split))),
                num_steps=args.num_steps,
                static_features=[],
                dynamic_targets=dynamic_targets,
                stats_dir=args.stats_dir,
                sample_type="all_time_steps",
            )
        finally:
            os.chdir(cwd)
    data_stats = dict(
        node={k: v.to(device) for k, v in ds.node_stats.items()},
        edge={k: v.to(device) for k, v in getattr(ds, "edge_stats", {}).items()},
        feature={k: v.to(device) for k, v in ds.feature_stats.items()},
    )
    pos_mean = data_stats["node"]["pos_mean"].view(1, 1, -1)
    pos_std = data_stats["node"]["pos_std"].view(1, 1, -1)
    all_names = [os.path.splitext(os.path.basename(p))[0]
                 for p in find_run_folders(os.path.join(data_dir, split))]

    for run, path in zip(run_names, vtp_paths):
        sample = ds[all_names.index(run)].to(device)
        pred = model(sample=sample, data_stats=data_stats)  # [N, T, Fo]
        true = sample.node_target

        # Positions back to mm, then displacement relative to t0
        coords0 = (sample.node_features["coords"] * pos_std[0] + pos_mean[0]).cpu().numpy()  # [N,3]
        pred_pos = (pred[:, :, :3] * pos_std + pos_mean).cpu().numpy()  # [N,T,3]
        true_pos = (true[:, :, :3] * pos_std + pos_mean).cpu().numpy()
        pos_err = np.linalg.norm(pred_pos - true_pos, axis=-1)  # [N,T]
        extra_pred = {k: pred[:, :, 3 + i].cpu().numpy() for i, k in enumerate(dynamic_targets)}
        extra_true = {k: true[:, :, 3 + i].cpu().numpy() for i, k in enumerate(dynamic_targets)}

        # Relative L2 error (%) per channel over all nodes and predicted frames
        pred_disp = pred_pos - coords0[:, None]
        true_disp = true_pos - coords0[:, None]
        row = {"run": run, "split": split}
        for col, channel in ERROR_COLUMNS.items():
            if channel == "disp":
                row[col] = rel_l2_pct(pred_disp, true_disp)
            elif channel in ("ux", "uy", "uz"):
                c = "xyz".index(channel[1])
                row[col] = rel_l2_pct(pred_disp[..., c], true_disp[..., c])
            elif channel in dynamic_targets:
                row[col] = rel_l2_pct(extra_pred[channel], extra_true[channel])
        rows.append(row)
        if acc is not None:
            a = acc.setdefault(split, {"rel": [], "rel_final": [], "r2": np.zeros(4), "r2_final": np.zeros(4),
                                       "sse_final": 0.0, "n_final": 0})
            a["rel"].append(row["disp_error_pct"])
            a["rel_final"].append(rel_l2_pct(pred_disp[:, -1], true_disp[:, -1]))
            for key, sl in (("r2", slice(None)), ("r2_final", slice(-1, None))):
                pm = np.linalg.norm(pred_disp[:, sl], axis=-1)
                tm = np.linalg.norm(true_disp[:, sl], axis=-1)
                a[key] += [tm.size, tm.sum(), np.square(tm).sum(), np.square(pm - tm).sum()]
            a["sse_final"] += float(np.square(pred_disp[:, -1] - true_disp[:, -1]).sum())
            a["n_final"] += pred_disp.shape[0]
        errors = ", ".join(f"{c.replace('_error_pct', '')} {v:.1f}%" for c, v in row.items() if c.endswith("_pct"))

        if args.no_vtu:
            print(f"{run} ({split}): {errors}")
            continue
        pvd_path, vtu_dir = run_paths(out_root, split, run)
        if os.path.exists(pvd_path) and not args.overwrite:
            print(f"{run} ({split}): VTU already exists, skipping export | {errors}")
            continue
        os.makedirs(vtu_dir, exist_ok=True)

        base = pv.read(path)
        base = pv.PolyData(base.points.copy(), faces=base.faces).cast_to_unstructured_grid()
        T = pred_pos.shape[1]
        entries = []
        for i in range(T + 1):
            grid = base.copy()
            if i == 0:  # undeformed input geometry
                grid.points = coords0
                grid.point_data["displacement"] = np.zeros_like(coords0)
                grid.point_data["error_displacement"] = np.zeros(len(coords0))
                for k in dynamic_targets:
                    grid.point_data[k] = np.zeros(len(coords0))
                    grid.point_data[f"error_{k}"] = np.zeros(len(coords0))
            else:
                t = i - 1
                grid.points = pred_pos[:, t]
                grid.point_data["displacement"] = pred_pos[:, t] - coords0
                grid.point_data["error_displacement"] = pos_err[:, t]
                for k in dynamic_targets:
                    grid.point_data[k] = extra_pred[k][:, t]
                    grid.point_data[f"error_{k}"] = extra_pred[k][:, t] - extra_true[k][:, t]
            time = round(i * DT, 6)
            grid.field_data["TimeValue"] = [time]
            fname = f"{run}_{i:03d}.vtu"
            grid.save(os.path.join(vtu_dir, fname), binary=True)
            entries.append((time, f"vtu/{fname}"))
        write_pvd(pvd_path, entries)
        print(f"{run} ({split}): wrote {len(entries)} frames | {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        default=os.path.join(PROJECT_DIR, "outputs", "checkpoints_stress", "GeoTransolverOneShot.0.1500.mdlus"),
        help="Path to a GeoTransolverOneShot .mdlus file.",
    )
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_DIR, "data_full", "vtk"),
                        help="Folder containing train/val/test runN.vtp files.")
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_DIR, "data_full", "compare_vtu", "predict"))
    parser.add_argument("--metrics", default=os.path.join(PROJECT_DIR, "data_full", "compare_vtu", "metrics.csv"),
                        help="Per-run error table (CSV).")
    parser.add_argument("--stats_dir", default=os.path.join(PROJECT_DIR, "outputs", "stats"),
                        help="Normalization stats written by train.py.")
    parser.add_argument("--global_features",
                        default=os.path.join(PROJECT_DIR, "dataset_generation", "global_features.json"))
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--runs", default=None, help="Comma-separated run names, e.g. run1,run91.")
    parser.add_argument("--num_steps", type=int, default=51)
    parser.add_argument("--overwrite", action="store_true", help="Re-export runs that already exist.")
    parser.add_argument("--run_dir", default=None,
                        help="Training folder (e.g. ../outputs_mgn): checkpoint, stats, features and dataset "
                             "type are taken from it. Overrides --ckpt/--stats_dir.")
    parser.add_argument("--tag", default=None,
                        help="With --run_dir: output name -> predict_<tag>/, metrics_<tag>.csv, summary_<tag>.json")
    parser.add_argument("--no_vtu", action="store_true", help="Only compute metrics, write no VTU files.")
    args = parser.parse_args()
    args.dataset_cls = CrashPointCloudDataset
    dynamic_targets = None
    if args.run_dir:
        from omegaconf import OmegaConf

        run_dir = os.path.abspath(args.run_dir)
        cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
        args.dataset_cls = getattr(datapipe, cfg.datapipe._target_.split(".")[-1])
        dynamic_targets = list(cfg.datapipe.dynamic_targets)
        args.ckpt = latest_checkpoint(run_dir) if args.ckpt == parser.get_default("ckpt") else args.ckpt
        args.stats_dir = os.path.join(run_dir, "stats")
        tag = args.tag or os.path.basename(run_dir).replace("outputs_", "")
        compare = os.path.join(PROJECT_DIR, "data_full", "compare_vtu")
        if args.out_dir == parser.get_default("out_dir"):
            args.out_dir = os.path.join(compare, f"predict_{tag}")
        if args.metrics == parser.get_default("metrics"):
            args.metrics = os.path.join(compare, f"metrics_{tag}.csv")
    for k in ("ckpt", "data_dir", "out_dir", "metrics", "stats_dir", "global_features"):
        setattr(args, k, os.path.abspath(getattr(args, k)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, kwargs = load_model(args.ckpt, device)
    if dynamic_targets is None:
        fo = kwargs["out_dim"] // (args.num_steps - 1)
        dynamic_targets = DYNAMIC_TARGETS if fo == 3 + len(DYNAMIC_TARGETS) else []

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Checkpoint : {args.ckpt}  ({model.__class__.__name__}, dynamic_targets={dynamic_targets})")
    print(f"Output     : {out_dir}")
    with open(os.path.join(out_dir, "checkpoint.txt"), "w") as f:
        f.write(os.path.relpath(args.ckpt, PROJECT_DIR).replace(os.sep, "/") + "\n")

    metrics_path = args.metrics
    err_cols = [c for c, ch in ERROR_COLUMNS.items() if ch in ("disp", "ux", "uy", "uz") or ch in dynamic_targets]
    fields = ["run", "split"] + err_cols
    # Keep per-run rows (same columns) of runs that are not processed this time (--runs/--splits)
    old_rows = []
    if os.path.exists(metrics_path):
        with open(metrics_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == fields:
                old_rows = [r for r in reader if r["run"] != "mean"]
    new_rows, acc = [], {}
    for split in args.splits.split(","):
        export_split(model, split, args.data_dir, args, out_dir, dynamic_targets, device, new_rows, acc)

    done = {r["run"] for r in new_rows}
    rows = [r for r in old_rows if r["run"] not in done] + new_rows
    rows.sort(key=lambda r: int(r["run"][3:]))

    # Split averages (mean of the per-run errors)
    means = []
    for split in list(SPLITS) + ["all"]:
        group = [r for r in rows if split == "all" or r["split"] == split]
        if group:
            means.append({"run": "mean", "split": split,
                          **{c: float(np.mean([float(r[c]) for r in group])) for c in err_cols}})

    def fmt(r):
        return {k: (f"{float(v):.3f}" if k in err_cols else v) for k, v in r.items()}

    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(fmt(r) for r in rows + means)

    # summary of this call: all frames vs final frame only
    summary = {"checkpoint": os.path.relpath(args.ckpt, PROJECT_DIR).replace(os.sep, "/"), "splits": {}}
    for split, a in acc.items():
        r2 = lambda v: float(1.0 - v[3] / (v[2] - v[1] ** 2 / v[0]))  # noqa: E731
        summary["splits"][split] = {
            "runs": len(a["rel"]),
            "disp_rel_l2_pct": {"mean": float(np.mean(a["rel"])), "median": float(np.median(a["rel"]))},
            "disp_rel_l2_pct_final": {"mean": float(np.mean(a["rel_final"])), "median": float(np.median(a["rel_final"]))},
            "disp_r2": r2(a["r2"]),
            "disp_r2_final": r2(a["r2_final"]),
            "disp_rmse_mm_final": float(np.sqrt(a["sse_final"] / a["n_final"])),
        }
    summary_path = os.path.join(os.path.dirname(metrics_path),
                                os.path.basename(metrics_path).replace("metrics", "summary").replace(".csv", ".json"))
    json.dump(summary, open(summary_path, "w"), indent=1)
    print(f"Summary: {summary_path}")
    for split, e in summary["splits"].items():
        print(f"  {split:<6} all frames: rel L2 {e['disp_rel_l2_pct']['mean']:.2f}% R2 {e['disp_r2']:.4f} | "
              f"final: rel L2 {e['disp_rel_l2_pct_final']['mean']:.2f}% R2 {e['disp_r2_final']:.4f} "
              f"RMSE {e['disp_rmse_mm_final']:.2f} mm")

    print(f"\nDone: {len(new_rows)} runs processed. Metrics: {metrics_path}")
    print("Average relative L2 error (%):")
    names = [c.replace("_error_pct", "") for c in err_cols]
    print(f"  {'split':<6}" + "".join(f"{n:>{max(9, len(n) + 2)}}" for n in names))
    for m in means:
        print(f"  {m['split']:<6}" + "".join(f"{m[c]:>{max(9, len(n) + 2)}.2f}" for c, n in zip(err_cols, names)))


if __name__ == "__main__":
    main()
