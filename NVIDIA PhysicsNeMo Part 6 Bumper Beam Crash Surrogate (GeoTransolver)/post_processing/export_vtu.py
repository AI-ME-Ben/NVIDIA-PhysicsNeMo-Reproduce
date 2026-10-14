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
Export every OpenRadioss result in data_full/vtk/ to per-timestep VTU files.

Each data_full/vtk/{train,val,test}/runN.vtp (Curator output) stores the whole
trajectory in one file (displacement_t*, cell_stress_vm_t*,
cell_effective_plastic_strain_t*). This script splits it into one
UnstructuredGrid per timestep, with the mesh moved to its deformed position,
and writes a .pvd so ParaView opens each run as a time series:

    data_full/compare_vtu/openradioss/<split>/runN/runN.pvd          <- open in ParaView
    data_full/compare_vtu/openradioss/<split>/runN/vtu/runN_000.vtu ... runN_050.vtu

Fields per frame:
    point_data: displacement (vector, relative to t0), thickness
    cell_data:  stress_vm, effective_plastic_strain

Usage:
    python export_vtu.py                               # all 135 runs
    python export_vtu.py --runs run1,run91 --overwrite
    python export_vtu.py --workers 8
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyvista as pv

SPLITS = ("train", "val", "test")
DISP_RE = re.compile(r"^displacement_t(\d+\.\d+)$")


def find_vtps(data_dir: str) -> dict[str, tuple[str, str]]:
    """Map run name -> (split, source .vtp path) across train/val/test."""
    vtps = {}
    for split in SPLITS:
        for path in glob.glob(os.path.join(data_dir, split, "run*.vtp")):
            vtps[os.path.splitext(os.path.basename(path))[0]] = (split, path)
    return dict(sorted(vtps.items(), key=lambda kv: int(kv[0][3:])))


def write_pvd(pvd_path: str, entries: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for t, fname in entries:
        lines.append(f'    <DataSet timestep="{t:g}" group="" part="0" file="{fname}"/>')
    lines += ["  </Collection>", "</VTKFile>", ""]
    with open(pvd_path, "w") as f:
        f.write("\n".join(lines))


def run_paths(out_root: str, split: str, run: str) -> tuple[str, str]:
    """<out_root>/<split>/<run>/<run>.pvd and the vtu/ folder next to it."""
    run_dir = os.path.join(out_root, split, run)
    return os.path.join(run_dir, f"{run}.pvd"), os.path.join(run_dir, "vtu")


def export_run(run: str, split: str, vtp_path: str, out_root: str, overwrite: bool) -> str:
    pvd_path, vtu_dir = run_paths(out_root, split, run)
    if os.path.exists(pvd_path) and not overwrite:
        return f"{run}: already exists, skipping"
    os.makedirs(vtu_dir, exist_ok=True)

    poly = pv.read(vtp_path)
    times = sorted(
        (float(m.group(1)), name)
        for name in poly.point_data.keys()
        if (m := DISP_RE.match(name))
    )
    if not times:
        raise ValueError(f"No displacement_t* fields in {vtp_path}")

    base = pv.PolyData(poly.points.copy(), faces=poly.faces).cast_to_unstructured_grid()
    coords = poly.points.copy()
    entries = []
    for i, (t, disp_name) in enumerate(times):
        grid = base.copy()
        disp = poly.point_data[disp_name]
        grid.points = coords + disp
        grid.point_data["displacement"] = disp
        if "thickness" in poly.point_data:
            grid.point_data["thickness"] = poly.point_data["thickness"]
        for field in ("stress_vm", "effective_plastic_strain"):
            src = f"cell_{field}_t{t:.3f}"
            if src in poly.cell_data:
                grid.cell_data[field] = poly.cell_data[src]
        grid.field_data["TimeValue"] = [t]

        fname = f"{run}_{i:03d}.vtu"
        grid.save(os.path.join(vtu_dir, fname), binary=True)
        entries.append((t, f"vtu/{fname}"))

    write_pvd(pvd_path, entries)
    return f"{run}: wrote {len(entries)} frames"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(here, "..", "data_full", "vtk"),
        help="Folder containing train/val/test runN.vtp files.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output folder (default: data_full/compare_vtu/openradioss).",
    )
    parser.add_argument(
        "--runs", default=None, help="Comma-separated run names, e.g. run1,run91."
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-export runs that already exist."
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_root = os.path.abspath(
        args.out_dir or os.path.join(here, "..", "data_full", "compare_vtu", "openradioss")
    )
    vtps = find_vtps(data_dir)
    if args.runs:
        wanted = set(args.runs.split(","))
        missing = wanted - vtps.keys()
        if missing:
            raise SystemExit(f"Runs not found in {data_dir}: {sorted(missing)}")
        vtps = {k: v for k, v in vtps.items() if k in wanted}

    print(f"Exporting {len(vtps)} runs from {data_dir} -> {out_root}")
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(export_run, run, split, path, out_root, args.overwrite): run
            for run, (split, path) in vtps.items()
        }
        for fut in as_completed(futures):
            run = futures[fut]
            try:
                print(fut.result())
            except Exception as e:  # keep going so one bad run doesn't stop the batch
                failed.append(run)
                print(f"{run}: FAILED ({e})")

    print(f"\nDone: {len(vtps) - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed runs:", ", ".join(sorted(failed, key=lambda r: int(r[3:]))))


if __name__ == "__main__":
    main()
