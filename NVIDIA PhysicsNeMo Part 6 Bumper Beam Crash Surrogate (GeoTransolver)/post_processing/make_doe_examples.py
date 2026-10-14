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
Show how each DOE variable changes the crash, with 11 example OpenRadioss runs.

Starting from the baseline run (geometry (1,1,1), vx = -5 mm/ms, thickness 1.0,
wall y = 0), one variable is changed at a time:

    G1..G5  geometry scale   (1,1,1) (1,0.5,1) (1,1,0.5) (1,2,1) (1,1,2)
    T1..T3  thickness scale  1.0 0.7 1.3
    W1..W3  wall origin y    0 120 240

G1, T1 and W1 are the same baseline run, so the 11 cases use 9 different runs.
The cases are looked up in dataset_generation/dataset/summary.csv.

Outputs (ground truth only, read from the export_vtu.py VTU files):
    outputs/doe_gifs/doe_<group>.gif          one GIF per group, one panel per case,
                                              displacement magnitude + rigid wall,
                                              shared color range within a group
    data_full/compare_vtu/doe_11cases/
        doe_11cases.pvd                       <- open in ParaView: all 11 cases side by side
        cases.csv                             case -> run, parameters, tile offset
        vtu/<case>_<run>_000.vtu ... _050.vtu (beam, translated to its tile)
        vtu/<case>_wall.vtu                   (rigid wall cylinder, same tile offset)

In the combined .pvd each group is one row (along +y) and each case one
column (along +x). Part i (0..10) is the beam of case i, part 11 + i its wall.
Every beam cell carries a `case_index` array (0..10, the order of cases.csv).

Usage:
    python make_doe_examples.py                 # GIFs + combined PVD
    python make_doe_examples.py --skip_gif
    python make_doe_examples.py --overwrite
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyvista as pv

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY_CSV = os.path.join(PROJECT_DIR, "dataset_generation", "dataset", "summary.csv")
GT_VTU_ROOT = os.path.join(PROJECT_DIR, "data_full", "compare_vtu", "openradioss")
GIF_DIR = os.path.join(PROJECT_DIR, "outputs", "doe_gifs")
PVD_DIR = os.path.join(PROJECT_DIR, "data_full", "compare_vtu", "doe_11cases")

PARAM_COLS = ["Run_ID", "Geo_Scale_X", "Geo_Scale_Y", "Geo_Scale_Z", "Vel_X", "Vel_Y", "Vel_Z",
              "Thickness_Factor", "Wall_Diameter", "Wall_X", "Wall_Y", "Wall_Z"]
BASELINE = {"Geo_Scale_X": 1.0, "Geo_Scale_Y": 1.0, "Geo_Scale_Z": 1.0, "Vel_X": -5.0,
            "Thickness_Factor": 1.0, "Wall_Y": 0.0}
GEOMETRY_SCALES = [(1.0, 1.0, 1.0), (1.0, 0.5, 1.0), (1.0, 1.0, 0.5), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)]
THICKNESS_SCALES = [1.0, 0.7, 1.3]
WALL_YS = [0.0, 120.0, 240.0]
GROUPS = {"G": "geometry", "T": "thickness", "W": "wall"}
GIF_ZOOM = {"geometry": 1.35, "thickness": 1.3, "wall": 1.3}  # geometry shares one scale with the 2x long beam
TILE_PITCH_X = 900.0    # mm between cases of a group (beam + wall are ~650 mm deep in x)
TILE_PITCH_Y = 4600.0   # mm between groups (longest beam, scale y = 2, is ~4050 mm)


def _find_run(summary: pd.DataFrame, **changes) -> pd.Series:
    want = {**BASELINE, **changes}
    mask = np.ones(len(summary), dtype=bool)
    for col, val in want.items():
        mask &= np.isclose(summary[col], val)
    rows = summary[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected 1 run for {want}, found {len(rows)}")
    return rows.iloc[0]


def select_cases(summary_csv: str = SUMMARY_CSV) -> pd.DataFrame:
    """The 11 example cases with their DOE parameters, ground-truth .pvd and tile offset."""
    summary = pd.read_csv(summary_csv)
    picks = []
    for sx, sy, sz in GEOMETRY_SCALES:
        picks.append(("G", f"geo ({sx:g}, {sy:g}, {sz:g})",
                      _find_run(summary, Geo_Scale_X=sx, Geo_Scale_Y=sy, Geo_Scale_Z=sz)))
    for th in THICKNESS_SCALES:
        picks.append(("T", f"thickness {th:g}", _find_run(summary, Thickness_Factor=th)))
    for wy in WALL_YS:
        picks.append(("W", f"wall y = {wy:g}", _find_run(summary, Wall_Y=wy)))

    rows, counters = [], {g: 0 for g in GROUPS}
    for row_idx, (group, label, run) in enumerate(picks):
        counters[group] += 1
        g_row = list(GROUPS).index(group)
        pvd = glob.glob(os.path.join(GT_VTU_ROOT, "*", run.Run_ID, f"{run.Run_ID}.pvd"))
        rows.append({
            "Case": f"{group}{counters[group]}",
            "Group": GROUPS[group],
            "Varied": label,
            **{c: run[c] for c in PARAM_COLS},
            "Split": os.path.basename(os.path.dirname(os.path.dirname(pvd[0]))) if pvd else None,
            "GT_PVD": pvd[0] if pvd else None,
            "Offset_X": (counters[group] - 1) * TILE_PITCH_X,
            "Offset_Y": g_row * TILE_PITCH_Y,
        })
    return pd.DataFrame(rows)


def _frames(pvd_path: str) -> list[str]:
    run = os.path.splitext(os.path.basename(pvd_path))[0]
    return sorted(glob.glob(os.path.join(os.path.dirname(pvd_path), "vtu", f"{run}_[0-9][0-9][0-9].vtu")))


def make_wall(case: pd.Series, z_extent: float) -> pv.PolyData:
    """/RWALL/CYL: axis along z through (Wall_X, Wall_Y, Wall_Z), diameter Wall_Diameter."""
    return pv.Cylinder(center=(case.Wall_X, case.Wall_Y, case.Wall_Z), direction=(0, 0, 1),
                       radius=case.Wall_Diameter / 2, height=z_extent, resolution=48)


def make_group_gif(cases: pd.DataFrame, output_gif: str, frame_step: int = 2, fps: int = 6,
                   camera: str = "iso", cmap: str = "jet", panel_size=(520, 440), zoom: float = 1.0) -> None:
    """One panel per case: deformed beam colored by |displacement| (shared color range) + wall."""
    series = []  # (case, [T] meshes, |u| [T,N])
    for _, case in cases.iterrows():
        meshes = [pv.read(f) for f in _frames(case.GT_PVD)[::frame_step]]
        mags = np.stack([np.linalg.norm(m.point_data["displacement"], axis=1) for m in meshes])
        series.append((case, meshes, mags))
    vmax = max(float(mags.max()) for _, _, mags in series)

    # One shared view box, so the panels are drawn at the same scale
    bounds = np.array([m.bounds for _, meshes, _ in series for m in meshes])
    walls = [make_wall(case, 300.0) for case, _, _ in series]
    wall_b = np.array([w.bounds for w in walls])
    lo = np.minimum(bounds[:, 0::2].min(0), wall_b[:, 0::2].min(0))
    hi = np.maximum(bounds[:, 1::2].max(0), wall_b[:, 1::2].max(0))
    view_bounds = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    n = len(series)
    plotter = pv.Plotter(shape=(1, n), off_screen=True, window_size=(panel_size[0] * n, panel_size[1]))
    live = []
    for col, ((case, meshes, mags), wall) in enumerate(zip(series, walls)):
        plotter.subplot(0, col)
        mesh = meshes[0].copy()
        mesh["Displacement (mm)"] = mags[0]
        plotter.add_mesh(mesh, scalars="Displacement (mm)", cmap=cmap, clim=[0, vmax],
                         show_scalar_bar=(col == n - 1))
        plotter.add_mesh(wall, color="lightgray", opacity=0.45)
        plotter.add_text(f"{case.Case}  {case.Run_ID}\n{case.Varied}", font_size=10)
        plotter.camera_position = camera
        plotter.camera.parallel_projection = True
        plotter.reset_camera(bounds=view_bounds)
        plotter.camera.zoom(zoom)
        plotter.add_axes(line_width=4, viewport=(0, 0, 0.3, 0.35))  # x/y/z triad (impact along -x, beam along y)
        live.append((mesh, meshes, mags))

    plotter.open_gif(output_gif, fps=fps)
    for t in range(len(series[0][1])):
        for mesh, meshes, mags in live:
            mesh.points = meshes[t].points
            mesh["Displacement (mm)"] = mags[t]
        plotter.write_frame()
    plotter.close()
    print(f"Saved {output_gif} ({len(series[0][1])} frames, {n} cases, |u| max {vmax:.0f} mm)")


def write_combined_pvd(cases: pd.DataFrame, out_dir: str = PVD_DIR, overwrite: bool = False) -> str:
    """All cases tiled in one scene: one .pvd, one part per beam and per wall."""
    vtu_dir = os.path.join(out_dir, "vtu")
    os.makedirs(vtu_dir, exist_ok=True)
    pvd_path = os.path.join(out_dir, "doe_11cases.pvd")
    entries = []  # (time, part, relative file)
    dt = 0.002    # same frame times as export_vtu.py (2 ms ANIM interval)
    n_cases = len(cases)

    for idx, (_, case) in enumerate(cases.iterrows()):
        shift = np.array([case.Offset_X, case.Offset_Y, 0.0])
        frames = _frames(case.GT_PVD)
        z_extent = 0.0
        for t, src in enumerate(frames):
            name = f"{case.Case}_{case.Run_ID}_{t:03d}.vtu"
            dst = os.path.join(vtu_dir, name)
            if overwrite or not os.path.exists(dst) or t == 0:
                grid = pv.read(src)
                if t == 0:
                    z_extent = 2.0 * (grid.bounds[5] - grid.bounds[4])
                if overwrite or not os.path.exists(dst):
                    grid.points = grid.points + shift
                    grid.cell_data["case_index"] = np.full(grid.n_cells, idx, dtype=np.int32)
                    grid.save(dst)
            entries.append((t * dt, idx, f"vtu/{name}"))

        wall_name = f"{case.Case}_wall.vtu"
        wall_path = os.path.join(vtu_dir, wall_name)
        if overwrite or not os.path.exists(wall_path):
            wall = make_wall(case, z_extent).translate(shift).cast_to_unstructured_grid()
            wall.save(wall_path)
        entries += [(t * dt, n_cases + idx, f"vtu/{wall_name}") for t in range(len(frames))]

    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             "  <Collection>"]
    for t, part, fname in sorted(entries):
        lines.append(f'    <DataSet timestep="{t:g}" group="" part="{part}" file="{fname}"/>')
    lines += ["  </Collection>", "</VTKFile>", ""]
    with open(pvd_path, "w") as f:
        f.write("\n".join(lines))

    cases.drop(columns=["GT_PVD"]).to_csv(os.path.join(out_dir, "cases.csv"), index=False)
    print(f"Saved {pvd_path} ({n_cases} cases x {len(frames)} frames, beams = parts 0-{n_cases - 1}, "
          f"walls = parts {n_cases}-{2 * n_cases - 1})")
    return pvd_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip_gif", action="store_true")
    parser.add_argument("--skip_pvd", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera", default="iso")
    args = parser.parse_args()

    cases = select_cases()
    print(cases[["Case", "Varied"] + PARAM_COLS].to_string(index=False))
    if not args.skip_gif:
        os.makedirs(GIF_DIR, exist_ok=True)
        for group in GROUPS.values():
            gif = os.path.join(GIF_DIR, f"doe_{group}.gif")
            if os.path.exists(gif) and not args.overwrite:
                print(f"{gif}: already exists, skipping")
                continue
            make_group_gif(cases[cases.Group == group], gif, camera=args.camera, zoom=GIF_ZOOM[group])
    if not args.skip_pvd:
        write_combined_pvd(cases, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
