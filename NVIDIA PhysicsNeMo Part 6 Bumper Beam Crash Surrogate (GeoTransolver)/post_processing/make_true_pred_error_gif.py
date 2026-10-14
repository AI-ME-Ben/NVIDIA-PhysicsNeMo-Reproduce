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
Render a Ground Truth / Prediction / Error animation (GIF) for one inference
run, from the frame_*_exact.vtp / frame_*_pred.vtp files written by
inference.py. Uses PyVista's own renderer (solid shaded mesh, jet colormap,
fixed orthographic camera) rather than matplotlib, to match a clean FE-style
visualization.

By default renders one row per field (displacement magnitude, von Mises
stress, effective plastic strain) x three columns:
    Ground Truth | Prediction | Error = abs(Prediction - Ground Truth)
The error column is drawn on the true deformed shape. For displacement it is
the position error |x_pred - x_true| (mm); for stress/strain it is the
absolute difference. It has its own color range (0 to the max error over all
frames).

Stress/strain require the run's VTP to carry the dynamic targets (see
dataset_generation/templates/_inject_stress_strain.py and
conf/bumper_openradioss_oneshot.yaml's `dynamic_targets`); pass
--fields displacement to render position-only.

The same renderer (render_true_pred_error_image) is used by train.py to log a
Ground Truth / Prediction / Error image to TensorBoard at each validation.

Usage:
    python make_true_pred_error_gif.py \
        --exact_dir ../outputs/exact_vtps_stress_final/rank0/run91 \
        --pred_dir ../outputs/predicted_vtps_stress_final/rank0/run91 \
        --reference_vtp ../data_full/vtk/test/run91.vtp \
        --output_gif run91_true_pred.gif \
        --fields displacement,stress_vm,effective_plastic_strain
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pyvista as pv

FIELD_SPECS = {
    "displacement": {
        "label": "Displacement Magnitude (mm)",
        "error_label": "Position Error (mm)",
        "kind": "displacement",
    },
    "stress_vm": {
        "label": "von Mises Stress",
        "error_label": "Stress Error (abs)",
        "kind": "scalar",
        "exact_key": "exact_stress_vm",
        "pred_key": "pred_stress_vm",
    },
    "effective_plastic_strain": {
        "label": "Effective Plastic Strain",
        "error_label": "Plastic Strain Error (abs)",
        "kind": "scalar",
        "exact_key": "exact_effective_plastic_strain",
        "pred_key": "pred_effective_plastic_strain",
    },
}
COLUMN_HEADERS = (
    "Ground Truth (CAE/OpenRadioss)",
    "Prediction (GeoTransolver)",
    "Error: abs(Prediction - Ground Truth)",
)


def _sorted_frames(directory: str, suffix: str) -> list[str]:
    files = glob.glob(os.path.join(directory, f"*_{suffix}.vtp"))

    def frame_idx(path: str) -> int:
        m = re.search(r"(\d+)_" + suffix + r"\.vtp$", os.path.basename(path))
        return int(m.group(1))

    return sorted(files, key=frame_idx)


def _field_series(coords0, true_pos, pred_pos, fields, true_scalars, pred_scalars):
    """Per field: (spec, true[T,N], pred[T,N], error[T,N], vmax, emax)."""
    out = []
    for name in fields:
        spec = FIELD_SPECS[name]
        if spec["kind"] == "displacement":
            true_series = np.linalg.norm(true_pos - coords0[None], axis=-1)
            pred_series = np.linalg.norm(pred_pos - coords0[None], axis=-1)
            err_series = np.linalg.norm(pred_pos - true_pos, axis=-1)
        else:
            true_series = np.asarray(true_scalars[name], dtype=np.float32)
            pred_series = np.asarray(pred_scalars[name], dtype=np.float32)
            err_series = np.abs(pred_series - true_series)
        vmax = float(max(true_series.max(), pred_series.max(), 1e-6))
        emax = float(max(err_series.max(), 1e-6))
        out.append((spec, true_series, pred_series, err_series, vmax, emax))
    return out


def _build_plotter(faces, true_pos, pred_pos, series, t0, camera, cmap, window_size):
    """Create the (rows x 3) plotter at frame t0; return it and an update(t) function."""
    n_rows = len(series)
    if window_size is None:
        window_size = (2100, 620 * n_rows)
    plotter = pv.Plotter(shape=(n_rows, 3), off_screen=True, window_size=window_size)

    meshes = []  # (mesh, positions[T,N,3], values[T,N], label) for every subplot
    for row, (spec, true_s, pred_s, err_s, vmax, emax) in enumerate(series):
        columns = [
            (true_pos, true_s, spec["label"], vmax),
            (pred_pos, pred_s, spec["label"], vmax),
            (true_pos, err_s, spec["error_label"], emax),
        ]
        for col, (pos, values, label, cmax) in enumerate(columns):
            mesh = pv.PolyData(pos[t0].copy(), faces)
            mesh[label] = values[t0]
            plotter.subplot(row, col)
            plotter.add_mesh(
                mesh, scalars=label, cmap=cmap, clim=[0, cmax],
                show_scalar_bar=True, show_edges=False,
            )
            if row == 0:
                plotter.add_text(COLUMN_HEADERS[col], font_size=12)
            plotter.camera_position = camera
            plotter.camera.parallel_projection = True
            plotter.reset_camera()
            plotter.show_axes()
            meshes.append((mesh, pos, values, label))

    def update(t):
        for mesh, pos, values, label in meshes:
            mesh.points = pos[t]
            mesh[label] = values[t]

    return plotter, update


def render_true_pred_error_gif(
    coords0, faces, true_pos, pred_pos, output_gif,
    fields=("displacement",), true_scalars=None, pred_scalars=None,
    fps=8, camera="iso", cmap="jet", window_size=None,
) -> None:
    """true_pos/pred_pos: [T,N,3] including t0; true_scalars/pred_scalars: {field: [T,N]}."""
    series = _field_series(coords0, true_pos, pred_pos, fields, true_scalars, pred_scalars)
    plotter, update = _build_plotter(faces, true_pos, pred_pos, series, 0, camera, cmap, window_size)
    plotter.open_gif(output_gif, fps=fps)
    plotter.write_frame()  # frame 0 (t=0, reference / no deformation)
    for t in range(1, true_pos.shape[0]):
        update(t)
        plotter.write_frame()
    plotter.close()


def render_true_pred_error_image(
    coords0, faces, true_pos, pred_pos,
    fields=("displacement",), true_scalars=None, pred_scalars=None,
    t=-1, camera="iso", cmap="jet", row_size=(1500, 400),
) -> np.ndarray:
    """Render one frame (default: the last) as an RGB array [H, W, 3]."""
    series = _field_series(coords0, true_pos, pred_pos, fields, true_scalars, pred_scalars)
    t = t % true_pos.shape[0]
    window_size = (row_size[0], row_size[1] * len(series))
    plotter, _ = _build_plotter(faces, true_pos, pred_pos, series, t, camera, cmap, window_size)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def build_multi_gif(
    exact_dir: str,
    pred_dir: str,
    reference_vtp: str,
    output_gif: str,
    fields: list[str] = ("displacement", "stress_vm", "effective_plastic_strain"),
    fps: int = 8,
    camera: str = "iso",
    cmap: str = "jet",
    window_size: tuple[int, int] | None = None,
) -> None:
    exact_files = _sorted_frames(exact_dir, "exact")
    pred_files = _sorted_frames(pred_dir, "pred")
    if not exact_files or not pred_files:
        raise FileNotFoundError(f"No frames found under {exact_dir} or {pred_dir}")

    ref = pv.read(reference_vtp)
    coords0 = ref.points.copy()
    faces = ref.faces.copy()

    exact_meshes = [pv.read(f) for f in exact_files]
    pred_meshes = [pv.read(f) for f in pred_files]

    exact_pos = np.stack([coords0] + [m.points for m in exact_meshes], axis=0)  # [T,N,3]
    pred_pos = np.stack([coords0] + [m.points for m in pred_meshes], axis=0)  # [T,N,3]
    T, N, _ = exact_pos.shape

    true_scalars, pred_scalars = {}, {}
    for name in fields:
        spec = FIELD_SPECS[name]
        if spec["kind"] != "scalar":
            continue
        true_series = np.zeros((T, N), dtype=np.float32)
        pred_series = np.zeros((T, N), dtype=np.float32)
        for t, m in enumerate(exact_meshes, start=1):
            if spec["exact_key"] not in m.point_data:
                raise KeyError(
                    f"'{spec['exact_key']}' not found in {exact_files[t - 1]} — "
                    "this run's VTP may not have dynamic_targets injected "
                    "(see _inject_stress_strain.py), or the checkpoint's "
                    "config didn't include this target."
                )
            true_series[t] = np.asarray(m.point_data[spec["exact_key"]]).squeeze()
            pred_series[t] = np.asarray(m.point_data[spec["pred_key"]]).squeeze()
        true_scalars[name] = true_series
        pred_scalars[name] = pred_series

    render_true_pred_error_gif(
        coords0, faces, exact_pos, pred_pos, output_gif,
        fields=fields, true_scalars=true_scalars, pred_scalars=pred_scalars,
        fps=fps, camera=camera, cmap=cmap, window_size=window_size,
    )
    print(f"Saved {output_gif} ({T} frames, {len(fields)} field(s): {', '.join(fields)})")


def build_gif(
    exact_dir: str,
    pred_dir: str,
    reference_vtp: str,
    output_gif: str,
    fps: int = 8,
    camera: str = "iso",
    cmap: str = "jet",
    window_size: tuple[int, int] = (2100, 620),
) -> None:
    """Backward-compatible single-field (displacement only) entry point."""
    build_multi_gif(
        exact_dir, pred_dir, reference_vtp, output_gif,
        fields=["displacement"], fps=fps, camera=camera, cmap=cmap,
        window_size=window_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--reference_vtp", required=True)
    parser.add_argument("--output_gif", required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--camera",
        default="iso",
        help="PyVista camera shorthand: iso, xy, xz, yz (or negated, e.g. -yz).",
    )
    parser.add_argument("--cmap", default="jet")
    parser.add_argument(
        "--fields",
        default="displacement,stress_vm,effective_plastic_strain",
        help="Comma-separated subset of: " + ", ".join(FIELD_SPECS.keys()),
    )
    args = parser.parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    build_multi_gif(
        args.exact_dir,
        args.pred_dir,
        args.reference_vtp,
        args.output_gif,
        fields=fields,
        fps=args.fps,
        camera=args.camera,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
