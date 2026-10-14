import glob
import sys
import time

import numpy as np
import pyvista as pv
from lasso.dyna import D3plot, ArrayType

DATASET = "/mnt/c/code/2-cae/physicsnemo/examples/structural_mechanics/openradioss_dataset_gen/bumper_beam/dataset"
VTP_DIR = "/tmp/bumper_full_vtp"
NUM_FRAMES = 51
TIME_STEP = 0.002


def extract_frame_arrays(vtk_path, shell_ids):
    m = pv.read(vtk_path)
    ctypes = np.asarray(m.celltypes)
    elem_id = np.asarray(m.cell_data["ELEMENT_ID"])
    vonm = np.asarray(m.cell_data["2DELEM_Von_Mises"])
    eps = np.asarray(m.cell_data["2DELEM_Plastic_Strain"])
    is_2d = np.isin(ctypes, [5, 9])
    shell_id_set = set(shell_ids.tolist())
    cand = np.where(is_2d & np.isin(elem_id, list(shell_id_set)))[0]
    id_to_cell = {}
    for ci in cand:
        cid = elem_id[ci]
        if cid not in id_to_cell:
            id_to_cell[cid] = ci
    missing = [sid for sid in shell_ids if sid not in id_to_cell]
    if missing:
        raise ValueError(f"{vtk_path}: missing {len(missing)} shell ids")
    idx = np.array([id_to_cell[sid] for sid in shell_ids])
    return vonm[idx].astype(np.float32), eps[idx].astype(np.float32)


def process_run(run_dir):
    run_name = run_dir.rstrip("/").split("/")[-1]
    d3plot_path = f"{run_dir}/d3plot"
    vtp_path = f"{VTP_DIR}/{run_name}.vtp"

    dp = D3plot(d3plot_path)
    shell_ids = dp.arrays[ArrayType.element_shell_ids]

    vtp = pv.read(vtp_path)
    if vtp.n_cells != len(shell_ids):
        raise ValueError(f"{run_name}: cell count mismatch {vtp.n_cells} vs {len(shell_ids)}")

    for frame_idx in range(1, NUM_FRAMES + 1):
        t = (frame_idx - 1) * TIME_STEP
        vtk_path = f"{run_dir}/Bumper_Beam_AP_meshedA{frame_idx:03d}.vtk"
        vonm, eps = extract_frame_arrays(vtk_path, shell_ids)
        vtp.cell_data[f"cell_stress_vm_t{t:.3f}"] = vonm
        vtp.cell_data[f"cell_effective_plastic_strain_t{t:.3f}"] = eps

    vtp.save(vtp_path)
    return run_name


def main():
    run_dirs = sorted(glob.glob(f"{DATASET}/run*/"))
    print("total runs:", len(run_dirs), flush=True)
    t0 = time.time()
    for i, run_dir in enumerate(run_dirs):
        run_name = process_run(run_dir)
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(run_dirs)}] {run_name} done (elapsed {elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
