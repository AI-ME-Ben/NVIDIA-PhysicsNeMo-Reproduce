"""Draw figures/pipeline_overview.png: the whole bumper beam pipeline, from the DOE table to
the result plots, with the scripts, the files in between, and the notebook / guide section of
every step (demo/bumper_beam_pipeline_tutorial.ipynb, BUMPER_BEAM_REPRODUCTION.md).

Read from top to bottom. The upper band runs in WSL (OpenRadioss and Curator are Linux
tools), the lower band in native Windows (CUDA training). The main chain is on the left;
side branches (stress / strain, ParaView, model comparison) are on the right.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

CAE = ("#e3f3e3", "#2f855a")       # simulation (OpenRadioss)
CONV = ("#fde7d4", "#c05621")      # file conversion / ETL
ML = ("#dbe8f7", "#2b6cb0")        # training / inference
OUT = ("#ede4f7", "#6b46c1")       # results you look at
SIDE = ("#f2f2f2", "#555555")      # optional branches

fig, ax = plt.subplots(figsize=(13, 17.8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 17.8)
ax.axis("off")


def box(x, y, w, h, colors, title, tag="", body="", ls="-"):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=fill, ec=edge, lw=2, linestyle=ls))
    ax.text(x + 0.18, y + h - 0.2, title, ha="left", va="top", fontsize=11.5,
            fontweight="bold", color=edge)
    if tag:
        ax.text(x + w - 0.15, y + h - 0.2, tag, ha="right", va="top", fontsize=9.5,
                color=edge, fontweight="bold")
    if body:
        ax.text(x + 0.18, y + h - 0.6, body, ha="left", va="top", fontsize=9.3,
                color="#222222", linespacing=1.4)


def arrow(x0, y0, x1, y1, color="#444444", label="", label_xy=None, ls="-", ha="left", rad=0.0,
          conn=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.8,
                                 color=color, linestyle=ls,
                                 connectionstyle=conn or f"arc3,rad={rad}"))
    if label:
        lx, ly = label_xy or ((x0 + x1) / 2 + 0.12, (y0 + y1) / 2)
        ax.text(lx, ly, label, ha=ha, va="center", fontsize=8.8, color=color, style="italic")


ax.text(6.5, 17.55, "Bumper beam pipeline: from CAE data to a crash surrogate", ha="center",
        fontsize=16, fontweight="bold", color="#222222")

# ---- legend
lx, ly = 0.4, 17.0
for i, (c, name) in enumerate([(CAE, "simulation"), (CONV, "conversion / ETL"), (ML, "training / inference"),
                               (OUT, "results to look at"), (SIDE, "branch (optional)")]):
    ax.add_patch(FancyBboxPatch((lx + i * 2.5, ly - 0.12), 0.32, 0.24, boxstyle="round,pad=0.01",
                                fc=c[0], ec=c[1], lw=1.5))
    ax.text(lx + i * 2.5 + 0.42, ly, name, fontsize=9.3, va="center", color="#333333")

# ---- environment bands
ax.add_patch(Rectangle((0.1, 7.55), 12.8, 9.05, fc="#fafaf5", ec="#bbbbbb", lw=1, ls="--"))
ax.text(0.25, 16.42, "WSL2  ·  Ubuntu 24.04  ·  /opt/radioss_venv  (OpenRadioss, Curator)",
        fontsize=10.5, fontweight="bold", color="#666666", va="center")
ax.add_patch(Rectangle((0.1, 0.1), 12.8, 7.3, fc="#f5f8fc", ec="#bbbbbb", lw=1, ls="--"))
ax.text(12.75, 7.22, "Windows  ·  .venv (PhysicsNeMo editable)  ·  CUDA GPU", fontsize=10.5,
        fontweight="bold", color="#666666", va="center", ha="right")

MX, MW = 0.4, 6.6      # main chain
SX, SW = 7.6, 4.8      # side branches
H = 1.3

# ---- WSL: simulation and ETL
y1 = 14.75
box(MX, y1, MW, H, CAE, "1  DOE table", "Part 2.2",
    "generate_dataset.py  ->  dataset/run1 ... run135/\n"
    "5 geometry x 3 velocity x 3 thickness x 3 wall y = 135 runs\n"
    "each: _0000.rad (starter), _0001.rad (engine), runN.json")
y2 = 12.95
box(MX, y2, MW, H, CAE, "2  OpenRadioss batch", "Part 2.3",
    "run_simulations.py: starter -> engine (5 jobs x 2 threads, i5-13400F)\n"
    "ANIM frames A001 ... A051 (0 to 100 ms, every 2 ms)\n"
    "~1,200 s per run in parallel, 135 runs in 8 to 10 h")
box(SX, y2 + 0.1, SW, 1.1, CONV, "global_features.json", "Part 2.5",
    "runN.json -> summary.json -> per-run velocity_x,\n"
    "thickness_scale, ...  ->  train.py (step 6)")
y3 = 11.15
box(MX, y3, MW, H, CONV, "3  ANIM -> d3plot", "Part 2.3",
    "vortex_radioss.Anim_to_D3plot  ->  d3plot, d3plot01 ... d3plot51\n"
    "(drops von Mises stress and plastic strain)\n"
    "rename_d3plot.py: <BASE>.d3plot*  ->  d3plot*  (Curator needs this name)")
box(SX, y3, SW, H, SIDE, "3b  ANIM -> .vtk", "Part 2.3",
    "anim_to_vtk  ->  A001.vtk ... A051.vtk\n"
    "keeps 2DELEM_Von_Mises, 2DELEM_Plastic_Strain\n"
    "not read by Curator")
y4 = 9.35
box(MX, y4, MW, H, CONV, "4  Curator ETL", "Part 3",
    "run_etl.py crash_etl, input_dir = dataset/   ->  runN.vtp (135 files, ~44 s)\n"
    "reads 4 d3plot arrays; removes rigid wall nodes: 13,882 -> 13,675\n"
    "point data: displacement_t0.000 ... t0.100 (51 frames)")
box(SX, y4, SW, H, SIDE, "4b  Add stress / strain", "Part 6.2",
    "_inject_stress_strain.py: .vtk cells -> VTP\n"
    "by ELEMENT_ID: cell_stress_vm_t*,\ncell_effective_plastic_strain_t*", ls="--")

arrow(MX + MW / 2, y1, MX + MW / 2, y2 + H)
arrow(MX + MW, y1 + 0.35, SX + 1.0, y2 + 1.2, CONV[1], "runN.json", (7.5, 14.35), rad=-0.2)
arrow(MX + MW / 2, y2, MX + MW / 2, y3 + H)
arrow(MX + MW - 0.4, y2, SX + 0.4, y3 + H, SIDE[1], "same ANIM files", (7.1, 12.7), rad=-0.15)
arrow(MX + MW / 2, y3, MX + MW / 2, y4 + H)
arrow(SX + SW / 2, y3, SX + SW / 2, y4 + H, SIDE[1], "Part 6 only", (SX + SW / 2 + 0.12, 10.8))
arrow(SX + 0.5, y4, MX + MW / 2 + 0.08, 8.15, SIDE[1], "adds cell arrays to runN.vtp", (6.3, 8.75),
      ls="--", ha="center")

# ---- Windows: split, export, train, infer, results
y6 = 5.6
box(MX, y6, MW, H, CONV, "5  Split", "Part 3.1",
    "data_full/vtk/train 121 | val 5 | test 9 (run91 ... run99)\n"
    "text sort of run names; test = one unseen DOE combination\n"
    "(geometry (1,2,1) x velocity -3)")
box(SX, y6, SW, H, OUT, "5b  Ground truth for ParaView", "Part 2.4",
    "export_vtu.py -> openradioss/<split>/runN/runN.pvd\n"
    "make_doe_examples.py -> 11 DOE GIFs\n"
    "    + doe_11cases/doe_11cases.pvd")
y7 = 3.85
box(MX, y7, MW, H, ML, "6  Train GeoTransolverOneShot", "Part 4",
    "train.py --config-name=bumper_openradioss_oneshot\n"
    "inputs: t0 coordinates + global_features.json;  output [13,675 x 50 x Fo]\n"
    "outputs/checkpoints_*, TensorBoard; plateau after a few hundred epochs")
box(SX, y7, SW, H, SIDE, "6b  Other models", "Part 9",
    "bumper_mgn_oneshot, bumper_dignn_oneshot\n"
    "same data -> MeshGraphNet, DI-GNN\n"
    "outputs_{geotransolver,mgn,dignn}/")
y8 = 2.1
box(MX, y8, MW, H, ML, "7  Inference + prediction VTU", "Part 5.1 - 5.3",
    "inference.py -> predicted_vtps_* / exact_vtps_*  (one run per model)\n"
    "export_pred_vtu.py -> compare_vtu/predict_disponly/, predict/\n"
    "metrics_disponly.csv, metrics.csv: relative L2 error (%) per run")
y9 = 0.4
box(MX, y9, MW, H, OUT, "8  Results", "Part 5.2 - 5.4, 6.4, 7",
    "position-only model first, then position + stress / strain model\n"
    "GIFs (outputs/test_gifs_disponly/, test_gifs/), error over time\n"
    "side-by-side comparison, per-field MSE vs epoch, per-node R²")
box(SX, y9, SW, H + 1.75, OUT, "ParaView", "",
    "open side by side:\n"
    "openradioss/<split>/runN/runN.pvd\n"
    "predict_disponly/<split>/runN/runN.pvd\n"
    "predict/<split>/runN/runN.pvd\n\n"
    "or all 11 DOE cases at once:\n"
    "doe_11cases/doe_11cases.pvd")

arrow(MX + MW / 2, y4, MX + MW / 2, y6 + H, "#444444", "runN.vtp\n(on /mnt/c, same disk)", (MX + 0.35, 8.2))
arrow(MX + MW, y6 + H / 2, SX, y6 + H / 2)
arrow(MX + MW / 2, y6, MX + MW / 2, y7 + H)
arrow(MX + MW, y7 + H / 2, SX, y7 + H / 2, SIDE[1], "", ls="--")
arrow(MX + MW / 2, y7, MX + MW / 2, y8 + H)
arrow(MX + MW / 2, y8, MX + MW / 2, y9 + H)
arrow(MX + MW, y8 + H / 2, SX, y8 + 0.35, OUT[1])
# 5b -> ParaView, routed along the right edge so it does not cross 6b
arrow(SX + SW, y6 + 0.35, SX + SW, y9 + H + 1.2, OUT[1], conn="bar,fraction=-0.06")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_overview.png")
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print("Saved", out)
