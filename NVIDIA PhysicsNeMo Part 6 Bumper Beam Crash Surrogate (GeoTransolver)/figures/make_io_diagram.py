"""Draw figures/geotransolver_io.png: GeoTransolver inputs / outputs for the bumper beam,
showing what is compressed into the shared context and what stays per node.

Based on physicsnemo/models/geotransolver (geotransolver.py forward, context_projector.py
GlobalContextBuilder.build_context) and rollout.GeoTransolverOneShot.

Read from bottom to top, like the architecture figure
(demo/images/geotransolver_architecture_bumper_beam.png): inputs at the bottom, output at
the top; the compressed context path on the left, the per-node path on the right.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

NODE = ("#dbe8f7", "#2b6cb0")      # per-node stream (not compressed)
CTX = ("#fde7d4", "#c05621")       # compressed context
PROP = ("#e3f3e3", "#2f855a")      # global properties
OUT = ("#ede4f7", "#6b46c1")       # output
GREY = ("#f2f2f2", "#555555")

fig, ax = plt.subplots(figsize=(12, 12.4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12.4)
ax.axis("off")


def box(x, y, w, h, colors, title, body="", title_size=12, body_size=10):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=edge, lw=2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=edge)
    if body:
        ax.text(x + w / 2, y + h - 0.72, body, ha="center", va="top", fontsize=body_size,
                color="#222222", linespacing=1.45)


def arrow(x0, y0, x1, y1, color, label="", label_xy=None, style="-|>", ls="-", ha="center"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=18,
                                 lw=2, color=color, linestyle=ls))
    if label:
        lx, ly = label_xy or ((x0 + x1) / 2, (y0 + y1) / 2 + 0.15)
        ax.text(lx, ly, label, ha=ha, va="center", fontsize=9.5, color=color, style="italic")


L_X, L_W = 0.4, 4.4    # left column: compressed context path
R_X, R_W = 7.0, 4.4    # right column: per-node path
L_C, R_C = L_X + L_W / 2, R_X + R_W / 2

# ---- inputs (bottom)
ax.text(6.0, 0.75, "INPUTS", ha="center", fontsize=13, fontweight="bold", color="#333333")
box(L_X, 1.2, L_W, 2.1, PROP, "Global properties",
    "[1 x 3], one per run\nvelocity_x, thickness_scale, rwall_origin_y")
box(R_X, 1.2, R_W, 2.1, NODE, "Node coordinates (t0)",
    "[13,675 nodes x 3]\nx, y, z of every mesh node\ngeometry scale (geo_scale_*) is already\nin these coordinates")

# ---- context projector (left) and node tokens (right)
box(L_X, 4.4, L_W, 2.2, CTX, "Context projector",
    "geometry + properties + neighbourhood\nfeatures are pooled over all nodes\n(N nodes -> 128 slices)")
box(R_X, 4.4, R_W, 2.2, NODE, "Node tokens",
    "[13,675 x 320]\none token per node\n(coords -> encoder, + local neighbourhood)")
arrow(L_C, 3.3, L_C, 4.4, CTX[1], "tokenized", (L_C + 0.15, 3.85), ha="left")
arrow(R_X + 0.6, 3.3, L_X + L_W, 4.75, CTX[1], "as geometry", (5.9, 3.7))
arrow(R_C, 3.3, R_C, 4.4, NODE[1], "per node,\nNOT compressed", (R_C + 0.15, 3.85), ha="left")

# ---- shared context (left) and GALE layers (right)
box(L_X, 7.6, L_W, 2.3, CTX, "Shared context  C",
    "[8 heads x 128 slices x D]\nfixed size: does NOT grow with the\nnumber of nodes\nbuilt once, used by all 6 layers")
box(R_X, 7.6, R_W, 2.3, GREY, "6 x GALE layers",
    "every node token reads the shared\ncontext (cross-attention)\nkeeps all 13,675 tokens")
arrow(L_C, 6.6, L_C, 7.6, CTX[1], "compressed", (L_C + 0.15, 7.1), ha="left")
arrow(R_C, 6.6, R_C, 7.6, NODE[1])
arrow(L_X + L_W, 8.75, R_X, 8.75, CTX[1], "read by\nevery node", (5.9, 9.25))

# ---- output (top)
OUT_X, OUT_W = 5.4, 6.0
box(OUT_X, 10.4, OUT_W, 1.55, OUT, "Per-node trajectory  [13,675 x 50 x Fo]",
    "50 future steps (2 ms ... 100 ms)\n"
    "Fo = 3: position (x, y, z), position-only model\n"
    "Fo = 5: position + plastic strain + von Mises stress")
arrow(R_C, 9.9, R_C, 10.4, OUT[1])
ax.text(OUT_X + OUT_W / 2, 12.2, "OUTPUT", ha="center", fontsize=13, fontweight="bold", color="#333333")

# ---- takeaway
ax.text(6.0, 0.2,
        "Compressed (orange): everything that describes the run as a whole -> one fixed-size context shared by all nodes.\n"
        "Not compressed (blue): one token per node, because the model must predict a separate trajectory for each of the 13,675 nodes.",
        ha="center", va="center", fontsize=9.5, color="#333333")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geotransolver_io.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("Saved", out)
