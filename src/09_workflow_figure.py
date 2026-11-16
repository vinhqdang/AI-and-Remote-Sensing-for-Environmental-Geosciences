"""Workflow schematic (Fig. 0) for HiPhyBoost transferable coseismic landslide forecasting.

Drawn entirely with matplotlib patches and text in a 0-100 x 0-60 data space.
Writes figures/fig0_workflow.pdf and figures/fig0_workflow.png (300 dpi).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

BLUE, RED, GREEN, ORANGE = "#4e79a7", "#e15759", "#59a14f", "#f28e2b"
EDGE, TXT, ARROW = "#3a3a3a", "#222222", "#333333"
BAND_A, BAND_B, BAND_C = "#eaf1f8", "#fbf3e8", "#edf5ea"
FS = 6.8

fig = plt.figure(figsize=(7.1, 4.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")


def band(y0, y1, color, label):
    ax.add_patch(FancyBboxPatch((0.3, y0), 99.4, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                fc=color, ec="none", zorder=0))
    ax.text(2.8, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
            fontsize=7, fontweight="bold", color=TXT, linespacing=1.15, zorder=1)


def box(x0, y0, x1, y1, fc="white", ec=EDGE, lw=0.6, text=None, fs=FS,
        weight="normal", zorder=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, zorder=zorder))
    if text is not None:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=fs, color=TXT, fontweight=weight, linespacing=1.25,
                zorder=zorder + 1)


def arrow(p0, p1, color=ARROW, lw=0.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))


# ---------------- Band A: inputs ----------------
band(42.5, 59.5, BAND_A, "Inputs\n(open, rapid)")
box(6.5, 49.5, 34, 58.5,
    text="Copernicus GLO-30 DEM\n(TanDEM-X InSAR)\n→ slope, relief, curvature,\nTPI, roughness")
box(6.5, 43.5, 34, 48.5, text="USGS ShakeMap\n→ PGA, PGV")
box(41, 43.5, 66, 58.5, fc="#d9e6f2", ec=BLUE, lw=0.8,
    text="Newmark sliding block\nFS, $a_c$, $D_N$ = $f(a_c/\\mathrm{PGA})$")
box(73, 43.5, 99, 58.5,
    text="Past-event inventories\n(8 Pacific Rim earthquakes)")

arrow((34, 54), (41, 53))          # DEM -> Newmark
arrow((34, 46), (41, 47))          # ShakeMap -> Newmark
arrow((53.5, 43.5), (53.5, 37.5))  # Newmark -> training
arrow((86, 43.5), (70, 37.5))      # inventories -> training (event offsets)

# ---------------- Band B: training ----------------
band(16.5, 38.5, BAND_B, "Training on past\nevents (HiPhyBoost)")
box(6.5, 17.5, 76, 37.5, ec=RED, lw=0.8)
ax.text(41.25, 35.0, r"$\mathrm{logit}\,p_i=\varphi(z_i)+g(\mathbf{x}_i)+b_{e(i)}$",
        ha="center", va="center", fontsize=8.5, color=TXT, zorder=5)
ax.text(9.5, 31.8, "φ: Newmark anchor (β ≥ 0)", ha="left", va="center", fontsize=6.5, zorder=5)
ax.text(9.5, 30.2, "g: monotone-constrained boosted residual", ha="left", va="center",
        fontsize=6.5, zorder=5)
ax.text(9.5, 28.6, r"$b_e$: latent event offset", ha="left", va="center", fontsize=6.5,
        zorder=5)

# block-coordinate descent loop (three steps, circular return arrow)
steps = [
    (8.5, 22.5, "1  fit φ\n| g, b", "#fbe3d0"),
    (27.0, 41.0, "2  boost g\n| φ + b", "#fbe3d0"),
    (45.5, 59.5, "3  solve $b_e$\n(score eq.)", "#fbe3d0"),
]
for x0, x1, s, fc in steps:
    box(x0, 20.5, x1, 26.5, fc=fc, ec=ORANGE, lw=0.7, text=s, zorder=4)
arrow((22.5, 23.5), (27.0, 23.5), color=ORANGE, lw=0.8)
arrow((41.0, 23.5), (45.5, 23.5), color=ORANGE, lw=0.8)
arrow((52.5, 20.5), (15.5, 20.5), color=ORANGE, lw=0.8, rad=-0.12)
ax.text(61.5, 23.5, "block-coordinate\ndescent, T = 3", ha="left", va="center",
        fontsize=6.5, style="italic", color=TXT, linespacing=1.25, zorder=5)

# Empirical-Bayes prior with inset Gaussian and event-offset dots
box(80, 17.5, 99, 37.5, fc="#e4f1de", ec=GREEN, lw=0.8)
ax.text(89.5, 35.4, "Empirical-Bayes prior", ha="center", va="center",
        fontsize=6.5, fontweight="bold", zorder=5)
ax.text(89.5, 33.2, r"$b \sim N(\mu, \tau^2)$", ha="center", va="center",
        fontsize=7, zorder=5)
ax.text(89.5, 31.2, "dots: 8 event offsets", ha="center", va="center",
        fontsize=5.8, color="#444444", zorder=5)

ax2 = fig.add_axes([82 / 100, 18.6 / 60, 15 / 100, 11.6 / 60])
ax2.patch.set_alpha(0)
xs = np.linspace(-3.2, 3.2, 300)
pdf = np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)
ax2.fill_between(xs, 0, pdf, color=GREEN, alpha=0.25, lw=0)
ax2.plot(xs, pdf, color=GREEN, lw=0.8)
offsets = np.array([-1.6, -0.9, -0.4, 0.0, 0.3, 0.7, 1.1, 1.8])
ax2.scatter(offsets, np.zeros_like(offsets), s=7, color=ORANGE, zorder=4,
            clip_on=False, linewidths=0)
ax2.set_xlim(-3.4, 3.4)
ax2.set_ylim(-0.02, 0.48)
ax2.set_xticks([])
ax2.set_yticks([])
for side in ("top", "right", "left"):
    ax2.spines[side].set_visible(False)
ax2.spines["bottom"].set_linewidth(0.4)

arrow((84, 17.5), (22, 10.8), color=GREEN, lw=0.7)   # prior -> zero-shot
arrow((92, 17.5), (72, 10.8), color=GREEN, lw=0.7)   # prior -> posterior

# ---------------- Band C: deployment ----------------
band(0.5, 13, BAND_C, "Deployment\n(new event)")
box(5, 2.2, 30, 10.8, fc="#e4edf6", ec=BLUE, lw=0.8,
    text="Zero-shot (hour 0)\n$p = \\mathrm{E}_b[\\sigma(m+b)]$\n+ 90% interval of\nlandslide total")
box(32.6, 2.2, 53.6, 10.8,
    text="Rapid mapping (days 1–3)\nk image tiles chosen where\nlandslides are visible")
box(56.2, 2.2, 82.2, 10.8, fc="#e4f1de", ec=GREEN, lw=0.8,
    text="Selection-aware posterior\nπ(b | C) ∝ L(C | b)\n/ C(N₊(b), k) · N(b; μ, τ²)")
box(84.8, 2.2, 99, 10.8, fc="#fbe9d6", ec=ORANGE, lw=0.8,
    text="Updated forecast\n+ 90% interval")

arrow((30, 6.5), (32.6, 6.5))      # zero-shot -> rapid mapping
arrow((53.6, 6.5), (56.2, 6.5))    # rapid -> posterior
arrow((82.2, 6.5), (84.8, 6.5))    # posterior -> updated
arrow((12, 17.5), (12, 10.8))      # training -> zero-shot
ax.text(13.5, 14.3, "scores m = φ + g", ha="left", va="center", fontsize=6.3,
        style="italic", color=TXT, zorder=5)

fig.savefig(os.path.join(FIG_DIR, "fig0_workflow.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig0_workflow.png"), dpi=300)
print("wrote", os.path.join(FIG_DIR, "fig0_workflow.pdf"))
