# Method-overview figure for paper/TVT/manuscript.tex (fig:method_overview).
# Sequential pipeline: received frame -> C1 shared-kernel spectral denoising
# (money shot, with spectral-gain mini-viz + 838x badge) -> unchanged AMC
# classifier; C2 domain-alignment kernel selection as an offline branch that
# configures C1's kernel. Palette: scientific-flowchart 'tech' (slate blue).
# Output: paper/TVT/figure/method_overview.pdf (vector, used by LaTeX) + .png preview.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Arc, Circle

# palette: bright blue-violet, sampled from paper/TVT/figure/slide_05.png
# (#2B5EFF classify-box, #DCE4FF/#E9EEFF periwinkle fills, #5E2592 dark violet,
#  #E2D0F2/#F1E8F9 lavender section) -- online path = blue family, offline C2 =
# lavender family, deep violet pills, royal-blue accent.
DARK     = '#5E2592'   # title pills
STROKE   = '#7D8BC4'   # online box borders, thin arrows
FILL     = '#DCE4FF'   # online module fill
CFILL    = '#F0F3FF'   # online container fill
TEXT     = '#1F2937'
ACCENT   = '#2B5EFF'   # main-flow arrows + speedup badge only
DATA     = '#2447C9'   # data lines (denoised trace, high-SNR gain curve)
GRAY     = '#9AA3B8'   # noisy trace / crossed-out naive path
C2FILL   = '#F4EDFA'   # offline container fill (lavender)
C2MOD    = '#E6D6F4'   # offline module fill
C2STROKE = '#A083C9'   # offline borders / arrows

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Cambria', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'text.color': TEXT,
    'pdf.fonttype': 42,    # TrueType 嵌入; matplotlib 默认 Type3 会被 IEEE PDF eXpress 拒
})

W, H = 1000, 440
fig = plt.figure(figsize=(7.16, 7.16 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

def rbox(x, y, w, h, fc=FILL, ec=STROKE, lw=1.2, ls='-', r=7, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=z))

def pill(cx, ycen, w, label, fs=7.5, tdx=0):
    h = 24
    rbox(cx - w / 2, ycen - h / 2, w, h, fc=DARK, ec='none', r=11, z=5)
    ax.text(cx + tdx, ycen, label, ha='center', va='center', color='white',
            fontsize=fs, fontweight='bold', zorder=6)

# --- small monochrome outline icons (stroke style matches box borders) ---
def icon_radio(cx, cy, s=7, color='white', lw=1.0):
    ax.add_patch(Circle((cx, cy), s * 0.16, fc=color, ec='none', zorder=7))
    for r in (s * 0.5, s * 0.85):
        ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=-40, theta2=40,
                         ec=color, lw=lw, zorder=7))
        ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=140, theta2=220,
                         ec=color, lw=lw, zorder=7))

def icon_nn(cx, cy, s=7, color='white', bg=DARK, lw=0.9):
    L = [(cx - s * 0.62, cy + s * 0.55), (cx - s * 0.62, cy),
         (cx - s * 0.62, cy - s * 0.55)]
    R = [(cx + s * 0.62, cy + s * 0.3), (cx + s * 0.62, cy - s * 0.3)]
    for p in L:
        for q in R:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw * 0.7, zorder=7)
    for p in L + R:
        ax.add_patch(Circle(p, s * 0.18, fc=bg, ec=color, lw=lw, zorder=8))

def icon_target(cx, cy, s=6.5, color=STROKE, lw=1.0):
    ax.add_patch(Circle((cx, cy), s * 0.95, fc='none', ec=color, lw=lw, zorder=7))
    ax.add_patch(Circle((cx, cy), s * 0.5, fc='none', ec=color, lw=lw, zorder=7))
    ax.add_patch(Circle((cx, cy), s * 0.14, fc=color, ec='none', zorder=7))

def icon_sliders(cx, cy, s=6.5, color=STROKE, lw=1.0):
    ys = (cy + s * 0.6, cy, cy - s * 0.6)
    xk = (cx - s * 0.35, cx + s * 0.45, cx - s * 0.1)
    for y, xkn in zip(ys, xk):
        ax.plot([cx - s, cx + s], [y, y], color=color, lw=lw,
                solid_capstyle='round', zorder=7)
        ax.add_patch(Circle((xkn, y), s * 0.2, fc='white', ec=color, lw=lw, zorder=8))

def icon_zap(cx, cy, s=7, color='white'):
    pts = [(0.25, 1.0), (-0.5, -0.08), (-0.05, -0.08),
           (-0.25, -1.0), (0.5, 0.08), (0.05, 0.08)]
    ax.add_patch(Polygon([(cx + px * s * 0.7, cy + py * s * 0.7) for px, py in pts],
                 closed=True, fc=color, ec='none', zorder=7))

def fat_arrow(x0, x1, y, half_tail=8, half_head=15, head_len=16):
    xh = x1 - head_len
    ax.add_patch(Polygon([(x0, y - half_tail), (xh, y - half_tail),
                          (xh, y - half_head), (x1, y),
                          (xh, y + half_head), (xh, y + half_tail),
                          (x0, y + half_tail)],
                 closed=True, fc=ACCENT, ec='none', zorder=4))

def thin_arrow(p0, p1, ls='-', lw=1.1, color=STROKE):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=9,
                 color=color, lw=lw, linestyle=ls, zorder=4,
                 shrinkA=0, shrinkB=0))

def inset(x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(STROKE); s.set_linewidth(0.6)
    a.set_facecolor('white')
    return a

# ---------------- synthetic data for thumbnails (illustrative) ----------------
rng = np.random.default_rng(7)
n = 128
t = np.arange(n)
clean = np.sin(2 * np.pi * 4 * t / n) + 0.5 * np.sin(2 * np.pi * 9 * t / n + 1.0)
noisy = clean + rng.normal(0, 0.7, n)

L = 5.0
K = np.exp(-(t[:, None] - t[None, :]) ** 2 / (2 * L ** 2))
lam, Q = np.linalg.eigh(K)
lam = lam[::-1]; Q = Q[:, ::-1]
lam = np.clip(lam, 0, None)
s2 = 0.49
den = Q @ ((lam / (lam + s2)) * (Q.T @ noisy))

# ================================ TOP ROW ====================================
# ---- input container ----
rbox(12, 195, 140, 215, fc=CFILL, ec=STROKE, ls='--', r=9, z=1)
pill(82, 410, 140, 'Received Frames', fs=7.0, tdx=8)
icon_radio(27, 410, s=7)
a = inset(26, 330, 112, 58)
a.plot(t, noisy, color=GRAY, lw=0.6)
ax.text(82, 308, r'$r[k]=s[k]+w[k]$', ha='center', va='center', fontsize=6.8)
ax.text(82, 290, r'$M$ frames, $n=128$', ha='center', va='center', fontsize=6.8)
ax.text(82, 258, 'coarse $\\widehat{\\mathrm{SNR}}$ $\\rightarrow$',
        ha='center', va='center', fontsize=6.8)
ax.text(82, 238, r'noise prior $\sigma_n^2$ (Eq. 2)',
        ha='center', va='center', fontsize=6.8)

# ---- main arrow input -> C1 ----
fat_arrow(154, 196, 300)
ax.text(175, 272, r'$Y,\ \{\sigma_n^2\}$', ha='center', va='center', fontsize=7)

# ---- C1 container (money shot) ----
rbox(198, 195, 500, 215, fc=CFILL, ec=DARK, lw=1.6, ls='--', r=9, z=1)
pill(448, 410, 330, 'C1 · Shared-Kernel Spectral Denoising', fs=7.5)

# internal chain: eigendecompose -> project -> gain -> back-project
bx_y, bx_h = 318, 70
rbox(210, bx_y, 122, bx_h)
ax.text(271, bx_y + 52, 'Eigendecompose once', ha='center', va='center',
        fontsize=6.6, fontweight='bold')
ax.text(271, bx_y + 33, r'$K=Q\Lambda Q^{\top}$', ha='center', va='center', fontsize=7.2)
ax.text(271, bx_y + 14, 'once per SNR group', ha='center', va='center', fontsize=6.0)

rbox(348, bx_y, 92, bx_h)
ax.text(394, bx_y + 48, 'Project', ha='center', va='center',
        fontsize=6.6, fontweight='bold')
ax.text(394, bx_y + 25, r'$V=Q^{\top}Y$', ha='center', va='center', fontsize=7.2)

rbox(456, bx_y, 124, bx_h)
ax.text(518, bx_y + 52, 'Per-sample gain', ha='center', va='center',
        fontsize=6.6, fontweight='bold')
ax.text(518, bx_y + 28, r'$G_{ij}=\frac{\lambda_i}{\lambda_i+\sigma_j^2}\,V_{ij}$',
        ha='center', va='center', fontsize=7.2)

rbox(596, bx_y, 92, bx_h)
ax.text(642, bx_y + 48, 'Back-project', ha='center', va='center',
        fontsize=6.6, fontweight='bold')
ax.text(642, bx_y + 25, r'$\widehat{Y}=Q\,G$', ha='center', va='center', fontsize=7.2)

thin_arrow((332, bx_y + bx_h / 2), (348, bx_y + bx_h / 2))
thin_arrow((440, bx_y + bx_h / 2), (456, bx_y + bx_h / 2))
thin_arrow((580, bx_y + bx_h / 2), (596, bx_y + bx_h / 2))

ax.text(448, 297, r'= per-sample Wiener filter in the kernel eigenbasis',
        ha='center', va='center', fontsize=6.8, style='italic')

# spectral-gain mini-viz (linear x, zoomed to the cutoff region)
g = inset(216, 210, 130, 72)
idx = np.arange(1, n + 1)
g.plot(idx, lam / (lam + 0.02), color=DATA, lw=1.0)
g.plot(idx, lam / (lam + 2.0), color=STROKE, lw=1.0, ls='--')
g.set_xlim(1, 40)
g.set_ylim(-0.05, 1.12)
g.set_xticks([]); g.set_yticks([])
g.text(17.0, 0.60, 'high SNR', fontsize=5.2, color=DATA)
g.text(3.0, 0.30, 'low SNR', fontsize=5.2, color=STROKE)
g.set_xlabel('eigenmode $i$', fontsize=5.4, labelpad=1)
g.set_ylabel('spectral gain', fontsize=5.4, labelpad=1)

# naive-vs-batched contrast + speedup badge
ax.text(520, 282,
        r'naive: $M$ separate solves $(K+\sigma_j^2 I)^{-1}y$ — $\mathcal{O}(M\,n^3)$',
        ha='center', va='center', fontsize=6.0, color=GRAY)
ax.plot([398, 642], [282, 282], color=GRAY, lw=0.8, zorder=5)
ax.text(530, 258,
        r'batched spectral path: $\mathcal{O}(G\,n^3+n^2 M)$',
        ha='center', va='center', fontsize=7.0)
rbox(452, 212, 156, 26, fc=ACCENT, ec='none', r=12, z=4)
icon_zap(466, 225, s=6.5)
ax.text(538, 225, '10–98$\\times$ measured', ha='center', va='center',
        fontsize=6.8, fontweight='bold', color='white', zorder=5)

# ---- main arrow C1 -> output ----
fat_arrow(700, 742, 300)
ax.text(721, 272, r'$\widehat{Y}$', ha='center', va='center', fontsize=7.5)

# ---- output container ----
rbox(744, 195, 244, 215, fc=CFILL, ec=STROKE, ls='--', r=9, z=1)
pill(866, 410, 152, 'Downstream AMC', fs=7.0, tdx=8)
icon_nn(804, 410, s=7)
a2 = inset(776, 330, 180, 58)
a2.plot(t, noisy, color=GRAY, lw=0.5)
a2.plot(t, den, color=DATA, lw=0.9)
ax.text(866, 305, 'denoised I/Q $\\rightarrow$ any AMC classifier',
        ha='center', va='center', fontsize=6.8)
ax.text(866, 287, '(six architectures, unchanged)',
        ha='center', va='center', fontsize=6.8)
thin_arrow((866, 276), (866, 252))
rbox(796, 218, 140, 32, fc=FILL, ec=STROKE)
ax.text(866, 234, 'modulation class', ha='center', va='center',
        fontsize=7.0, fontweight='bold')

# ============================== BOTTOM ROW ===================================
# ---- C2 container (offline, lavender family) ----
rbox(198, 25, 500, 130, fc=C2FILL, ec=C2STROKE, ls='--', r=9, z=1)
pill(448, 155, 400, 'C2 · Domain-Alignment Kernel Selection (offline)', fs=7.5)

c2_y, c2_h = 45, 72
rbox(210, c2_y, 142, c2_h, fc=C2MOD, ec=C2STROKE)
icon_nn(224, c2_y + 52, s=5.5, color=C2STROKE, bg=C2MOD)
ax.text(236, c2_y + 52, 'Reference classifier', ha='left', va='center',
        fontsize=6.6, fontweight='bold')
ax.text(281, c2_y + 33, r'$f_{\mathrm{ref}}$ trained on', ha='center', va='center', fontsize=6.6)
ax.text(281, c2_y + 15, '18 dB clean data', ha='center', va='center', fontsize=6.6)

rbox(368, c2_y, 184, c2_h, fc=C2MOD, ec=C2STROKE)
icon_target(382, c2_y + 52, s=5.5, color=C2STROKE)
ax.text(394, c2_y + 52, 'Task-aligned score (Eq. 3)', ha='left', va='center',
        fontsize=6.6, fontweight='bold')
ax.text(460, c2_y + 26,
        r'$\max\ \sum_i \log p_{f_{\mathrm{ref}}}(c_i\,|\,\mathbf{x}_i)$',
        ha='center', va='center', fontsize=7.0)

rbox(568, c2_y, 120, c2_h, fc=C2MOD, ec=C2STROKE)
icon_sliders(582, c2_y + 54, s=5.5, color=C2STROKE)
ax.text(594, c2_y + 54, 'Selected kernel', ha='left', va='center',
        fontsize=6.6, fontweight='bold')
ax.text(628, c2_y + 35, 'RBF,', ha='center', va='center', fontsize=6.6)
ax.text(628, c2_y + 15, r'$L=L_0(1+\beta|\mathrm{SNR}|)$',
        ha='center', va='center', fontsize=6.6)

thin_arrow((352, c2_y + c2_h / 2), (368, c2_y + c2_h / 2), color=C2STROKE)
thin_arrow((552, c2_y + c2_h / 2), (568, c2_y + c2_h / 2), color=C2STROKE)

# dashed arrow C2 -> C1 (configures kernel)
thin_arrow((628, 155 + 12), (628, 195), ls='--', lw=1.2, color=C2STROKE)
ax.text(642, 178, r'fixes $K$ and $L(\mathrm{SNR})$', ha='left', va='center',
        fontsize=6.6, style='italic')

import os
out_dir = os.path.join('paper', 'TVT', 'figure')
fig.savefig(os.path.join(out_dir, 'method_overview.pdf'))
fig.savefig(os.path.join(out_dir, 'method_overview.png'), dpi=350)
print('saved', os.path.join(out_dir, 'method_overview.pdf'))
