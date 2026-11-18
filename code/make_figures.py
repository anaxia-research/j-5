"""Illustrations for the paper (not used in any proof)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
OUT = "figures/"

# jet-loss rates
fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=220)
rho = np.linspace(0.25, 2.0, 400)
gs = [-2, -1, -0.5, 0, 0.5, 1, 2]
cmap = cm.coolwarm
for g in gs:
    col = "k" if g == 0 else cmap(0.5 + g/4.4)
    ax.plot(rho, rho**g, color=col, lw=2.4 if g == 0 else 1.6,
            label=(r"$\gamma=%g$" % g) + (" (Maxwell)" if g == 0 else ""))
ax.axvspan(0.62, 1.0, color="#fff3c4", zorder=0)
for x, lab in [(0.62, "s"), (1.0, "r")]:
    ax.axvline(x, color="0.45", lw=0.8)
    ax.text(x + 0.02, 3.65, "$%s$" % lab, fontsize=12)
ax.set_ylim(0, 4); ax.set_xlim(0.25, 2.0)
ax.set_xlabel(r"jet speed $\rho$"); ax.set_ylabel(r"jet-loss rate $\propto\rho^\gamma$")
ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(OUT + "fig_rates.png", bbox_inches="tight", pad_inches=0.04); plt.close()

# certificate
s, lam, eps = 0.62, 0.62, 0.05
def h(y):
    e = np.array([0, 0, 1.0])
    t1 = np.sum((y - s*e)**2, axis=-1); t2 = np.sum((y + s*e)**2, axis=-1)
    return (t1/(1 + t1))**4 * (t2/(1 + t2))**4
def grad_h(y, d=1e-6):
    return np.array([(h(y + d*np.eye(3)[i]) - h(y - d*np.eye(3)[i]))/(2*d) for i in range(3)])
def m_eps(z):
    y = z.copy()
    for _ in range(200): y = z - 0.5*eps*lam*grad_h(y)
    return np.sum((y - z)**2)/lam + eps*h(y)
fig = plt.figure(figsize=(9.4, 4.0), dpi=220)
ax = fig.add_subplot(1, 2, 1, projection="3d")
u, v = np.mgrid[0:2*np.pi:140j, 0:np.pi:90j]
X, Y, Z = s*np.cos(u)*np.sin(v), s*np.sin(u)*np.sin(v), s*np.cos(v)
H = h(np.stack([X, Y, Z], axis=-1))
ax.plot_surface(X, Y, Z, facecolors=cm.viridis(H/H.max()), linewidth=0, shade=False, rstride=1, cstride=1)
for sg in (1, -1):
    ax.scatter([0], [0], [sg*s], s=55, color="#e4572e", edgecolor="k", depthshade=False)
ax.text(0.05, 0, s + 0.12, "$+se$"); ax.text(0.05, 0, -s - 0.22, "$-se$")
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=12, azim=35)
ax.set_xlim(-0.65, 0.65); ax.set_ylim(-0.65, 0.65); ax.set_zlim(-0.65, 0.65)
mpb = cm.ScalarMappable(cmap="viridis"); mpb.set_array([0, H.max()])
cb = fig.colorbar(mpb, ax=ax, shrink=0.6, pad=0.0); cb.set_label(r"$h_B(s\sigma)$", fontsize=9); cb.ax.tick_params(labelsize=7)
ax2 = fig.add_subplot(1, 2, 2)
th = np.linspace(0, np.pi, 181)
mv = np.array([m_eps(s*np.array([np.sin(t), 0, np.cos(t)])) for t in th])
hv = eps*h(np.stack([s*np.sin(th), 0*th, s*np.cos(th)], axis=-1))
ax2.fill_between(np.degrees(th), 0, 1e5*hv, color="#9ad1d4", alpha=0.5)
ax2.plot(np.degrees(th), 1e5*hv, color="#1b6ca8", lw=1.8, label=r"$10^5\,\varepsilon h_B(s\sigma)$ (perturbation of $\psi$)")
ax2.plot(np.degrees(th), 1e5*mv, color="#e4572e", ls="--", lw=1.6, label=r"$10^5\,m_\varepsilon(s\sigma)$ (response of $\varphi$)")
gap = np.where(hv > 1e-12, (hv - mv)/np.maximum(hv, 1e-300), np.nan)
ax3 = ax2.twinx()
ax3.plot(np.degrees(th), 1e3*gap, color="#7b2cbf", lw=1.1, ls=":", label=r"$10^3\,(\varepsilon h_B-m_\varepsilon)/(\varepsilon h_B)$")
ax3.set_ylim(0, None); ax3.tick_params(axis="y", colors="#7b2cbf")
ax3.spines["top"].set_visible(False)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax2.set_xlabel(r"polar angle of $\sigma$ (degrees)"); ax2.set_xlim(0, 180); ax2.set_xticks([0, 45, 90, 135, 180])
ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.27))
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(OUT + "fig_certificate.png", bbox_inches="tight", pad_inches=0.05); plt.close()

# map of results over gamma
gE = 8 + 4*np.sqrt(110/3)
fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=220)
def bar(y, a, b, col, txt=None, tc="k", h=0.34):
    ax.add_patch(plt.Rectangle((a, y - h/2), b - a, h, color=col, lw=0))
    if txt: ax.text((a + b)/2, y, txt, ha="center", va="center", fontsize=8.2, color=tc)
yT, yE = 1.6, 0.6
bar(yT, -3, -0.12, "#f4a259"); bar(yT, 0.12, 46, "#f4a259", r"two solutions can move apart (Theorems A--D)")
ax.plot([0], [yT], marker="o", ms=8, color="#2a9d8f", mec="k", zorder=5)
ax.text(-0.3, yT + 0.3, "Maxwell: contraction", ha="right", fontsize=8, color="#1d6f65")
bar(yE, -3, 2, "#2a9d8f"); bar(yE, 2, 2*np.sqrt(3), "#7ccfc4"); bar(yE, 2*np.sqrt(3), np.sqrt(19), "#b8e3dd")
bar(yE, np.sqrt(19), gE, "#e6e6e6", "sign of $\\mathcal{S}$ unknown", tc="0.3")
bar(yE, gE, 46, "#e63946", "$W_2(f_t,M)$ can grow (E)", tc="w")
ax.text(-0.5, yE - 0.36, r"$\mathcal{S}\leq 0$ (F)", fontsize=8.2, ha="center", color="#1d6f65")
for x, lab in [(2, "$2$"), (2*np.sqrt(3), r"$2\sqrt3$"), (np.sqrt(19), r"$\sqrt{19}$"), (gE, r"$\gamma_E$")]:
    ax.plot([x, x], [yE - 0.2, yE + 0.2], color="k", lw=0.8)
ax.text(4.8, yE + 0.3, r"$2$: every $b$;  $2\sqrt{3}$: $b$ constant;  $\sqrt{19}$: Landau", fontsize=7.5, ha="left")
ax.axvline(1, color="#457b9d", lw=0.8, ls="--"); ax.text(1.3, 2.3, "hard spheres", fontsize=7.5, ha="left", color="#457b9d")
ax.axvline(-3, color="0.3", lw=0.8)
ax.set_xlim(-3.5, 46); ax.set_ylim(0, 2.6); ax.set_yticks([yE, yT]); ax.set_yticklabels(["distance to $M$", "two solutions"])
ax.set_xlabel(r"exponent $\gamma$ of the kinetic factor $|v-v_*|^\gamma$")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout(); plt.savefig(OUT + "fig_map.png", bbox_inches="tight", pad_inches=0.04); plt.close()

# q_min(gamma)
g = np.linspace(-3, 60, 800)
qmin = 6 - (9*g + 84)**2/(4*(3*g**2 + 69*g + 490))
fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=220)
ax.axhline(0, color="k", lw=0.8)
ax.fill_between(g, qmin, 0, where=qmin >= 0, color="#2a9d8f", alpha=0.35, label=r"$\min_\mu q>0$: $\mathcal{S}<0$ on $Y+\mu\Psi$")
ax.fill_between(g, qmin, 0, where=qmin < 0, color="#e63946", alpha=0.45, label=r"$\min_\mu q<0$: $\mathcal{S}(Y+\mu\Psi)>0$")
ax.plot(g, qmin, color="#264653", lw=1.8)
ax.axvline(gE, color="#e63946", ls="--", lw=1); ax.text(gE + 0.6, 1.3, r"$\gamma_E\approx32.22$", color="#b3202c", fontsize=10)
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\min_\mu q(\mu)$")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(OUT + "fig_qmin.png", bbox_inches="tight", pad_inches=0.04); plt.close()
