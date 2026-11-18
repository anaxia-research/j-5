"""Illustrations for the paper (not used in any proof)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})

# Figure 2: jet-loss rates rho^gamma
fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200)
rho = np.linspace(0.25, 2.0, 400)
styles = {-2: (0, (1, 1.5)), -1: (0, (4, 2)), 0: "-", 1: (0, (6, 2, 1, 2)), 2: (0, (8, 3))}
for g, ls in styles.items():
    ax.plot(rho, rho**g, color="k", ls=ls, lw=1.6 if g == 0 else 1.2,
            label=r"$\gamma=%d$" % g + (" (Maxwell)" if g == 0 else ""))
for x, lab in [(0.62, "s"), (1.0, "r")]:
    ax.axvline(x, color="0.55", lw=0.8)
    ax.text(x + 0.02, 3.65, "$%s$" % lab, fontsize=12)
ax.set_ylim(0, 4); ax.set_xlim(0.25, 2.0)
ax.set_xlabel(r"jet speed $\rho$"); ax.set_ylabel(r"jet-loss rate $\propto\rho^\gamma$")
ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig("fig_rates.png", bbox_inches="tight", pad_inches=0.04)
plt.close()

# Figure 3: the certificate on the sphere of radius s
s, lam, eps = 0.62, 0.62, 0.05
def h(y):
    e = np.array([0, 0, 1.0])
    t1 = np.sum((y - s*e)**2, axis=-1); t2 = np.sum((y + s*e)**2, axis=-1)
    return (t1/(1 + t1))**4 * (t2/(1 + t2))**4
def grad_h(y, d=1e-6):
    return np.array([(h(y + d*np.eye(3)[i]) - h(y - d*np.eye(3)[i]))/(2*d) for i in range(3)])
def m_eps(z):
    # unique minimiser of |y-z|^2/lam + eps*h(y): fixed point of y = z - (eps*lam/2) grad h(y)
    y = z.copy()
    for _ in range(200):
        y = z - 0.5*eps*lam*grad_h(y)
    return np.sum((y - z)**2)/lam + eps*h(y)
fig = plt.figure(figsize=(9.2, 3.9), dpi=200)
ax = fig.add_subplot(1, 2, 1, projection="3d")
u, v = np.mgrid[0:2*np.pi:120j, 0:np.pi:80j]
X, Y, Z = s*np.cos(u)*np.sin(v), s*np.sin(u)*np.sin(v), s*np.cos(v)
H = h(np.stack([X, Y, Z], axis=-1))
ax.plot_surface(X, Y, Z, facecolors=cm.Greys(0.15 + 0.8*H/H.max()), linewidth=0, shade=False, rstride=1, cstride=1)
for sg in (1, -1):
    ax.scatter([0], [0], [sg*s], s=40, color="w", edgecolor="k", depthshade=False)
ax.text(0.05, 0, s + 0.12, "$+se$"); ax.text(0.05, 0, -s - 0.22, "$-se$")
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=12, azim=35)
ax.set_xlim(-0.65, 0.65); ax.set_ylim(-0.65, 0.65); ax.set_zlim(-0.65, 0.65)

ax2 = fig.add_subplot(1, 2, 2)
th = np.linspace(0, np.pi, 181)
mvals = [m_eps(s*np.array([np.sin(t), 0, np.cos(t)])) for t in th]
hv = eps*h(np.stack([s*np.sin(th), 0*th, s*np.cos(th)], axis=-1))
ax2.plot(np.degrees(th), 1e5*np.array(hv), color="k", lw=1.4, label=r"$10^5\,\varepsilon h_B(s\sigma)$  (perturbation of $\psi$)")
ax2.plot(np.degrees(th), 1e5*np.array(mvals), color="k", ls="--", lw=1.4, label=r"$10^5\,m_\varepsilon(s\sigma)$  (response of $\varphi$)")
hv_arr = np.array(hv); mv_arr = np.array(mvals)
gap = np.where(hv_arr > 1e-12, (hv_arr - mv_arr)/np.maximum(hv_arr, 1e-300), np.nan)
ax3 = ax2.twinx()
ax3.plot(np.degrees(th), 1e3*gap, color="0.55", lw=1.0, ls=":", label=r"$10^3\,(\varepsilon h_B-m_\varepsilon)/(\varepsilon h_B)$")
ax3.set_ylim(0, None); ax3.tick_params(axis="y", colors="0.4")
for sp in ("top",):
    ax3.spines[sp].set_visible(False)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
ax2.set_xlabel(r"polar angle of $\sigma$ (degrees)"); ax2.set_xlim(0, 180); ax2.set_xticks([0, 45, 90, 135, 180])
ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.27))
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig("fig_certificate.png", bbox_inches="tight", pad_inches=0.05)
