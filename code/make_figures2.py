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
s, lam, eps = 0.62, 0.62, 0.15
def h(y):
    e = np.array([0, 0, 1.0])
    d1 = np.linalg.norm(y - s*e, axis=-1); d2 = np.linalg.norm(y + s*e, axis=-1)
    return np.minimum(np.minimum(1.0, d1), d2)**8
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
ys = np.linspace(-1.6, 1.6, 321)
YY, ZZ = np.meshgrid(ys, ys)
P = np.stack([YY, np.zeros_like(YY), ZZ], axis=-1)
HP = eps*h(P)
mvals = []
for t in th:
    z = s*np.array([np.sin(t), 0, np.cos(t)])
    mvals.append(np.min((1/lam)*((YY - z[0])**2 + (ZZ - z[2])**2) + HP))
hv = eps*h(np.stack([s*np.sin(th), 0*th, s*np.cos(th)], axis=-1))
ax2.plot(np.degrees(th), hv, color="k", lw=1.4, label=r"$\varepsilon h(s\sigma)$  (perturbation of $\psi$)")
ax2.plot(np.degrees(th), mvals, color="k", ls="--", lw=1.4, label=r"$m_\varepsilon(s\sigma)$  (response of $\varphi$)")
ax2.set_xlabel(r"polar angle of $\sigma$ (degrees)"); ax2.set_xlim(0, 180); ax2.set_xticks([0, 45, 90, 135, 180])
ax2.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.18))
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig("fig_certificate.png", bbox_inches="tight", pad_inches=0.05)
