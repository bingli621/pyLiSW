# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent) + "/pyLiSW")
from Atoms import Atoms
from Bonds import Bonds
from Sample import Sample
from LSWT import LSWT
from utils import gamma_fnc
from matplotlib import pyplot as plt

# -------------------------------------------------------------
# 1d AFM chain, Neel type, spin along z
# -------------------------------------------------------------
if __name__ == "__main__":
    # lattice parameters in Angstrom
    a = 3
    # determin the effective lattice parameters
    a_eff = a
    b_eff = a
    c_eff = a
    # propagation vector
    tau = (0.2, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (0, 1, 0)
    # temperature
    te = 20
    afm_chain_CEI = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc)
    afm_chain_DMI = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 5 / 2
    aniso = [[0, 0, 0], [0, 0, 0], [0, 0, 0.0]]
    # atom postions with effective lattice parameters
    atoms = [
        Atoms(t=(0, 0, 0), ion="Mn2", spin=s1, aniso=aniso, theta=-np.pi * tau[0] * 3),
    ]
    afm_chain_DMI.add_atoms(atoms)
    afm_chain_CEI.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = -1  #  FM
    j2 = -j1 / 4 / np.cos(2 * np.pi * tau[0])
    dm = j1 * np.tan(2 * np.pi * tau[0])
    j_DMI = [[j1, 0, -dm], [0, j1, 0], [dm, 0, j1]]

    bonds_CEI = [
        Bonds(afm_chain_CEI, 0, 0, r1=(1, 0, 0), j=j1),
        Bonds(afm_chain_CEI, 0, 0, r1=(2, 0, 0), j=j2),
    ]
    afm_chain_CEI.add_bonds(bonds_CEI)

    bonds_DMI = [
        Bonds(afm_chain_DMI, 0, 0, r1=(1, 0, 0), j=j_DMI),
    ]
    afm_chain_DMI.add_bonds(bonds_DMI)

    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    qe_range = (
        [-0.00025, 1.5025, 0.005],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-30.005, 35.005, 0.01],
    )
    sim_qespace_CEI = LSWT(qe_range, afm_chain_CEI)
    sim_qespace_DMI = LSWT(qe_range, afm_chain_DMI)
    # sim_qespace.dispersion_calc()
    # sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    slice_range0 = (
        [-0.0025, 1.5025, 0.005],
        [0.00, 0.01],
        [0.00, 0.01],
        [-0, 15.005, 0.01],
    )
    sim_qespace_CEI.inten_calc()
    x0, y0, sim0, xlab0, ylab0, tit0 = sim_qespace_CEI.slice(
        slice_range0, plot_axes=(0, 3), PLOT=False, vmin=0, vmax=5
    )
    slice_range1 = (
        [-0.0025, 1.5025, 0.005],
        [0.00, 0.01],
        [0.00, 0.01],
        [-0, 35.005, 0.01],
    )
    sim_qespace_DMI.inten_calc()
    x1, y1, sim1, xlab1, ylab1, tit1 = sim_qespace_DMI.slice(
        slice_range1, plot_axes=(0, 3), PLOT=False, vmin=0, vmax=5
    )

    # -------------- making plots --------------------
    n_row = 2
    n_col = 1

    fig, plot_axes = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        sharex=True,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
            "height_ratios": [15, 35],
        },
    )

    ax = plot_axes[0]
    im = ax.pcolormesh(x0, y0, sim0, cmap="jet", vmin=0, vmax=5)
    # ax.set_xticks([0, 0.5, 1])
    # ax.set_yticks([0, 5, 10])
    ax.grid(alpha=0.6)
    props = dict(facecolor="white")
    ax.text(
        0.05,
        0.95,
        "CEI " + "$J_1=-1,J_2=0.81$",
        transform=ax.transAxes,
        fontsize=16,
        va="top",
        bbox=props,
    )
    plot_axes[0].set_ylabel(ylab0)
    # plot_axes[0, 0].set_xlabel(xla0b)
    # plot_axes[0, 0].set_xticklabels(["0", "0.5", "1"])

    ax = plot_axes[1]
    im = ax.pcolormesh(x1, y1, sim1, cmap="jet", vmin=0, vmax=5)
    # ax.set_xticks([0, 0.5, 1])
    # ax.set_yticks([0, 5, 10])
    ax.grid(alpha=0.6)
    props = dict(facecolor="white")
    ax.text(
        0.05,
        0.95,
        "DMI " + "$J_1=-1,D_y=3.08$",
        transform=ax.transAxes,
        fontsize=16,
        va="top",
        bbox=props,
    )
    plot_axes[1].set_ylabel(ylab1)
    plot_axes[1].set_xlabel(xlab1)
    # plot_axes[0, 0].set_xticklabels(["0", "0.5", "1"])

    gs = fig.add_gridspec(nrows=1, ncols=1, left=0.92, right=0.93, wspace=0.0)
    ax_cb = fig.add_subplot(gs[0, 0])
    ax_cb.set_title("max")
    ax_cb.text(0.0, -0.02, "0", transform=ax_cb.transAxes, va="top", fontsize="x-large")
    fig.colorbar(im, cax=ax_cb)
    ax_cb.tick_params(axis="both", which="both", length=0)
    ax_cb.set_yticks([])
    ax_cb.set_yticklabels([])
    ax_cb.set_xticks([])

    plt.tight_layout()

    plt.show()
