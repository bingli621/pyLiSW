# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent) + "/pyLiSW")
from Atoms import Atoms
from Bonds import Bonds
from Sample import Sample
from LSWT import LSWT
from utils import gamma_fnc_50
from matplotlib import pyplot as plt
import numpy as np


# -------------------------------------------------------------
# 1d AFM chain of three ions ABA, spin along z
# -------------------------------------------------------------
if __name__ == "__main__":
    # lattice parameters in Angstrom
    a = 3
    # determin the effective lattice parameters
    a_eff = a
    b_eff = a
    c_eff = a
    # propagation vector
    tau = (0.5, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (0, 1, 0)
    # temperature
    te = 20
    afm_chain_ABA = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc_50)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 5 / 2
    s2 = 1
    z = 0.2
    aniso = [[0, 0, 0], [0, 0, 0], [0, 0, -0.1]]
    # atom postions with effective lattice parameters
    atoms = [
        Atoms(t=(0.5, 0, 0), ion="Mn2", spin=s1, aniso=aniso),
        Atoms(
            t=(0.5 + z, 0, 0),
            ion="Mn2",
            spin=s2,
            theta=np.pi,
            n_Rp=(0, 1, 0),
            aniso=aniso,
        ),
        Atoms(
            t=(0.5 - z, 0, 0),
            ion="Mn2",
            spin=s2,
            theta=np.pi,
            n_Rp=(0, 1, 0),
            aniso=aniso,
        ),
    ]
    afm_chain_ABA.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    jMS = 5  # AFM
    jMM = 0.25  # AFM
    bonds = [
        Bonds(afm_chain_ABA, 0, 1, j=jMS),
        Bonds(afm_chain_ABA, 0, 2, r1=(0, 0, 0), j=jMS),
        Bonds(afm_chain_ABA, 0, 0, r1=(1, 0, 0), j=jMM),
    ]
    afm_chain_ABA.add_bonds(bonds)
    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    axes = ["(H,0,0)", "(0,K,0)", "(0,0,L)"]
    proj_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    qe_range = (
        [1e-3, 5.02, 0.02],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.01],
    )
    sim_qespace = LSWT(qe_range, afm_chain_ABA, proj_axes)
    sim_qespace.dispersion_calc()
    sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    slice_range = (
        [1e-3, 5.02, 0.02],
        [0.00, 0.01],
        [0.00, 0.01],
        [-15, 15.01, 0.01],
    )
    sim_qespace.inten_calc()
    sim_qespace.slice(slice_range, plot_axes=(0, 3), PLOT=True, vmin=0, vmax=0.75)
    # -------------------------------------------------------------
    # Making cuts
    # -------------------------------------------------------------
    # cut_range = (
    #     [0.5, 0.52],
    #     [0.00, 0.01],
    #     [0.00, 0.01],
    #     [-15, 15.01, 0.01],
    # )
    # sim_qespace.cut(cut_range, plot_axis=3, SIM=True)
    #
    # -------------- making plots --------------------
    n_row = 2
    n_col = 1

    fig, plot_axes = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        sharex=True,
        sharey=True,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
            "height_ratios": [1, 1],
        },
    )

    slice_range = (
        [1e-3, 4.02, 0.02],
        [0.00, 0.01],
        [0.00, 0.01],
        [0, 15.01, 0.01],
    )

    x, y, sim, xlab, ylab, tit = sim_qespace.slice(
        slice_range, plot_axes=(0, 3), PLOT=False
    )

    ax = plot_axes[0]
    for i in range(int(np.shape(sim_qespace.eng)[-1] / 2)):
        ax.plot(
            sim_qespace.xlist,
            sim_qespace.eng[:, 0, 0, i],
            label="{}".format(i),
            linewidth=3,
        )
    plot_axes[0].set_ylabel(ylab)
    ax.grid(alpha=0.6)
    ax.set_xlim([0, 4])
    ax.legend()

    ax = plot_axes[1]
    im = ax.pcolormesh(x, y, sim, cmap="jet", vmin=0, vmax=0.75)
    ax.grid(alpha=0.6)

    plot_axes[1].set_ylabel(ylab)
    plot_axes[1].set_xlabel(xlab)

    gs = fig.add_gridspec(nrows=1, ncols=1, left=0.92, right=0.93, wspace=0.0)
    ax_cb = fig.add_subplot(gs[0, 0])
    ax_cb.set_title("max")
    ax_cb.text(0.0, -0.02, "0", transform=ax_cb.transAxes, va="top", fontsize="x-large")
    fig.colorbar(im, cax=ax_cb)
    ax_cb.tick_params(axis="both", which="both", length=0)
    ax_cb.set_yticks([])
    ax_cb.set_yticklabels([])
    ax_cb.set_xticks([])

    plt.show()
