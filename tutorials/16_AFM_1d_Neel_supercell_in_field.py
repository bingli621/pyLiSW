# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent) + "/pyLiSW")
from Atoms import Atoms
from Bonds import Bonds
from Sample import Sample
from LSWT import LSWT
from utils import gamma_fnc
from matplotlib import pyplot as plt
import numpy as np

# -------------------------------------------------------------
# 1d AFM chain, Neel type, spin along z
# -------------------------------------------------------------
if __name__ == "__main__":
    # lattice parameters in Angstrom
    a = 3
    # determin the effective lattice parameters
    a_eff = 2 * a
    b_eff = a
    c_eff = a
    # propagation vector
    tau = (0, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (0, 1, 0)
    # temperature
    te = 2

    s = 5 / 2
    dz = 0.05
    g = 2
    j1 = 1  # AFM is positive, FM is negative
    eng_AF = -2 * j1 * s**2 - 2 * dz * s**2

    aniso = [[0, 0, 0], [0, 0, 0], [0, 0, -dz]]

    n_row = 4
    n_col = 4

    fig, plot_axes = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(10, 8),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
        },
    )

    mag_fields = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110]
    theta_list = []
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_row + j
            mag_field = mag_fields[idx]
            afm_chain = Sample(
                (a_eff, b_eff, c_eff),
                tau,
                n,
                te,
                mag=[0, 0, -mag_field],
                gamma_fnc=gamma_fnc,
            )

            # ----------------- calculate classical ground state energy -----------------

            acos = g * LSWT.mu_B * mag_field / (2 * s * (2 * j1 - dz))
            if acos > 1:
                theta0 = 0
            else:
                theta0 = np.arccos(acos)

            eng_SF = (
                +2 * j1 * s**2 * np.cos(2 * theta0)
                - 2 * dz * s**2 * np.cos(theta0) ** 2
                - 2 * g * LSWT.mu_B * mag_field * s * np.cos(theta0)
            )

            # ------------------------- Add atoms------------------------------
            # atom postions with effective lattice parameters
            if eng_AF < eng_SF:  # AF
                theta_list.append(90)
                print(
                    "Antiferromagnetic, B={} T, theta={}".format(
                        mag_field, np.round(theta0 / np.pi * 180, 1)
                    )
                )
                atoms = [
                    Atoms(t=(0, 0, 0), ion="Mn2", spin=s, aniso=aniso),
                    Atoms(
                        t=(0.5, 0, 0),
                        ion="Mn2",
                        spin=s,
                        aniso=aniso,
                        theta=np.pi,
                        n_Rp=(0, 1, 0),
                    ),
                ]
            else:  # SF
                theta_list.append(theta0 * 180 / np.pi)
                print(
                    "In spin-flop state, B={} T, theta={}".format(
                        mag_field, np.round(theta0 / np.pi * 180, 1)
                    )
                )
                atoms = [
                    Atoms(
                        t=(0, 0, 0),
                        ion="Mn2",
                        spin=s,
                        aniso=aniso,
                        theta=-theta0,
                        n_Rp=(0, 1, 0),
                    ),
                    Atoms(
                        t=(0.5, 0, 0),
                        ion="Mn2",
                        spin=s,
                        aniso=aniso,
                        theta=theta0,
                        n_Rp=(0, 1, 0),
                    ),
                ]

            afm_chain.add_atoms(atoms)
            # ----------------------- Add bonds -------------------------------

            bonds = [
                Bonds(afm_chain, 0, 1, j=j1),
                Bonds(afm_chain, 0, 1, r1=(-1, 0, 0), j=j1),
            ]
            afm_chain.add_bonds(bonds)
            # -------------------------------------------------------------
            # Simulate dispersion
            # -------------------------------------------------------------
            proj = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            axes = ["(H,0,0)", "(0,K,0)", "(0,0,L)"]
            qe_range = (
                [0, 3.02, 0.02],
                [0.00, 0.01, 0.01],
                [0.00, 0.01, 0.01],
                [-30.005, 30.005, 0.01],
            )
            sim_qespace = LSWT(qe_range, afm_chain, proj_axes=proj, axes=axes)
            # sim_qespace.dispersion_calc()
            # sim_qespace.plot_disp("x")

            # -------------------------------------------------------------
            # Simulate intensities
            # -------------------------------------------------------------
            slice_range = (
                [0, 2.02, 0.02],
                [0.00, 0.01],
                [0.00, 0.01],
                [-0.005, 15.01, 0.01],
            )
            sim_qespace.inten_calc(mask=None)
            x, y, sim, xlab, ylab, tit = sim_qespace.slice(
                slice_range, plot_axes=(0, 3), PLOT=False, vmin=0, vmax=20
            )
            # -------------------------------------------------------------
            # Making figure
            # -------------------------------------------------------------

            ax = plot_axes[i, j]
            im = ax.pcolormesh(x / 2, y, sim, cmap="jet", vmin=0, vmax=20)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 5, 10])
            ax.grid(alpha=0.6)
            props = dict(facecolor="white")
            ax.text(
                0.05,
                0.95,
                "{} T".format(mag_field),
                transform=ax.transAxes,
                fontsize=16,
                va="top",
                bbox=props,
            )

    print(theta_list)

    # plt.setp(ax1.get_yticklabels(), visible=False)
    for i in range(n_row):
        plot_axes[i, 0].set_ylabel(ylab)
    for j in range(n_col):
        plot_axes[-1, j].set_xlabel(xlab)
        plot_axes[-1, j].set_xticklabels(["0", "0.5", "1"])

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
