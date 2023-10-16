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
    num = 10  # size of supercell
    # lattice parameters in Angstrom
    a = 3 * num
    # determin the effective lattice parameters
    a_eff = a
    b_eff = a
    c_eff = a
    # propagation vector
    tau = (0, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (0, 1, 0)
    # temperature
    te = 20
    afm_chain = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 5 / 2

    aniso = [[0, 0, 0], [0, 0.0, 0], [0, 0, 0.0]]
    theta0 = -np.pi / 2 * 0

    # atom postions with effective lattice parameters
    atoms = [
        Atoms(
            t=(1 / num * i, 0, 0),
            ion="Mn2",
            spin=s1,
            aniso=aniso,
            n_Rp=(0, 1, 0),
            theta=theta0 + 2 * np.pi / num * i,
        )
        for i in range(num)
    ]
    afm_chain.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = -1  #  FM
    j2 = -j1 / 4 / np.cos(2 * np.pi / num)
    bonds = np.concatenate(
        [
            [Bonds(afm_chain, i, i + 1, j=j1) for i in range(num - 1)],
            [
                Bonds(afm_chain, num - 1, 0, r1=(1, 0, 0), j=j1),
            ],
            [Bonds(afm_chain, i, i + 2, j=j2) for i in range(num - 2)],
            [
                Bonds(afm_chain, num - 2, 0, r1=(1, 0, 0), j=j2),
                Bonds(afm_chain, num - 1, 1, r1=(1, 0, 0), j=j2),
            ],
        ]
    )
    afm_chain.add_bonds(bonds)
    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    qe_range = (
        [-0.025, 3 * num + 0.025, 0.05],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.01],
    )
    sim_qespace = LSWT(qe_range, afm_chain)
    sim_qespace.dispersion_calc()
    sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    slice_range = (
        [-0.025, 3 * num + 0.025, 0.05],
        [0.00, 0.01],
        [0.00, 0.01],
        [-12, 12.01, 0.01],
    )
    sim_qespace.inten_calc()
    sim_qespace.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=5)
    # -------------------------------------------------------------
    # Making cuts
    # -------------------------------------------------------------
    # cut_range = (
    #     [0.5, 0.52],
    #     [0.00, 0.01],
    #     [0.00, 0.01],
    #     [-0.5, 0.51, 0.01],
    # )
    # sim_qespace.cut(cut_range, plot_axis=3, SIM=True)

    plt.show()
