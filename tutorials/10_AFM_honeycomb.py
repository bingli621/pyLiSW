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
# Neel type AFM on a Honeycomb lattice, spin along +/- z
# -------------------------------------------------------------
if __name__ == "__main__":
    # lattice parameters in Angstrom
    a = 3
    # determin the effective lattice parameters
    a_eff = 3 * a
    b_eff = np.sqrt(3) * a
    c_eff = a
    # propagation vector
    tau = (0, 0, 0)
    # temperature
    te = 2
    afm_hc = Sample((a_eff, b_eff, c_eff), tau, te=te, gamma_fnc=gamma_fnc)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 1
    d = [[0, 0, 0], [0, 0, 0], [0, 0, -0.001]]

    # atom postions with effective lattice parameters
    atoms = [
        Atoms(t=(1 / 3, 0, 0), ion="Mn2", g=2, spin=s1, aniso=d),
        Atoms(t=(2 / 3, 0, 0), ion="Mn2", g=2, spin=s1, aniso=d, theta=np.pi),
        Atoms(t=(1 / 6, 1 / 2, 0), ion="Mn2", g=2, spin=s1, aniso=d, theta=np.pi),
        Atoms(t=(5 / 6, 1 / 2, 0), ion="Mn2", g=2, spin=s1, aniso=d),
    ]
    afm_hc.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = 1  # AFM is positive, FM is negative
    bonds = [
        Bonds(afm_hc, 0, 1, j=j1),
        Bonds(afm_hc, 0, 2, j=j1),
        Bonds(afm_hc, 1, 3, j=j1),
        Bonds(afm_hc, 2, 0, r1=(0, 1, 0), j=j1),
        Bonds(afm_hc, 3, 1, r1=(0, 1, 0), j=j1),
        Bonds(afm_hc, 3, 2, r1=(1, 0, 0), j=j1),
    ]
    afm_hc.add_bonds(bonds)

    # -------------------------------------------------------------
    # Simulate dispersion along (1,0,0)
    # -------------------------------------------------------------
    proj_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes = ["(H,H,0)", "(-K,K,0)", "(0,0,L)"]
    qe_range_110 = (
        [0, 4.01, 0.01],
        [0, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-5, 5, 0.01],
    )
    sim_qespace_110 = LSWT(qe_range_110, afm_hc, proj_axes=proj_axes, axes=axes)
    # calculate dispersion relation
    sim_qespace_110.dispersion_calc()
    sim_qespace_110.plot_disp("x")

    qe_range_m110 = (
        [2, 2.01, 0.01],
        [0, 4.01, 0.01],
        [0.00, 0.01, 0.01],
        [-5, 5, 0.01],
    )
    sim_qespace_m110 = LSWT(qe_range_m110, afm_hc, proj_axes=proj_axes, axes=axes)
    # calculate dispersion relation
    sim_qespace_m110.dispersion_calc()
    sim_qespace_m110.plot_disp("y")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    sim_qespace_110.inten_calc()

    slice_range_110 = (
        [0, 4.01, 0.01],
        [0.00, 0.01],
        [0.00, 0.01],
        [-5, 5.01, 0.01],
    )

    sim_qespace_110.slice(slice_range_110, plot_axes=(0, 3), vmin=0, vmax=5)

    sim_qespace_m110.inten_calc()
    slice_range_m110 = (
        [2, 2.01],
        [0, 4.01, 0.01],
        [0.00, 0.01],
        [-5, 5.01, 0.01],
    )

    sim_qespace_m110.slice(slice_range_m110, plot_axes=(1, 3), vmin=0, vmax=5)

    plt.show()
