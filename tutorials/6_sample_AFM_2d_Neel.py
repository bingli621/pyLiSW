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
# 2d AFM Neel on square lattice, spin along z
# -------------------------------------------------------------
if __name__ == "__main__":
    # lattice parameters in Angstrom
    a = 3
    # determin the effective lattice parameters
    a_eff = a
    b_eff = a
    c_eff = a

    # propagation vector
    tau = (1 / 2, 1 / 2, 0)
    # vector perpendicular to the plane of rotation
    n = (-1, 1, 0)
    # temperature
    te = 20

    afm_Neel = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 5 / 2
    aniso = [[0, 0, 0], [0, 0, 0], [0, 0, -0.1]]
    # atom postions with effective lattice parameters
    atoms = [
        Atoms(t=(0, 0, 0), ion="Mn2", spin=s1, aniso=aniso),
    ]
    afm_Neel.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = 1  # AFM is positive, FM is negative
    bonds = [
        Bonds(afm_Neel, 0, 0, r1=(1, 0, 0), j=j1),
        Bonds(afm_Neel, 0, 0, r1=(0, 1, 0), j=j1),
    ]
    afm_Neel.add_bonds(bonds)
    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    # High-symmetry directions
    proj = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes = ["(H,0,0)", "(0,K,0)", "(0,0,L)"]
    qe_range = (
        [-2, 2.01, 0.01],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.1],
    )
    sim_qespace = LSWT(qe_range, afm_Neel, proj_axes=proj, axes=axes)
    sim_qespace.dispersion_calc()
    sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # proj2 = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    # axes2 = ["(H,H,0)", "(-K,K,0)", "(0,0,L)"]
    # qe_range2 = (
    #     [-2, 2.01, 0.01],
    #     [-0.00, 0.01, 0.01],
    #     [-0.00, 0.01, 0.01],
    #     [-20, 20, 0.1],
    # )
    # sim_qespace2 = LSWT(qe_range2, afm_Neel, proj_axes=proj2, axes=axes2)
    # sim_qespace2.dispersion_calc()
    # sim_qespace2.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    sim_qespace.inten_calc()
    slice_range = (
        [-2, 4.01, 0.01],
        [-0.00, 0.01],
        [-0.00, 0.01],
        [-20, 20, 0.1],
    )
    sim_qespace.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=5)
    # -------------------------------------------------------------
    # sim_qespace2.inten_calc()
    # slice_range2 = (
    #     [-2, 4.01, 0.01],
    #     [-0.00, 0.01],
    #     [-0.00, 0.01],
    #     [-20, 20, 0.1],
    # )
    # sim_qespace2.slice(slice_range2, plot_axes=(0, 3), SIM=True, vmin=0, vmax=3)
    # -------------------------------------------------------------
    qe_range3 = (
        [-1, 1.01, 0.01],
        [-1, 1.01, 0.01],
        [-0.00, 0.01, 0.01],
        [-20, 20, 0.1],
    )
    sim_qespace3 = LSWT(qe_range3, afm_Neel)
    sim_qespace3.inten_calc()

    slice_range3 = (
        [-1, 1.01, 0.01],
        [-1, 1.01, 0.01],
        [-0.00, 0.01],
        [5, 6],
    )
    sim_qespace3.slice(slice_range3, plot_axes=(0, 1), SIM=True, vmin=0, vmax=5)

    plt.show()
