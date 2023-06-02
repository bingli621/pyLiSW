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
    a_eff = 2 * a
    b_eff = 2 * a
    c_eff = a

    # propagation vector
    tau = (0, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (0, 1, 0)
    # temperature
    te = 2

    afm_Neel = Sample((a_eff, b_eff, c_eff), tau, n, te)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 1
    aniso = [[0, 0, 0], [0, 0, 0], [0, 0, -0.01]]
    # atom postions with effective lattice parameters
    atoms = [
        Atoms(t=(0, 0, 0), ion="Mn2", spin=s1, aniso=aniso),
        Atoms(t=(1 / 2, 0, 0), ion="Mn2", spin=s1, theta=np.pi, n=n, aniso=aniso),
        Atoms(t=(0, 1 / 2, 0), ion="Mn2", spin=s1, theta=np.pi, n=n, aniso=aniso),
        Atoms(t=(1 / 2, 1 / 2, 0), ion="Mn2", spin=s1, aniso=aniso),
    ]
    afm_Neel.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = 1  # AFM is positive, FM is negative
    bonds = [
        # Bonds(self, 0, 1, j=j1),
        Bonds(afm_Neel, 0, 1, j=j1),
        Bonds(afm_Neel, 1, 0, r1=(1, 0, 0), j=j1),
        Bonds(afm_Neel, 0, 2, j=j1),
        Bonds(afm_Neel, 2, 0, r1=(0, 1, 0), j=j1),
        Bonds(afm_Neel, 1, 3, j=j1),
        Bonds(afm_Neel, 3, 1, r1=(0, 1, 0), j=j1),
        Bonds(afm_Neel, 2, 3, j=j1),
        Bonds(afm_Neel, 3, 2, r1=(1, 0, 0), j=j1),
    ]
    afm_Neel.add_bonds(bonds)
    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    # High-symmetry directions

    proj_axes = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    axes = ["(H,H,0)", "(-K,K,0)", "(0,0,L)"]

    qe_range_HH = (
        [0, 2.01, 0.01],
        [-0.02, 0.03, 0.01],
        [-0.02, 0.03, 0.01],
        [-20, 20, 0.1],
    )
    sim_qespace_HH = LSWT(qe_range_HH, afm_Neel, proj_axes, axes)
    sim_qespace_HH.dispersion_calc()
    sim_qespace_HH.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    sim_qespace_HH.inten_calc()
    slice_range_HH = (
        [0, 2.01, 0.01],
        [-0.02, 0.03],
        [-0.02, 0.03],
        [-6, 6, 0.1],
    )
    sim_qespace_HH.slice(slice_range_HH, plot_axes=(0, 3), SIM=True, vmin=0, vmax=5)
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    # High-symmetry directions
    proj_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes = ["(H,0,0)", "(0,K,0)", "(0,0,L)"]
    qe_range = (
        [0, 2.01, 0.01],
        [1 - 0.00, 1 + 0.01, 0.01],
        [-0.00, 0.01, 0.01],
        [-20, 20, 0.1],
    )
    sim_qespace = LSWT(qe_range, afm_Neel, proj_axes, axes)
    sim_qespace.dispersion_calc()
    sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    sim_qespace.inten_calc()
    slice_range = (
        [0, 2.01, 0.01],
        [1 - 0.00, 1 + 0.01],
        [-0.00, 0.01],
        [-6, 6, 0.1],
    )
    sim_qespace.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=5)
    # -------------------------------------------------------------

    # # -------------------------------------------------------------
    # # Simulate dispersion
    # # -------------------------------------------------------------
    # qe_range2 = (
    #     [1, 1.005, 0.005],
    #     [0, 2, 0.01],
    #     [-0.00, 0.01, 0.01],
    #     [-20, 20, 0.1],
    # )
    # sim_qespace2 = LSWT(qe_range2, afm_Neel)
    # # sim_qespace2.dispersion_calc()
    # # sim_qespace2.plot_disp("y")
    # # -------------------------------------------------------------
    # # Simulate intensities
    # # -------------------------------------------------------------
    # sim_qespace2.inten_calc()
    # slice_range2 = (
    #     [1, 1.005],
    #     [0, 2, 0.01],
    #     [-0.00, 0.01],
    #     [-6, 6, 0.1],
    # )
    # sim_qespace2.slice(slice_range2, plot_axes=(1, 3), SIM=True, vmin=0, vmax=20)
    # # -------------------------------------------------------------
    # # Simulate intensities of contour plots
    # # -------------------------------------------------------------
    # qe_range = (
    #     [-1, 1.01, 0.01],
    #     [-1, 1.01, 0.01],
    #     [-0.00, 0.01, 0.01],
    #     [-6, 6, 0.1],
    # )
    # sim_qespace = LSWT(qe_range, afm_Neel)
    # sim_qespace.inten_calc()

    # slice_range = (
    #     [-1, 1.01, 0.01],
    #     [-1, 1.01, 0.01],
    #     [-0.00, 0.01],
    #     [2, 3],
    # )
    # sim_qespace.slice(slice_range, plot_axes=(0, 1), SIM=True, vmin=0, vmax=1)

    plt.show()
