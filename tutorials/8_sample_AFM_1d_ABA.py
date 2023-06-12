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
    afm_chain_ABA = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc)
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
            t=(0.5 + z, 0, 0), ion="Mn2", spin=s2, theta=np.pi, n=(0, 1, 0), aniso=aniso
        ),
        Atoms(
            t=(0.5 - z, 0, 0), ion="Mn2", spin=s2, theta=np.pi, n=(0, 1, 0), aniso=aniso
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
    sim_qespace.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=1)
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

    plt.show()
