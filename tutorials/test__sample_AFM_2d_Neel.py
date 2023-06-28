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
    n_R = (0, 1, 0)
    # temperature
    te = 20

    afm_Neel = Sample((a_eff, b_eff, c_eff), tau, n_R, te, gamma_fnc=gamma_fnc)
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
    qe_range3 = (
        [-1, 1.01, 0.1],
        [-1, 1.01, 0.1],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.1],
    )
    sim_qespace3 = LSWT(qe_range3, afm_Neel)
    sim_qespace3.inten_calc()

    slice_range3 = (
        [-1, 1.01, 0.1],
        [-1, 1.01, 0.1],
        [-0.00, 0.01],
        [5, 6],
    )
    sim_qespace3.slice(slice_range3, plot_axes=(0, 1), vmin=0, vmax=5)

    plt.show()
