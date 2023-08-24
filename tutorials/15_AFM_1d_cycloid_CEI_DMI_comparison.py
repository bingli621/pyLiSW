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
        [-0.0, 3.01, 0.005],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-30, 30, 0.01],
    )
    sim_qespace_CEI = LSWT(qe_range, afm_chain_CEI)
    sim_qespace_DMI = LSWT(qe_range, afm_chain_DMI)
    # sim_qespace.dispersion_calc()
    # sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    slice_range = (
        [0, 3.01, 0.005],
        [0.00, 0.01],
        [0.00, 0.01],
        [-30, 30.01, 0.01],
    )
    sim_qespace_CEI.inten_calc()
    sim_qespace_CEI.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=5)
    sim_qespace_DMI.inten_calc()
    sim_qespace_DMI.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=5)

    plt.show()
