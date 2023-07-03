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

# -------------------------------------------------------------
# 1d FM chain, with alternating J1-J2 interadtions, spin along z
# -------------------------------------------------------------
if __name__ == "__main__":
    # lattice parameters in Angstrom
    a = 3
    # determin the effective lattice parameters
    a_eff = a
    b_eff = a
    c_eff = a
    # propagation vector
    tau = (0, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (0, 1, 0)
    # temperature
    te = 2
    fm_j1j2_chain = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 5 / 2
    s2 = 5 / 2
    # atom postions with effective lattice parameters
    atoms = [
        Atoms(t=(0, 0, 0), ion="Mn2", spin=s1),
        Atoms(t=(0.5, 0, 0), ion="Mn2", spin=s2),
    ]
    fm_j1j2_chain.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = -1  # AFM is positive, FM is negative
    j2 = -0.75  # FM
    bonds = [
        Bonds(fm_j1j2_chain, 0, 1, j=j1),
        Bonds(fm_j1j2_chain, 1, 0, r1=(1, 0, 0), j=j2),
    ]
    fm_j1j2_chain.add_bonds(bonds)
    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    qe_range = (
        [0, 3.02, 0.02],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.01],
    )
    sim_qespace = LSWT(qe_range, fm_j1j2_chain)
    sim_qespace.dispersion_calc()
    sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    slice_range = (
        [0, 3.02, 0.02],
        [0.00, 0.01],
        [0.00, 0.01],
        [-15, 15.01, 0.01],
    )
    sim_qespace.inten_calc()
    sim_qespace.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=1)
    # -------------------------------------------------------------
    # Making cuts
    # -------------------------------------------------------------
    cut_range = (
        [0.5, 0.52],
        [0.00, 0.01],
        [0.00, 0.01],
        [0, 10.01, 0.01],
    )
    sim_qespace.cut(cut_range, plot_axis=3, SIM=True)

    plt.show()
