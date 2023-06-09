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
# 1d FM chain, spin along z
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
    fm_chain = Sample((a_eff, b_eff, c_eff), tau, n, te, gamma_fnc=gamma_fnc)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 2

    # atom postions with effective lattice parameters
    atoms = [
        Atoms(
            t=(0, 0, 0),
            ion="Tb3",
            g=1.5,
            spin=s1,
            aniso=[[0, 0, 0], [0, 0, 0], [0, 0, -0.1]],
        ),
    ]
    fm_chain.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = -1  # AFM is positive, FM is negative
    bonds = [
        Bonds(fm_chain, 0, 0, r1=(1, 0, 0), j=j1),
        Bonds(fm_chain, 0, 0, r1=(0, 1, 0), j=j1),
    ]
    fm_chain.add_bonds(bonds)

    # -------------------------------------------------------------
    # Simulate dispersion along (1,0,0)
    # -------------------------------------------------------------
    proj_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes = ["(H,0,0)", "(0,K,0)", "(0,0,L)"]
    qe_range = (
        [0, 3.01, 0.01],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.01],
    )
    sim_qespace = LSWT(qe_range, fm_chain, proj_axes=proj_axes, axes=axes)
    # calculate dispersion relation
    sim_qespace.dispersion_calc()
    sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    sim_qespace.inten_calc()

    slice_range = (
        [0, 3.01, 0.01],
        [0.00, 0.01],
        [0.00, 0.01],
        [-20, 20.01, 0.01],
    )

    sim_qespace.slice(slice_range, plot_axes=(0, 3), vmin=0, vmax=2)

    # -------------------------------------------------------------
    # Simulate dispersion along (110)
    # -------------------------------------------------------------
    proj_axes2 = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    axes2 = ["(H,H,0)", "(-K,K,0)", "(0,0,L)"]
    qe_range2 = (
        [0, 3.01, 0.01],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.01],
    )
    sim_qespace2 = LSWT(qe_range2, fm_chain, proj_axes=proj_axes2, axes=axes2)
    # calculate dispersion relation
    sim_qespace2.dispersion_calc()
    sim_qespace2.plot_disp("x")

    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    sim_qespace2.inten_calc()

    slice_range2 = (
        [0, 3.01, 0.01],
        [0.00, 0.01],
        [0.00, 0.01],
        [-20, 20.01, 0.01],
    )

    sim_qespace2.slice(slice_range2, plot_axes=(0, 3), vmin=0, vmax=2)

    # -------------------------------------------------------------
    # Making cuts
    # -------------------------------------------------------------
    cut_range = (
        [0.5, 0.51],
        [0.00, 0.01],
        [0.00, 0.01],
        [-20, 20.01, 0.01],
    )
    sim_qespace.cut(cut_range, plot_axis=3, SIM=True)
    cut_range = (
        [1.0, 1.01],
        [0.00, 0.01],
        [0.00, 0.01],
        [-20, 20.01, 0.01],
    )
    sim_qespace.cut(cut_range, plot_axis=3, SIM=True)

    plt.show()
