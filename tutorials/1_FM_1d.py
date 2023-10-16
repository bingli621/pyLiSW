# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent) + "/pyLiSW")
from Atoms import Atoms
from Bonds import Bonds
from Sample import Sample
from LSWT import LSWT
import matplotlib.pylab as plt
from utils import gamma_fnc_50

# -------------------------------------------------------------
# 1d FM chain, spin along z
# -------------------------------------------------------------
if __name__ == "__main__":
    axes = ["(H,0,0)", "(0,K,0)", "(0,0,L)"]
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
    fm_chain = Sample(
        (a_eff, b_eff, c_eff), tau, n, te, mag=[0, 0, -0], gamma_fnc=gamma_fnc_50
    )
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 2

    # atom postions with effective lattice parameters
    atoms = [
        Atoms(
            t=(0, 0, 0),
            ion="Mn2",
            spin=s1,
            aniso=[[0, 0, 0], [0, 0, 0], [0, 0, -0.01]],
        ),
    ]
    fm_chain.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = -1  # AFM is positive, FM is negative
    bonds = [
        Bonds(fm_chain, 0, 0, r1=(1, 0, 0), j=j1),
    ]
    fm_chain.add_bonds(bonds)

    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    qe_range = (
        [0, 3.01, 0.01],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.01],
    )
    sim_qespace = LSWT(qe_range, fm_chain)
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

    sim_qespace.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=2)
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
