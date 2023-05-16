# -*- coding: utf-8 -*-
from Atoms import Atoms, Bonds
from Atoms.Sample import Sample
from LSWT import LSWT
from matplotlib import pyplot as plt
from utils import gamma_fnc

# -------------------------------------------------------------
# 1d AFM chain, Neel type, spin along z
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
    tau = (0.5, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (0, 1, 0)
    # temperature
    te = 20
    afm_chain = Sample((a_eff, b_eff, c_eff), tau, n, axes, te, gamma_fnc=gamma_fnc)
    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 5 / 2
    aniso = [[0, 0, 0], [0, 0, 0], [0, 0, -0.1]]
    # atom postions with effective lattice parameters
    atoms = [
        Atoms(t=(0, 0, 0), ion="Mn2", spin=s1, aniso=aniso),
    ]
    afm_chain.add_atoms(atoms)
    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = 1  # AFM is positive, FM is negative
    bonds = [
        Bonds(afm_chain, 0, 0, r1=(1, 0, 0), j=j1),
    ]
    afm_chain.add_bonds(bonds)
    # -------------------------------------------------------------
    # Simulate dispersion
    # -------------------------------------------------------------
    qe_range = (
        [-0.0, 3.01, 0.01],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-20, 20, 0.01],
    )
    sim_qespace = LSWT(qe_range, afm_chain)
    # sim_qespace.dispersion_calc()
    # sim_qespace.plot_disp("x")
    # -------------------------------------------------------------
    # Simulate intensities
    # -------------------------------------------------------------
    slice_range = (
        [0, 3.01, 0.01],
        [0.00, 0.01],
        [0.00, 0.01],
        [-6, 6.01, 0.01],
    )
    sim_qespace.inten_calc()
    sim_qespace.slice(slice_range, plot_axes=(0, 3), SIM=True, vmin=0, vmax=10)
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
