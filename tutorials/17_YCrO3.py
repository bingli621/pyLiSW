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

# Information about YCrO3 can be found in the following reference
# https://iopscience.iop.org/article/10.1088/1361-648X/abd781


if __name__ == "__main__":
    # lattice parameters in Angstrom
    # a = 3
    # # determin the effective lattice parameters
    lat_params = [5.5225, 7.5474, 5.2521]
    # propagation vector
    tau = (0, 0, 0)
    # vector perpendicular to the plane of rotation
    n = (1, 0, 0)
    # temperature
    te = 0
    ycro3 = Sample(lat_params, tau, n, te, gamma_fnc=gamma_fnc)

    # -------------------------------------------------------------
    # Add atoms
    # -------------------------------------------------------------
    s1 = 3 / 2
    aniso = [[0, 0, 0], [0, 0, 0], [0, 0, -0.01]]  # force moments along c
    # atom postions with effective lattice parameters
    atoms = [
        Atoms(
            t=(0.5, 0.5, 0.0),
            ion="Cr3",
            spin=s1,
            aniso=aniso,
            theta=0,
            n_Rp=(1, 0, 0),
        ),  # moments along positive c-axis
        Atoms(
            t=(0.0, 0.5, 0.5),
            ion="Cr3",
            spin=s1,
            aniso=aniso,
            theta=np.pi,
            n_Rp=(1, 0, 0),
        ),  # moments along negative c-axis, rotation needed
        Atoms(
            t=(0.0, 0.0, 0.5),
            ion="Cr3",
            spin=s1,
            aniso=aniso,
            theta=0,
            n_Rp=(1, 0, 0),
        ),  # moments along positive c-axis
        Atoms(
            t=(0.5, 0.0, 0.0),
            ion="Cr3",
            spin=s1,
            aniso=aniso,
            theta=np.pi,
            n_Rp=(1, 0, 0),
        ),  # moments along negative c-axis, rotation needed
    ]

    ycro3.add_atoms(atoms)

    # -------------------------------------------------------------
    # Add bonds
    # -------------------------------------------------------------
    j1 = 2.725  # meV  # AFM is positive, FM is negative
    # my_bonds = np.array([[1, 2], [0, 3], [1, 0], [2, 3]])
    j2 = 0.1  # meV
    # my_bonds = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [3, 1], [0, 2]])
    bonds = [
        # j1_1 bonds
        Bonds(ycro3, idx0=1, idx1=2, r0=(0, 0, 0), r1=(0, 0, 0), j=j1),
        Bonds(ycro3, idx0=1, idx1=2, r0=(0, 0, 0), r1=(0, 1, 0), j=j1),
        Bonds(ycro3, idx0=0, idx1=3, r0=(0, 0, 0), r1=(0, 0, 0), j=j1),
        Bonds(ycro3, idx0=0, idx1=3, r0=(0, 0, 0), r1=(0, 1, 0), j=j1),
        # j1_2 bonds
        Bonds(ycro3, idx0=1, idx1=0, r0=(0, 0, 0), r1=(0, 0, 0), j=j1),
        Bonds(ycro3, idx0=1, idx1=0, r0=(0, 0, 0), r1=(0, 0, 1), j=j1),
        Bonds(ycro3, idx0=0, idx1=1, r0=(0, 0, 0), r1=(1, 0, 0), j=j1),
        Bonds(ycro3, idx0=0, idx1=1, r0=(0, 0, 1), r1=(1, 0, 0), j=j1),
        Bonds(ycro3, idx0=2, idx1=3, r0=(0, 0, 0), r1=(0, 0, 0), j=j1),
        Bonds(ycro3, idx0=2, idx1=3, r0=(0, 0, 0), r1=(0, 0, 1), j=j1),
        Bonds(ycro3, idx0=2, idx1=3, r0=(1, 0, 0), r1=(0, 0, 0), j=j1),
        Bonds(ycro3, idx0=2, idx1=3, r0=(1, 0, 0), r1=(0, 0, 1), j=j1),
        # j2_1 bonds
        Bonds(ycro3, idx0=0, idx1=0, r0=(0, 0, 0), r1=(0, 0, 1), j=j2),
        Bonds(ycro3, idx0=0, idx1=0, r0=(0, 0, 0), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=1, idx1=1, r0=(0, 0, 0), r1=(0, 0, 1), j=j2),
        Bonds(ycro3, idx0=1, idx1=1, r0=(0, 0, 0), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=2, idx1=2, r0=(0, 0, 0), r1=(0, 0, 1), j=j2),
        Bonds(ycro3, idx0=2, idx1=2, r0=(0, 0, 0), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=3, r0=(0, 0, 0), r1=(0, 0, 1), j=j2),
        Bonds(ycro3, idx0=3, idx1=3, r0=(0, 0, 0), r1=(1, 0, 0), j=j2),
        # j2_2 bonds
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 0, 0), r1=(0, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 1, 0), r1=(0, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 0, 1), r1=(0, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 1, 1), r1=(0, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 0, 0), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 1, 0), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 0, 1), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=3, idx1=1, r0=(0, 1, 1), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 0), r1=(0, 0, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 1), r1=(0, 0, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 0), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 1), r1=(1, 0, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 0), r1=(0, 1, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 1), r1=(0, 1, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 0), r1=(1, 1, 0), j=j2),
        Bonds(ycro3, idx0=0, idx1=2, r0=(0, 0, 1), r1=(1, 1, 0), j=j2),
    ]
    ycro3.add_bonds(bonds)

    # -------------------------------------------------------------
    # Simulate dispersion along (H00)
    # -------------------------------------------------------------
    proj = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes = ["(H,0,0)", "(0,K,0)", "(0,0,L)"]
    qe_range = (
        [0.00, 2.01, 0.01],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-30, 30, 0.01],
    )
    sim_qespace = LSWT(qe_range, ycro3, proj_axes=proj, axes=axes)
    sim_qespace.dispersion_calc()
    sim_qespace.plot_disp("x")

    slice_range = (
        [0, 2.01, 0.01],
        [0.00, 0.01],
        [0.00, 0.01],
        [-30, 30.01, 0.01],
    )
    sim_qespace.inten_calc()
    sim_qespace.slice(slice_range, plot_axes=(0, 3), vmin=0, vmax=5)

    # -------------------------------------------------------------
    # Simulate dispersion along (HH0)
    # -------------------------------------------------------------
    proj = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    axes = ["(H,H,0)", "(-K,K,0)", "(0,0,L)"]
    qe_range_2 = (
        [0.00, 2.01, 0.01],
        [0.00, 0.01, 0.01],
        [0.00, 0.01, 0.01],
        [-30, 30, 0.01],
    )
    # Qlab  = {'[001]'  '\Gamma'   '[010]' '\Gamma' '[100]' };

    sim_qespace_2 = LSWT(qe_range_2, ycro3, proj_axes=proj, axes=axes)
    sim_qespace_2.dispersion_calc()
    sim_qespace_2.plot_disp("x")

    slice_range_2 = (
        [0, 2.01, 0.01],
        [0.00, 0.01],
        [0.00, 0.01],
        [-30, 30.01, 0.01],
    )
    sim_qespace_2.inten_calc()
    sim_qespace_2.slice(slice_range_2, plot_axes=(0, 3), vmin=0, vmax=5)

    plt.show()
