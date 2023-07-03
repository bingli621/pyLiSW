# -*- coding: utf-8 -*-
import numpy as np
from Bonds import Bonds
from tabulate import tabulate


class Sample(object):
    """
    Sample related parameters, including:
    -------------------------------------------------------------------------
    Attributes
    -------------------------------------------------------------------------
    a_eff, b_eff, c_eff     Effective lattice constats in Cartesian coordinate
    tau                     Propagation vector, in units of 2 pi/lat_param_eff
    te                      temperature, for Bose factor calculation
    n_R                     axis of rotation of R, for global rotation
    n_dim                   n_atoms, dimension of the Hamiltonian is 2 * n_dim
    atoms                   unique magnetic atoms
    bonds                   magnetic interaction J, 3 by 3 matrix
    mat_R1                  matrix R1, for intensity calculation
    mat_R2                  matrix R2, for intensity calculation
    -------------------------------------------------------------------------
    Methods
    -------------------------------------------------------------------------
    add_atoms               add unique magnetic ions
    add_bonds               add bonds J, scalar or 3x3 matrix
    sort_atoms              print atoms added
    list_bonds              print bonds added
    -------------------------------------------------------------------------
    """

    divider_str = "-" * 150
    divider_str2 = "=" * 150

    def __init__(
        self,
        lattice_parameters=(1, 1, 1),
        tau=(0, 0, 0),
        n_R=(0, 1, 0),
        te=2,
        gamma_fnc=None,
    ):
        (a, b, c) = lattice_parameters  # determine the effective lattice parameters
        self.a_eff = a
        self.b_eff = b
        self.c_eff = c

        self.tau = tau  # propagation vector
        self.n_R = n_R  # vector perpendicular to the plane of rotation
        self.n_R = self.n_R / np.linalg.norm(self.n_R)
        self.te = te  # temperature

        mat_nx = np.array(
            [
                [0, -self.n_R[2], self.n_R[1]],
                [self.n_R[2], 0, -self.n_R[0]],
                [-self.n_R[1], self.n_R[0], 0],
            ]
        )
        self.mat_R2 = np.outer(self.n_R, self.n_R)
        self.mat_R1 = (np.eye(3) - 1j * mat_nx - self.mat_R2) / 2

        self.atoms = None
        self.bonds = None
        self.dimers = None

        if gamma_fnc is None:
            self.gamma_fnc = (
                lambda x: 5 * np.nanmean(np.abs(np.diff(x))) * np.ones_like(x)
            )
        else:
            self.gamma_fnc = gamma_fnc

    @staticmethod
    def list_bonds(Bonds, latt_params=(1, 1, 1)):
        """
        Do Not modify
        """
        a, b, c = latt_params
        print(Sample.divider_str)
        print("Added {} bonds".format(np.shape(Bonds)[0]))
        headers = [
            "index",
            "atom0",
            "r0",
            "atom1",
            "r1",
            "dij",
            "j (not S*j, J<0 for FM)",
        ]
        to_print = []
        for idx, Bond in enumerate(Bonds):
            if np.array(Bond.j).ndim:
                j = np.array(Bond.j).tolist()
            else:
                j = Bond.j
            d = np.round(
                np.sqrt(
                    (Bond.dij[0] * a) ** 2
                    + (Bond.dij[1] * b) ** 2
                    + (Bond.dij[2] * c) ** 2,
                ),
                3,
            )
            idx_str = "{:<2}({:<2}->{:<2})".format(
                idx,
                *Bond.idx,
            )
            d_str = "{}={}".format(np.round(Bond.dij, 3), d)
            to_print.append(
                [
                    idx_str,
                    Bond.atom0.ion,
                    np.round(Bond.r0, 3),
                    Bond.atom1.ion,
                    np.round(Bond.r1, 3),
                    d_str,
                    np.round(j, 3).tolist(),
                ]
            )

        print(
            tabulate(
                to_print,
                headers=headers,
                floatfmt=".3f",
                numalign="right",
                stralign="right",
            )
        )

    def add_atoms(self, all_atoms):
        """
        Add unique atoms in the first magnetic unit cell,
        as many as the  number of magnon bands
        """
        # atoms = []
        # for atom in all_atoms:
        #     if atom.s:  # spin non-zero
        #         atoms.append(atom)
        # self.atoms = atoms
        self.atoms = all_atoms
        self.sort_atoms()
        self.n_dim = np.shape(self.atoms)[0]

    def add_bonds(self, all_bonds):
        """
        j<0 for FM
        value is j, not S*j
        add bonds in pairs of forward and backward hopping
        """
        # filter zero Js
        bonds = []
        for bond in all_bonds:
            if not np.any(bond.j):  # zero interaction strength
                print("Bond with zero strength ignored.")
                continue
            if not (bond.s0 and bond.s1):
                print("Bond coupled to spin of zero ignored.")
                continue
            bonds.append(bond)

        # add the reverse pair
        n_bonds = np.shape(bonds)[0]
        for i in range(n_bonds):
            idx1, idx0 = bonds[i].idx
            r1 = bonds[i].r0
            r0 = bonds[i].r1
            j = bonds[i].j
            bonds.append(Bonds(self, idx0, idx1, r0, r1, j))
        self.Bonds = bonds

        Sample.list_bonds(self.Bonds, (self.a_eff, self.b_eff, self.c_eff))

    def sort_atoms(self):
        """
        print out added atoms
        """
        atoms = self.atoms
        print(Sample.divider_str2)
        print(
            "Added {} unique magnetic ions in the 1st cell".format(np.shape(atoms)[0])
        )
        headers = [
            "index",
            "ion",
            "position t(in 1st cell)",
            "spin S' (not S or S'')",
            "single-ion anisotropy",
        ]
        to_print = []
        for idx, atom in enumerate(atoms):
            # self.atoms[atom.lat, atom.basis] = atom
            to_print.append(
                [
                    idx,
                    atom.ion,
                    atom.t,
                    atom.spin_p,
                    atom.aniso.tolist(),
                ]
            )

        print(
            tabulate(
                to_print,
                headers=headers,
                floatfmt=".3f",
                numalign="right",
                stralign="right",
            )
        )

    def mat_R(self, rn):
        """
        To check the rotation of spins
        """
        mat_R = (
            np.exp(2 * np.pi * 1j * np.dot(self.tau, rn)) * self.mat_R1
            + np.exp(-2 * np.pi * 1j * np.dot(self.tau, rn)) * np.conj(self.mat_R1)
            + self.mat_R2
        )
        if np.any(mat_R.imag):
            print("R is complex, check propagation vector.")
        else:
            print("R={}".format(np.round(mat_R.real, 3).tolist()))

    # @property
    # def mag_field(self):
    #     mag_field = 0
    #     return mag_field

    def add_dimers(self, all_dimers):
        """
        j<0 for FM
        value is j, not S*j
        add dimers in pairs
        """
        # filter zero Js
        dimers = []
        for dimer in all_dimers:
            if not np.any(dimer.j):  # zero interaction strength
                print("Dimer with zero coupling strength ignored.")
                continue
            if not (dimer.s0 and dimer.s1):
                print("Dimer coupled to spin of zero ignored.")
                continue
            dimers.append(dimer)
        self.dimers = dimers

        Sample.list_bonds(self.dimers, (self.a_eff, self.b_eff, self.c_eff))


# ----------------------------------- testing -----------------------------------
if __name__ == "__main__":
    sample = Sample()
    print("R1={}".format(np.round(sample.mat_R1, 3).tolist()))
    print("R2={}".format(np.round(sample.mat_R2, 3).tolist()))
    sample.mat_R([1, 0, 0])
