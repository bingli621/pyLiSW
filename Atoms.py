# -*- coding: utf-8 -*-
import numpy as np
from utils import rot_vec
import periodictable as pt


class Atoms(object):
    """
    Atoms class stores information about a given ion
    -------------------------------------------------------------------------
    Attributes
    -------------------------------------------------------------------------

    t = (tx, ty, tz)            ion position in the first unit cell
                                in units of effective lattice parameters
    ion                     type of magnetic ion, e.g. 'Mn2'
    g                           gyromagnetic ratio, g = 2 by default
    s                           absolute value of spin, scalar
    spin_p                      S' after global rotation R, 3x1 vector
    spin_pp                     S'' = [0,0,s] after local rotation R'', 3x1 vector
    ff_params                   parameters for magnetic form factors
    u                           after local rotation R_prime, 3x1 vector
    v                           after local rotation R_prime, 3x1 vector
    -------------------------------------------------------------------------
    See J. Phys.: Condens. Matter 27 (2015) 166002 for definition of terms
    in the Hamiltonian
    """

    # dictionary of magnetic form factors j0: A, a, B, b, C, c, D
    # ff_dic = {
    #     "Mn2": (0.422, 17.684, 0.5948, 6.005, 0.0043, -0.609, -0.0219),
    # }

    def __init__(
        self,
        t=None,
        ion=None,
        g=2,
        spin=1,
        theta=0,
        n=(0, 1, 0),
        aniso=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        test=None,
    ):
        """
        t is the basis vector in units of effective lattice parameters
        theta and phi corresponds to the Euler angles,
        """
        self.t = np.array(t)
        self.ion = ion
        self.g = g
        # magnetic form factor parameters
        valence = int(ion[-1])
        elem = getattr(pt, ion[:-1]).ion[valence]
        if g == 2:
            self.ff_params = elem.magnetic_ff[valence].j0
        else:
            self.ff_params = np.concatenate(
                [elem.magnetic_ff[valence].j0, elem.magnetic_ff[valence].j2]
            )

        self.s = np.linalg.norm(spin)
        self.spin_pp = np.array([0, 0, spin]).T

        rp_mat = rot_vec(theta, n)  # identity matrices by default
        self.u = np.round([rp_mat[:, 0] + 1j * rp_mat[:, 1]], 3).T
        self.v = np.round([rp_mat[:, 2]], 3).T
        self.spin_p = np.round(rp_mat @ self.spin_pp, 3).T
        # print((self.u))
        # print((self.v))
        self.aniso = np.array(aniso)

    @staticmethod
    def form_factor(k_mesh, ff_params, g=2):
        kx, ky, kz = k_mesh
        k_sq = np.square(kx) + np.square(ky) + np.square(kz)
        s_sq = k_sq / ((4 * np.pi) ** 2)

        if np.shape(ff_params)[0] == 7:  # j0 only
            A, a, B, b, C, c, D = ff_params
            j0 = (
                A * np.exp(-a * s_sq)
                + B * np.exp(-b * s_sq)
                + C * np.exp(-c * s_sq)
                + D
            )
            ff = j0

        elif np.shape(ff_params)[0] == 14:  # j2 as well
            A, a, B, b, C, c, D, A2, a2, B2, b2, C2, c2, D2 = ff_params
            j0 = (
                A * np.exp(-a * s_sq)
                + B * np.exp(-b * s_sq)
                + C * np.exp(-c * s_sq)
                + D
            )
            j2 = (
                A2 * np.exp(-a2 * s_sq)
                + B2 * np.exp(-b2 * s_sq)
                + C2 * np.exp(-c2 * s_sq)
                + D2
            ) * s_sq
            ff = j0 + (1 - 2 / g) * j2

        else:
            print("Unrecognized form factor parameters")

        # enforce zero ff for large q
        if np.any(ff < 0):
            idx_of_zero = np.unravel_index(np.argmin(ff**2, axis=None), ff.shape)
            ff[s_sq > s_sq[idx_of_zero]] = 0
        #     idx_of_zero = np.argmin(ff**2)
        #     ff[s_sq > s_sq[idx_of_zero]] = 0

        return ff


class Bonds(object):
    """
    -------------------------------------------------------------------------
    Attributes
    -------------------------------------------------------------------------
    idx                       index of (atom0,atom1)
    atom0                       first atom
    atom1                       second atom
    r0 = (0,0,0)
    r1 = (0,0,0)
    s0                          spin size of first atom
    s1                          spin size of second atom
    dij                         neighboring distance, in units of eff. latt. params.
    r_mn                        rotation matrix Rn-m
    j                           strength of magnetic interaction
                                can be a scalar or a 3 by 3 matrix
    -------------------------------------------------------------------------
    See J. Phys.: Condens. Matter 27 (2015) 166002 for definition of terms
    in the Hamiltonian
    """

    def __init__(self, Sample, idx0=None, idx1=None, r0=(0, 0, 0), r1=(0, 0, 0), j=1):
        self.idx = (idx0, idx1)
        self.atom0 = Sample.atoms[idx0]
        self.atom1 = Sample.atoms[idx1]
        self.r0 = np.array(r0)
        self.r1 = np.array(r1)
        self.s0 = self.atom0.s
        self.s1 = self.atom1.s
        self.dij = np.round(self.atom1.t - self.atom0.t + r1 - r0, 3)

        if np.array(j).ndim:
            self.j = j
        else:  # input j as a scalar
            self.j = j * np.eye(3)

        tau = 2 * np.pi * np.array(Sample.tau)
        dmn = np.array(self.r1 - self.r0)
        self.r_mn = np.round(rot_vec(np.dot(tau, dmn), Sample.n), 3)


class Dimers(object):
    """
    -------------------------------------------------------------------------
    Attributes
    -------------------------------------------------------------------------
    idx                       index of (atom0,atom1)
    atom0                       first atom
    atom1                       second atom
    r0 = (0,0,0)
    r1 = (0,0,0)
    s0                          spin size of first atom
    s1                          spin size of second atom
    dij                         neighboring distance, in units of eff. latt. params.
    r_mn                        rotation matrix Rn-m
    j                           strength of magnetic interaction, scalar
    -------------------------------------------------------------------------
    See Neutron Scatterings in Condensed Matter Physics,
    by Albert Furrer, Joel Mesot and Thierry Strassle
    """

    def __init__(self, Sample, idx0=None, idx1=None, r0=(0, 0, 0), r1=(0, 0, 0), j=1):
        self.idx = (idx0, idx1)
        self.atom0 = Sample.atoms[idx0]
        self.atom1 = Sample.atoms[idx1]
        if not self.atom0.ion == self.atom1.ion:
            print("Caution! Dimer of different ions.")  # not implemented
        self.r0 = np.array(r0)
        self.r1 = np.array(r1)
        self.s0 = self.atom0.s
        self.s1 = self.atom1.s
        if not self.s0 == self.s1:
            print("Caution! Dimer of different spin sizes.")  # not implemented
        self.dij = np.round(self.atom1.t - self.atom0.t + r1 - r0, 3)

        if np.array(j).ndim:
            print("Magnetic interaction J needs to be a scalar.")
        else:
            self.j = j


# testing
if __name__ == "__main__":
    at1 = Atoms(t=(0, 0, 0), ion="Mn2", spin=1)
    # print(at1.ff_params)
    # -------------------------- form factor --------------------------
    kx = np.arange(0, 100, 10)
    ky = 0
    kz = 0
    k_mesh = np.meshgrid(kx, ky, kz, indexing="ij")
    print(at1.form_factor(k_mesh, ff_params=at1.ff_params))
    # ------------------------------------------------------------------
