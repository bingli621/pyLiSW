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
    ion                         type of magnetic ion, e.g. 'Mn2'
    g                           gyromagnetic ratio, g = 2 by default
    s                           absolute value of spin, scalar
    spin_p                      S' after global rotation R, 3x1 vector
    spin_pp                     S'' = [0,0,s] after local rotation R'', 3x1 vector
    ff_params                   parameters for magnetic form factors
    u                           after local rotation R_prime, 3x1 vector
    v                           after local rotation R_prime, 3x1 vector
    mag_eff                     Effective field after global rotation
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
        n_Rp=(0, 1, 0),
        aniso=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
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

        rp_mat = rot_vec(theta, n_Rp)  # identity matrices by default
        self.u = np.round([rp_mat[:, 0] + 1j * rp_mat[:, 1]], 3).T
        self.v = np.round([rp_mat[:, 2]], 3).T
        self.spin_p = np.round(rp_mat @ self.spin_pp, 3).T
        # print((self.u))
        # print((self.v))
        self.aniso = np.array(aniso)
        #  self.mag_eff = [0, 0, 0]

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

    @staticmethod
    def exp_ikt(k_mesh, t):
        """
        return Fourier tansformation of exp(i*k*t)
        """
        kx, ky, kz = k_mesh
        tx, ty, tz = t
        exp_ikt = np.exp(1j * (kx * tx + ky * ty + kz * tz))
        return exp_ikt


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
