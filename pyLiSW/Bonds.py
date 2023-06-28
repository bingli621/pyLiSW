# -*- coding: utf-8 -*-
import numpy as np
from utils import rot_vec


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
    dmn                         neighboring unit cell distance
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
        self.dij = np.round(self.atom1.t - self.atom0.t + self.r1 - self.r0, 3)
        self.dmn = np.round(self.r1 - self.r0, 3)

        if np.array(j).ndim:
            self.j = j
        else:  # input j as a scalar
            self.j = j * np.eye(3)

        tau = 2 * np.pi * np.array(Sample.tau)
        self.r_mn = np.round(rot_vec(np.dot(tau, self.dmn), Sample.n_R), 3)
