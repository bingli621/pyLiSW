# -*- coding: utf-8 -*-
import numpy as np


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
