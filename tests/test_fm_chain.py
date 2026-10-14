import numpy as np
import pytest

import pyLiSW
from pyLiSW import LSWT, Atoms, Bonds, Sample

J = -1.0  # FM exchange (meV); AFM is positive
S = 2
D = -0.01  # easy-axis single-ion anisotropy (meV)


@pytest.fixture
def fm_chain():
    sample = Sample((3, 3, 3), (0, 0, 0), (0, 1, 0), 2, mag=[0, 0, 0])
    sample.add_atoms(
        [Atoms(t=(0, 0, 0), ion="Mn2", spin=S, aniso=[[0, 0, 0], [0, 0, 0], [0, 0, D]])]
    )
    sample.add_bonds([Bonds(sample, 0, 0, r1=(1, 0, 0), j=J)])
    return sample


def test_version():
    assert isinstance(pyLiSW.__version__, str)


def test_fm_chain_dispersion_matches_analytic(fm_chain):
    qe_range = ([0, 1.01, 0.05], [0, 0.01, 0.01], [0, 0.01, 0.01], [-20, 20, 0.01])
    sim = LSWT(qe_range, fm_chain)
    sim.dispersion_calc()

    h = np.squeeze(sim.h)
    expected = 2 * abs(J) * S * (1 - np.cos(2 * np.pi * h)) + 2 * abs(D) * S
    eng = np.squeeze(sim.eng)
    np.testing.assert_allclose(eng.max(axis=-1), expected, atol=1e-4)
    np.testing.assert_allclose(eng.min(axis=-1), -expected, atol=1e-4)


def test_fm_chain_intensity_is_finite(fm_chain):
    qe_range = ([0, 1.01, 0.05], [0, 0.01, 0.01], [0, 0.01, 0.01], [-10, 10, 0.1])
    sim = LSWT(qe_range, fm_chain)
    sim.inten_calc()
    assert np.all(np.isfinite(sim.inten))
    assert sim.inten.max() > 0
