# -*- coding: utf-8 -*-
import time

# from functools import partial
# from multiprocessing import Pool
import concurrent.futures
import numpy as np
import sympy as sp

from Atoms import Atoms
from QEspace import QEspace

from matplotlib import pyplot as plt


class LSWT(QEspace):
    """
    Calculate S(Q,w) using linear spin wave theory (LSWT)
    -------------------------------------------------------------------------
    Class attributes
    -------------------------------------------------------------------------
    mu_B                    1 mu_B = 0.05788 meV/T
    kelvin_by_meV           1 meV = 11.6045 K
    gamma_r0_squared        290.6109 mbarn
    -------------------------------------------------------------------------
    Attributes
    -------------------------------------------------------------------------
    h_plus_tau              mesh of h plus tau if AFM
    k_plus_tau              mesh of k plus tau if AFM
    l_plus_tau              mesh of l plus tau if AFM
    h_minus_tau             mesh of h plus tau if AFM
    k_minus_tau             mesh of k plus tau if AFM
    l_minus_tau             mesh of l plus tau if AFM
    sqw_prime_components    components of S'(q+/-tau,w)^{alpha, beta} that contributes
                            to the INS intensity, with dimension (n_prime_sqw, 3)
                            [[tau, alpha, beta], ....], tau = 0,1,2 for
                            S'(q,w)^{alpha, beta}, S'(q+tau,w)^{alpha, beta} or
                            S'(q-tau,w)^{alpha, beta}, respectively
    sqw_components          components of S(q,w)^{alpha, beta} that contributes
                            to the INS intensity, with dimension (n_sqw, 2)
                            [[alpha, beta], ....]
    sqw_components_coeffs   [[index, coefficients], ...] of S(q,w)^{alpha, beta}
                            index of corresponding component in sqw_prime_components
    mag_form_factors        magnetic form factor matrix, with dimension
                            (n_sqw_prime, qx, qy, qz, 2*n_ion, 2*n_ion)
    dipolar_factors         dipole factor delta_ab - q_a*q_b for S(q,E)_ab
                            dimension as (n_sqw, qx, qy, qz)
    eng                     disperion relation energy en(q),
                            dimension (qx, qy, qz, 2*n_ion))
    eng_plus_tau            en(q + tau)
    eng_minus_tau           en(q - tau)
    mat_T                   matrix T(q)
    mat_Td                  matrix T^dagger(q)
    mat_T_plus_tau          T(q + tau)
    mat_T_minus_tau         T(q - tau)
    mat_Td_plus_tau         T^dagger(q + tau)
    mat_Td_minus_tau        T^dagger(q - tau)
    sqw_prime               S'(q,w) * form_factor * diploar_factor, with dimension
                            (n_sqw_prime, qx, qy, qz, 2*n_ion)
    sqw                     S(q,w) * form_factor * diploar_factor , with dimension
                            (n_sqw, qx, qy, qz, 2 * n_ion, tau_idx = 1) for FM,
                            (n_sqw, qx, qy, qz, 2 * n_ion, tau_idx = 3) for AFM
    bose_factor             Bose factor
    gamma_list              damping factor as a function of energy
    inten                   INS intensity
    amp                     amplitute of INS intensity
    processes               number of cores for parallel computing
    ei                      incident neutron energy to determine kinematic boundary
    -------------------------------------------------------------------------
    Static methods
    -------------------------------------------------------------------------
    sqw_components_print    print the components of S'(q,w)^{alpha, beta}
                            that contributes to the INS intensity
    mat_h                   Hamiltonian in reciprocal space
    ham_solver              Diagonalization of the Hamiltonian
    damping_factor_init
    damping_factor_mapping
    -------------------------------------------------------------------------
    Methods
    -------------------------------------------------------------------------
    eng_calc                calculate the dipersion relation and eigenvectors
    dispersion_calc         calculate dispersion relation
    sqw_component_calc      check the components of S'(q,w)^{alpha,beta}

    dipolar_factor_calc     calculate the dipolar factors
    mag_form_factor_calc(   calculate the magnetic form factor matrices,
        FORM_FACTOR)        if FORM_FACTOR = Fasle, spin = 1, form factor not included
    sqw_prime_calc          calculate S'(q,w) * form_factor * diploar_factor
    sqw_calc                calculate S(q,w) from S'(q,w)
    bose_factor_calc        calculate the Bose factor

    inten_calc              calculate INS intensities
    kin_lim_calc            calculate the kinematic limit
    -------------------------------------------------------------------------
    """

    mu_B = 0.05788  # 1 mu_B = 0.05788 meV/T
    kelvin_by_meV = 11.6045  # 1 meV = 11.6045 K
    gamma_r0_squared = 290.6109  # mbarn#

    # mu_B = 1  # 1 mu_B = 0.05788 meV/T
    # gamma_r0_squared = 1  # mbarn#

    def __init__(
        self,
        qe_ranges,
        Sample,
        proj_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        axes=["(H,0,0)", "(0,K,0)", "(0,0,L)"],
        ei=None,
    ):
        """
        inheritance from QEspace
        """
        super().__init__(qe_ranges, Sample, proj_axes, axes)

        if not Sample.tau == (0, 0, 0):
            self.h_plus_tau = self.h + self.Sample.tau[0]
            self.k_plus_tau = self.k + self.Sample.tau[1]
            self.l_plus_tau = self.l + self.Sample.tau[2]
            self.h_minus_tau = self.h - self.Sample.tau[0]
            self.k_minus_tau = self.k - self.Sample.tau[1]
            self.l_minus_tau = self.l - self.Sample.tau[2]

        self.sqw_prime_components = self.sqw_prime_components_calc()
        LSWT.sqw_components_print(self.sqw_prime_components)
        self.sqw_components, self.sqw_components_coeffs = self.sqw_components_calc()
        LSWT.sqw_components_print(
            self.sqw_components, self.sqw_prime_components, self.sqw_components_coeffs
        )
        self.dipolar_factors = self.dipolar_factor_calc()
        self.sqw_components_update()

        self.mag_form_factors = self.mag_form_factor_calc(FORM_FACTOR=True)
        self.BOSE_FACTOR = True
        self.gamma_list = None  # in energy
        self.processes = 4  # multiprocessing
        self.ei = ei

    @staticmethod
    def bose_factor_calc(elist, te, BOSE_FACTOR=True):
        """
        S(Q,w) = X''(Q,w) * (1+n)
        1+n = 1/(1-exp(-hbar.w/kB.T)))
        """
        # return 1 + 1.0 / (np.exp(en * LSWT.kelvin_by_meV / self.Sample.te) - 1)
        # avoid warning of true_division
        if BOSE_FACTOR:
            np.seterr(divide="ignore", invalid="ignore")
            bose = np.nan_to_num(1.0 / (1.0 - np.exp(-elist * LSWT.kelvin_by_meV / te)))
            np.seterr(divide="warn", invalid="warn")
            # advoid zero energy divergence
            epsilon = 1e-4
            bose = np.where(np.abs(elist) < epsilon, 1, bose)
        else:  # for test purposes
            bose = np.ones_like(elist)
            bose[elist < 0] = -1
        return bose

    @staticmethod
    def mat_h(hkl, Sample):
        """
        build Hamiltonian matrix g
        written as a static method, for parallelization

        """
        kx, ky, kz = np.array(list(zip(*hkl)))
        n_dim = Sample.n_dim
        k_dim = np.shape(kx)[0]
        sp = (k_dim, n_dim, n_dim)
        mat_A = np.zeros(shape=sp, dtype="complex_")
        mat_B = np.zeros(shape=sp, dtype="complex_")
        mat_mA = np.zeros(shape=sp, dtype="complex_")
        mat_mB = np.zeros(shape=sp, dtype="complex_")
        mat_C = np.zeros(shape=sp, dtype="complex_")

        for Bond in Sample.Bonds:  # already doubled due to the reverse pair
            idx0, idx1 = Bond.idx
            ui_T = Bond.atom0.u.T
            uj = Bond.atom1.u
            uj_co = np.conj(Bond.atom1.u)
            vi_T = Bond.atom0.v.T
            vl = Bond.atom1.v
            j_p = Bond.j @ Bond.r_mn
            dmn = Bond.dmn
            a = np.sqrt(Bond.s0 * Bond.s1) * (ui_T @ j_p @ uj_co).item() / 2
            b = np.sqrt(Bond.s0 * Bond.s1) * (ui_T @ j_p @ uj).item() / 2
            c = Bond.s1 * (vi_T @ j_p @ vl).item()
            exp_k = np.exp(-2 * 1j * np.pi * (kx * dmn[0] + ky * dmn[1] + kz * dmn[2]))
            exp_mk = np.exp(2 * 1j * np.pi * (kx * dmn[0] + ky * dmn[1] + kz * dmn[2]))
            mat_A[:, idx0, idx1] += a * exp_mk
            mat_B[:, idx0, idx1] += b * exp_mk
            mat_mA[:, idx0, idx1] += a * exp_k
            mat_mB[:, idx0, idx1] += b * exp_k
            mat_C[:, idx0, idx0] += c * np.ones_like(exp_k)

        for i in range(n_dim):  # Doubled!!!
            atom_i = Sample.atoms[i]
            if np.any(atom_i.aniso):
                s = atom_i.s
                u = atom_i.u
                v = atom_i.v
                a = s * (u.T @ atom_i.aniso @ u.conj()).item()
                b = s * (u.T @ atom_i.aniso @ u).item()
                c = s * (v.T @ atom_i.aniso @ v).item() * 2
                mat_A[:, i, i] += a
                mat_B[:, i, i] += b
                mat_mA[:, i, i] += a
                mat_mB[:, i, i] += b
                mat_C[:, i, i] += c
            else:  # no anisotropy
                pass

        if np.any(Sample.mag):  # doubled!!
            for i in range(n_dim):
                atom_i = Sample.atoms[i]
                v_i = atom_i.v
                h = -atom_i.g * LSWT.mu_B * np.dot(Sample.mag, v_i)
                mat_A[:, i, i] += h
                mat_mA[:, i, i] += h
        else:
            pass

        hmat = np.block(
            [[mat_A - mat_C, mat_B], [np.conj(mat_mB), np.conj(mat_mA) - mat_C]]
        )

        return hmat

    @staticmethod
    def ham_solver(hmat):
        """
        find eigenvalues and eigenvectors of the Hamiltonian matrix g
        written as a static method, for parallelization

        """
        ZERO_ENERGY = False
        evals_k = []
        evecs_k = []
        k_dim = np.shape(hmat)[0]
        n_ion = int(np.shape(hmat)[1] / 2)
        # zero_mat = np.zeros_like(one_mat)
        gmat = np.diag(np.array([1] * n_ion + [-1] * n_ion))
        epsilon = 1e-3
        # delta_mat = np.diag(np.array([1] * n_ion + [1] * n_ion)) * epsilon

        for i in range(k_dim):
            hmat_i = hmat[i, :, :]

            # if not np.all(np.unique(np.abs(evals))):  # zero energies
            #     ZERO_ENERGY = True
            #     hmat[i, :, :] += delta_mat
            # -------------------- Cholesky decomposition ------------------------
            try:
                kmat_d = np.linalg.cholesky(hmat_i)
                kmat = kmat_d.T.conj()
                kgkdmat = kmat @ gmat @ kmat_d
                evals, evecs = np.linalg.eigh(kgkdmat)  # complex hermitian matrix
                evals_r = np.round(evals.real, 4)
                # evecs_r = np.round(evecs.real, 8) + np.round(evecs.imag, 8) * 1j
                # --------------- sort eigenvalues and eigenvectors --------------
                idx = evals.argsort()[::-1]
                evals_s = evals_r[idx]
                evals_k.append(evals_s)

                umat = evecs[:, idx]
                lmat = np.round(umat.T.conj() @ kgkdmat @ umat, 4)
                emat = np.round((gmat @ lmat).real, 4)
                for i in range(n_ion * 2):
                    for j in range(n_ion * 2):
                        if not i == j:
                            if np.abs(emat[i, j]) > 1e-6:
                                print("Warning: eigenvectors non-orthogonal!")
                                # print(emat)
                                # print(umat[:, 0])
                            else:  # zero non-diagonal
                                pass
                        else:  # diagonal
                            pass

                matT = np.linalg.inv(kmat) @ umat @ np.sqrt(emat)
                evecs_k.append(matT)

            except np.linalg.LinAlgError as err:
                print("Warning! Choleskey decomposition failure!")
                print(err)
                # The proposed magnetic structure might mot be
                # the ground statet of the Hamiltonian

                try:  # solve hamiltonian directly
                    evals, evecs = np.linalg.eig(gmat @ hmat_i)
                    # if not np.all(np.round(evals.real, 4)):  # zero energies
                    #     ZERO_ENERGY = True
                    #     # hmat[i, :, :] += delta_mat
                    #     evals += epsilon
                    evals_r = np.round(evals.real, 4)  # discarding the imaginary part
                    if np.any(evals.imag > epsilon):
                        print(
                            "Warining! Imaginary part of eigenvaules larger than {}".format(
                                epsilon
                            )
                        )
                    # --------------- sort eigenvalues and eigenvectors --------------
                    idx = evals.argsort()[::-1]
                    evals_s = evals_r[idx]
                    evals_k.append(evals_s)
                    matT = np.round(evecs[:, idx], 4)  # discarding the imaginary part
                    # --------------- normalize eigenvectors ---------------
                    matTd = matT.T.conj()
                    norms = (matTd @ gmat @ matT).round(8)
                    for i in range(n_ion):
                        matT[:, i] = matT[:, i] / np.sqrt(norms[i, i])
                        matT[:, -1 - i] = matT[:, -1 - i] / np.sqrt(
                            -norms[-1 - i, -1 - i]
                        )
                    evecs_k.append(matT)

                except np.linalg.LinAlgError as err:
                    print("Hamiltonian diagonalization failure!")
                    print(err)
                    print(
                        """The proposed magnetic structure might mot be """
                        + """the ground statet of the Hamiltonian."""
                    )
                    exit()

        # if ZERO_ENERGY:
        #     print(
        #         "Zero eigenvalue encountered. A small value of {}".format(epsilon)
        #         + " is added to the Hamiltonian to avoid divergence."
        #     )

        return np.array(evals_k), np.array(evecs_k)

    @staticmethod
    def sqw_components_print(*args):
        """
        print indices of S(q,w) or S'(q,w)^{alpha, beta} that are non-trivial
        """
        num_of_args = len(args)
        sqw_str = np.array(
            [
                ["xx", "xy", "xz"],
                ["yx", "yy", "yz"],
                ["zx", "zy", "zz"],
            ]
        )
        tau_str = np.array(
            [
                "S'(q,w)",
                "S'(q+tau,w)",
                "S'(q-tau,w)",
            ]
        )
        print(QEspace.divider_str)
        sqw_components_str = []
        if num_of_args == 1:  # S'(q,w)
            sqw_components = args[0]
            for sqw_comp in sqw_components:
                idx0, idx1, idx2 = sqw_comp
                sqw_components_str.append(tau_str[idx0] + sqw_str[idx1][idx2])
            print(
                "The non-trivial components of S' are {}.".format(
                    ", ".join(sqw_components_str)
                )
            )
        elif num_of_args == 3:  # S(q,w)
            sqw_components, sqw_prime_components, sqw_components_coeff = args
            for sqw_comp in sqw_components:
                idx0, idx1 = sqw_comp
                sqw_components_str.append(sqw_str[idx0][idx1])
            print(
                "The non-trivial components of S are S(q,w){}.".format(
                    ", S(q,w)".join(sqw_components_str)
                )
            )
            for idx, sqw_comp in enumerate(sqw_components):
                idx0, idx1 = sqw_comp
                print("S(q,w){} =".format(sqw_str[idx0][idx1]), end="")
                # print(sqw_components_coeff[idx])
                for coeff in sqw_components_coeff[idx]:
                    if sp.re(coeff[1]) and sp.im(coeff[1]) == 0:
                        if sp.re(coeff[1]) > 0:
                            print(" +", end="")
                        else:
                            print(" ", end="")
                        print(
                            "{} * ".format(np.round(float(sp.re(coeff[1])), 3)),
                            end="",
                        )
                    elif sp.re(coeff[1]) == 0 and sp.im(coeff[1]):
                        if sp.im(coeff[1]) > 0:
                            print(" +", end="")
                        else:
                            print(" ", end="")
                        print(
                            "{} * I * ".format(
                                np.round(float(sp.im(coeff[1])), 3),
                            ),
                            end="",
                        )
                    elif sp.re(coeff[1]) and sp.im(coeff[1]):
                        if sp.re(coeff[1]) > 0:
                            print(" +", end="")
                        else:
                            print(" ", end="")
                        print(
                            "({}".format(
                                np.round(float(sp.re(coeff[1])), 3),
                            ),
                            end="",
                        )
                        if sp.im(coeff[1]) > 0:
                            print(" +", end="")
                        else:
                            print(" ", end="")
                        print(
                            "{} * I) * ".format(
                                np.round(float(sp.im(coeff[1])), 3),
                            ),
                            end="",
                        )

                    id0, id1, id2 = sqw_prime_components[coeff[0]]
                    print(tau_str[id0], end="")
                    print(sqw_str[id1, id2], end="")
                print("\r")

    # @staticmethod
    # def sig_conv(x, y):
    #     sz_q, sz_e = y.shape
    #     del_x = np.mean(np.diff(x))
    #     result = np.zeros_like(y)
    #     for idx0 in range(sz_q):
    #         for idx, y0 in enumerate(y[idx0, :]):
    #             result[idx0, :] += y0 * del_x * LSWT.instru_rez(x, x[idx])
    #     return result

    @staticmethod
    def chi_bose(en, engs, intens, te, gamma, BOSE_FACTOR=True):
        """
        chi(energy) as the under-damped harmonic oscillator (UDHO)
        chi times bose factor has the integrated intensity of S(q,w)
        """
        qx, qy, qz, n_dim = np.shape(engs)
        chi_bose = np.zeros((qx, qy, qz) + np.shape(en))
        n_ion = int(n_dim / 2)
        en_step = np.mean(np.diff(en))
        bose = LSWT.bose_factor_calc(en, te, BOSE_FACTOR=BOSE_FACTOR)

        # check degeneracay
        _, dgs = np.unique(engs, axis=3, return_counts=True)
        dg = np.min(dgs)
        if dg > 1:
            print(QEspace.divider_str)
            print("A {}-fold degeneracy is detected.".format(dg))
            print(QEspace.divider_str)

        for i in range(int(n_ion / dg)):  # go through positive energies only
            eng_i = engs[:, :, :, i * dg]
            gamma_i = gamma[:, :, :, i * dg]
            inten_i_pos = np.zeros_like(intens[:, :, :, i * dg])
            inten_i_neg = np.zeros_like(intens[:, :, :, i * dg])
            for j in range(dg):  # add intensity of all degenerate bands
                inten_i_pos += intens[:, :, :, i * dg + j]
                inten_i_neg += intens[:, :, :, 2 * n_ion - 1 - i * dg - j]

            if np.any(np.abs(eng_i + engs[:, :, :, 2 * n_ion - i * dg - 1]) > 1e-6):
                print(
                    "Energies of energy-gain and -loss sides are not the same, "
                    + "we may have a problem."
                )
            # check if intensity on both sizes match
            diff = np.abs(inten_i_pos - inten_i_neg) > 1e-3
            if np.sum(diff) > 0.1 * np.size(diff):
                # more than 10% difference
                print(
                    "Intensity of energy-gain and -loss sides are not the same, "
                    + "we may have a problem related to the band-degeneracy."
                )

            chi_bose_i = (
                en[None, None, None, :]
                * gamma_i[:, :, :, None]
                / (
                    (en[None, None, None, :] ** 2 - eng_i[:, :, :, None] ** 2) ** 2
                    + (en[None, None, None, :] * gamma_i[:, :, :, None]) ** 2
                )
                * bose[None, None, None, :]
            )
            # ------------ Alternativly, Gaussian times Bose factor -----------
            # chi_bose = np.exp(-((x[None, None, None, :] - eng0[:, :, :, None])
            #                     / gamma[:, :, :, None])**2 * 4 * np.log(2)) * \
            #     bose[None, None, None, :]
            # -----------------------------------------------------------------
            # ------ clean the tails if intensity is less than 0.5% of max --------
            # CUT_OFF_PERCENT = 0.001
            # chi_bose_cut_off = CUT_OFF_PERCENT * np.amax(chi_bose_i, axis=-1)
            # chi_bose_i[chi_bose_i < chi_bose_cut_off[:, :, :, None]] = 0
            # ------------------------------------------------------------
            # chi_bose_sum = np.sum(chi_bose_i[:, :, :, en > 0], axis=-1)
            # # chi_bose_sum = np.sum(chi_bose_i, axis=-1)
            # chi_bose_norm = chi_bose_i / chi_bose_sum[:, :, :, None] / en_step
            # # chi_bose += chi_bose_norm * inten_i[:, :, :, None]
            # chi_bose += (
            #     chi_bose_norm
            #     * (inten_i * (LSWT.bose_factor_calc(eng_i, te)+1))[:, :, :, None]
            # )
            # ---------------Normalized to 2*n_B + 1 ------------------------------
            chi_bose_sum = np.sum(chi_bose_i, axis=-1)
            chi_bose_norm = chi_bose_i / chi_bose_sum[:, :, :, None] / en_step
            chi_bose += (
                chi_bose_norm
                * (inten_i_pos * (LSWT.bose_factor_calc(eng_i, te) * 2 + 1))[
                    :, :, :, None
                ]
            )
        return chi_bose

    @staticmethod
    def damping_factor_init(eng_k, step, gamma_fcn=None):
        """
        Initialize dampping factor as a function of energy
        defalt is energy step size, or a constant value
        """
        pstart = np.min(eng_k.real)
        pend = np.max(eng_k.real)
        disp_list = np.arange(pstart, pend, step)
        if not gamma_fcn:
            gamma = np.min(np.abs(np.diff(eng_k)))
            # gamma as a constant
            # gamma = np.array([0.115 * 2 * np.sqrt(2 * np.log(2))])
            gamma = np.array([0.03 * 2 * np.sqrt(2 * np.log(2))])
            gamma_list = np.tile(gamma, disp_list.shape)
        else:
            gamma_list = gamma_fcn(disp_list)
        print("Size of gamma_list is {}".format(gamma_list.size))
        return disp_list, gamma_list

    @staticmethod
    def damping_factor_mapping(eng_k, disp_list, gamma_list):
        """
        Map damping factor gamma from energy w to momentum q
        """
        sz = np.shape(eng_k)
        ek = np.ravel(eng_k)
        gamma_mat = np.empty_like(ek)
        for idx0, ek0 in enumerate(ek):
            if any(disp_list >= ek0):
                idx = np.argmax(disp_list >= ek0) - 1
                if idx == -1:  # special care for minimum
                    idx == 0
            else:
                idx = np.size(disp_list) - 1
            gamma_mat[idx0] = gamma_list[idx]
        return np.reshape(gamma_mat, sz)

    def sqw_prime_components_calc(self):
        """
        return [[tau, alpha, beta], ... ] if S'(q +/- tau,w)^(alpha, beta} is non-trivial
        """
        sqw_components = []
        for i in range(3):
            for j in range(3):
                # this if-statment works for collinear case
                # but miss components for incommensurate non-collinear cases
                if np.any(self.mat_YZVW(i, j).real) or np.any(self.mat_YZVW(i, j).imag):
                    if self.Sample.tau == (0, 0, 0):  # FM
                        sqw_components.append([0, i, j])
                    else:  # AFM
                        for tau in range(3):
                            sqw_components.append([tau, i, j])

                # if not (i == 2 and j == 2):  # ignores S'zz
                #     if self.Sample.tau == (0, 0, 0):  # FM
                #         sqw_components.append([0, i, j])
                #     else:  # AFM
                #         for tau in range(3):
                #             sqw_components.append([tau, i, j])

        return sqw_components

    def sqw_components_calc(self):
        """
        return [[alpha, beta], ... ] if S(q,w)^(alpha, beta} is non-trivial
        return [[(idx, coeff),...], ...],

        """
        # print(self.sqw_prime_components)
        xx, xy, xz, yx, yy, yz, zx, zy, zz = sp.symbols("xx xy xz yx yy yz zx zy zz")
        sqw_prime_array = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])

        sqw_prime = np.zeros_like(sqw_prime_array)
        sqw_prime_plus_tau = np.zeros_like(sqw_prime_array)
        sqw_prime_minus_tau = np.zeros_like(sqw_prime_array)

        for sqw_prime_comp in self.sqw_prime_components:
            idx0, idx1, idx2 = sqw_prime_comp
            match idx0:
                case 0:
                    sqw_prime[idx1, idx2] = sqw_prime_array[idx1, idx2]
                case 1:
                    sqw_prime_plus_tau[idx1, idx2] = sqw_prime_array[idx1, idx2]
                case 2:
                    sqw_prime_minus_tau[idx1, idx2] = sqw_prime_array[idx1, idx2]

        sqw_components_coeff = []

        if self.Sample.tau == (0, 0, 0):  # FM
            sqw_components = []
            sqw = sqw_prime
            for sqw_component in self.sqw_prime_components:
                sqw_component_coeff = []
                _, idx1, idx2 = sqw_component
                sqw_components.append([idx1, idx2])
                for i in range(3):
                    for j in range(3):
                        coeff = sqw[idx1, idx2].coeff(sqw_prime_array[i, j])
                        if coeff:  # non-zero coefficient
                            for idx, item in enumerate(self.sqw_prime_components):
                                if item == [0, i, j]:
                                    sqw_component_coeff.append([idx, coeff])
                sqw_components_coeff.append(sqw_component_coeff)
            # self.sqw_components = sqw_components

        else:  # AFM
            sqw = sqw_prime @ self.Sample.mat_R2
            sqw_plus_tau = sqw_prime_plus_tau @ self.Sample.mat_R1
            sqw_minus_tau = sqw_prime_minus_tau @ np.conj(self.Sample.mat_R1)

            sqw_components = []
            for i in range(3):
                for j in range(3):
                    if np.any([sqw[i, j], sqw_plus_tau[i, j], sqw_minus_tau[i, j]]):
                        sqw_components.append([i, j])
                    else:  # do nothing if all zeros
                        pass

            for sqw_component in sqw_components:
                sqw_component_coeff = []
                idx0, idx1 = sqw_component
                for i in range(3):
                    for j in range(3):
                        coeff = sqw[idx0, idx1].coeff(sqw_prime_array[i, j])
                        if coeff:
                            for idx, item in enumerate(self.sqw_prime_components):
                                if item == [0, i, j]:
                                    sqw_component_coeff.append([idx, coeff])
                        coeff_plus_tau = sqw_plus_tau[idx0, idx1].coeff(
                            sqw_prime_array[i, j]
                        )

                        if coeff_plus_tau:
                            for idx, item in enumerate(self.sqw_prime_components):
                                if item == [1, i, j]:
                                    sqw_component_coeff.append([idx, coeff_plus_tau])
                        coeff_minus_tau = sqw_minus_tau[idx0, idx1].coeff(
                            sqw_prime_array[i, j]
                        )

                        if coeff_minus_tau:
                            for idx, item in enumerate(self.sqw_prime_components):
                                if item == [2, i, j]:
                                    sqw_component_coeff.append([idx, coeff_minus_tau])
                sqw_components_coeff.append(sqw_component_coeff)
            # print(sqw_components_coeff)
        return sqw_components, sqw_components_coeff

    def energy_calc(self):
        """
        Calculated dispersion energy and eigenvalues for q, q+tau and q-tau
        """
        n_dim = 2 * self.Sample.n_dim
        sz_evals = np.shape(self.q_mesh)[1:] + (n_dim,)
        sz_evecs = np.shape(self.q_mesh)[1:] + (n_dim, n_dim)

        hkl = zip(np.ravel(self.h), np.ravel(self.k), np.ravel(self.l))

        if not self.Sample.tau == (0, 0, 0):  # AFM
            hkl_plus_tau = zip(
                np.ravel(self.h_plus_tau),
                np.ravel(self.k_plus_tau),
                np.ravel(self.l_plus_tau),
            )
            hkl_minus_tau = zip(
                np.ravel(self.h_minus_tau),
                np.ravel(self.k_minus_tau),
                np.ravel(self.l_minus_tau),
            )
            # mat_h_plus_tau = LSWT.mat_h(hkl_minus_tau, self.Sample)
            # mat_h_minus_tau = LSWT.mat_h(hkl_plus_tau, self.Sample)
            mat_h_plus_tau = LSWT.mat_h(hkl_plus_tau, self.Sample)
            mat_h_minus_tau = LSWT.mat_h(hkl_minus_tau, self.Sample)

        mat_h = LSWT.mat_h(hkl, self.Sample)

        if self.processes > 1:  # multiprocessing
            with concurrent.futures.ProcessPoolExecutor() as executor:
                if not self.Sample.tau == (0, 0, 0):  # AFM
                    mat_h_plus_tau_split = np.array_split(
                        mat_h_plus_tau, self.processes
                    )
                    eval_plus_tau_list = []
                    evec_plus_tau_list = []
                    results_plus_tau = executor.map(
                        LSWT.ham_solver, mat_h_plus_tau_split
                    )
                    for result in results_plus_tau:
                        eval_plus_tau_list.append(result[0])
                        evec_plus_tau_list.append(result[1])

                    mat_h_minus_tau_split = np.array_split(
                        mat_h_minus_tau, self.processes
                    )
                    eval_minus_tau_list = []
                    evec_minus_tau_list = []
                    results_minus_tau = executor.map(
                        LSWT.ham_solver, mat_h_minus_tau_split
                    )
                    for result in results_minus_tau:
                        eval_minus_tau_list.append(result[0])
                        evec_minus_tau_list.append(result[1])

                    evals_plus_tau = np.concatenate(eval_plus_tau_list, axis=0)
                    evecs_plus_tau = np.concatenate(evec_plus_tau_list, axis=0)
                    evals_minus_tau = np.concatenate(eval_minus_tau_list, axis=0)
                    evecs_minus_tau = np.concatenate(evec_minus_tau_list, axis=0)

                mat_h_split = np.array_split(mat_h, self.processes)
                eval_list = []
                evec_list = []
                results = executor.map(LSWT.ham_solver, mat_h_split)
                for result in results:
                    eval_list.append(result[0])
                    evec_list.append(result[1])
                evals = np.concatenate(eval_list, axis=0)
                evecs = np.concatenate(evec_list, axis=0)

        else:  # single-core
            if not self.Sample.tau == (0, 0, 0):  # AFM
                evals_minus_tau, evecs_minus_tau = LSWT.ham_solver(mat_h_plus_tau)
                evals_plus_tau, evecs_plus_tau = LSWT.ham_solver(mat_h_minus_tau)
            evals, evecs = LSWT.ham_solver(mat_h)

        if not self.Sample.tau == (0, 0, 0):  # AFM
            self.eng_plus_tau = np.reshape(np.round(evals_plus_tau.real, 8), sz_evals)
            self.mat_T_plus_tau = np.reshape(evecs_plus_tau, sz_evecs)
            self.mat_Td_plus_tau = np.reshape(
                np.transpose(evecs_plus_tau.conj(), (0, 2, 1)), sz_evecs
            )

            self.eng_minus_tau = np.reshape(np.round(evals_minus_tau.real, 8), sz_evals)
            self.mat_T_minus_tau = np.reshape(evecs_minus_tau, sz_evecs)
            self.mat_Td_minus_tau = np.reshape(
                np.transpose(evecs_minus_tau.conj(), (0, 2, 1)), sz_evecs
            )
        self.eng = np.reshape(np.round(evals.real, 8), sz_evals)
        self.mat_T = np.reshape(evecs, sz_evecs)
        self.mat_Td = np.reshape(np.transpose(evecs.conj(), (0, 2, 1)), sz_evecs)
        # self.mat_Td = np.transpose(np.reshape(evecs.conj(), sz_evecs),(0,1,2,4,3))

    def dispersion_calc(self):
        """
        Calculated dispersion relation.
        """
        n_dim = 2 * self.Sample.n_dim
        sz_evals = np.shape(self.q_mesh)[1:] + (n_dim,)
        hkl = zip(np.ravel(self.h), np.ravel(self.k), np.ravel(self.l))
        evals, _ = LSWT.ham_solver(LSWT.mat_h(hkl, self.Sample))
        self.eng = np.reshape(np.round(evals.real, 8), sz_evals)

        if not self.Sample.tau == (0, 0, 0):  # AFM
            hkl_plus_tau = zip(
                np.ravel(self.h_plus_tau),
                np.ravel(self.k_plus_tau),
                np.ravel(self.l_plus_tau),
            )
            evals_plus_tau, _ = LSWT.ham_solver(LSWT.mat_h(hkl_plus_tau, self.Sample))
            hkl_minus_tau = zip(
                np.ravel(self.h_minus_tau),
                np.ravel(self.k_minus_tau),
                np.ravel(self.l_minus_tau),
            )
            evals_minus_tau, _ = LSWT.ham_solver(LSWT.mat_h(hkl_minus_tau, self.Sample))

            self.eng_plus_tau = np.reshape(np.round(evals_plus_tau.real, 8), sz_evals)
            self.eng_minus_tau = np.reshape(np.round(evals_minus_tau.real, 8), sz_evals)

    def mat_YZVW(self, alpha, beta):
        """
        matrix of dimension 2*n_ion by 2*n_ion,
        to determine which of the S'(q,w) terms are trivial
        """
        n_ion = self.Sample.n_dim
        matY = np.zeros((n_ion, n_ion), dtype="complex_")
        matZ = np.zeros((n_ion, n_ion), dtype="complex_")
        matV = np.zeros((n_ion, n_ion), dtype="complex_")
        matW = np.zeros((n_ion, n_ion), dtype="complex_")
        mat = np.zeros((2 * n_ion, 2 * n_ion), dtype="complex_")

        for i in range(n_ion):  # number of ions
            si = self.Sample.atoms[i].s
            ui = self.Sample.atoms[i].u
            for j in range(n_ion):
                sj = self.Sample.atoms[j].s
                uj = self.Sample.atoms[j].u
                matY[i, j] = np.sqrt(si * sj) * ui[alpha] * np.conj(uj[beta])
                matZ[i, j] = np.sqrt(si * sj) * ui[alpha] * uj[beta]
                matV[i, j] = np.sqrt(si * sj) * np.conj(ui[alpha]) * np.conj(uj[beta])
                matW[i, j] = np.sqrt(si * sj) * np.conj(ui[alpha]) * uj[beta]

        # mat = np.block([[matY[:, :], matZ[:, :]], [matV[:, :], matW[:, :]]])
        mat = np.block([[matY, matZ], [matV, matW]])
        return mat

    def dipolar_factor_calc(self):
        """
        Dipolar factor calculation.
        Update sqw_components and sqw_components_coeffs if dipolar factors are zeros.

        """
        start = time.perf_counter()
        dipole_factors = []
        # sz = np.shape(self.q_mesh)[1:]
        qx, qy, qz = self.q_mesh

        sqw_comps = []
        sqw_components_coeffs = []
        for idx, sqw_comp in enumerate(self.sqw_components):
            idx0, idx1 = sqw_comp
            q_sq_sum = np.square(qx) + np.square(qy) + np.square(qz)
            q_dict = {0: qx, 1: qy, 2: qz}
            delta = lambda idx0, idx1: 1 if idx0 == idx1 else 0

            # avoid warning of true_division
            np.seterr(divide="ignore", invalid="ignore")
            qq = np.nan_to_num(q_dict[idx0] * q_dict[idx1] / q_sq_sum)
            np.seterr(divide="warn", invalid="warn")

            dipole_factor = delta(idx0, idx1) - qq
            if np.any(dipole_factor[np.nonzero(q_sq_sum)]):  # non-zero
                dipole_factors.append(dipole_factor)
                sqw_comps.append(sqw_comp)
                sqw_components_coeffs.append(self.sqw_components_coeffs[idx])
            else:  # zero dipolar factors, remove the corresponding sqw_components_coeff
                pass
        print(QEspace.divider_str)
        finish = time.perf_counter()
        print(
            "Dipolar factor calculated in {} second(s)".format(round(finish - start, 4))
        )
        if not len(sqw_comps) == len(self.sqw_components):
            print("Dipolar factor suppresses certain S(q,w) component")
            self.sqw_components = sqw_comps
            self.sqw_components_coeffs = sqw_components_coeffs
            LSWT.sqw_components_print(
                self.sqw_components,
                self.sqw_prime_components,
                self.sqw_components_coeffs,
            )
        return np.array(dipole_factors)

    def sqw_components_update(self):
        """
        Update sqw_prime_components, removed the ones not needed
        update the index in sqw_component_coeff accordingly
        """
        sqw_prime_components = []
        n_sqw = np.shape(self.sqw_components)[0]
        for n in range(n_sqw):
            for sqw_component in self.sqw_components_coeffs[n]:
                idx, _ = sqw_component
                if not self.sqw_prime_components[idx] in sqw_prime_components:
                    sqw_prime_components.append(self.sqw_prime_components[idx])

        for n in range(n_sqw):
            for i, sqw_component in enumerate(self.sqw_components_coeffs[n]):
                idx, _ = sqw_component
                idx_new = sqw_prime_components.index(self.sqw_prime_components[idx])
                self.sqw_components_coeffs[n][i][0] = idx_new

        self.sqw_prime_components = sqw_prime_components

    def mag_form_factor_calc(self, FORM_FACTOR=True):
        """
        matrix of magnetic form factors in the YZVW matrix,
        with dimension (n_sqw, qx, qy, qz, 2*n_ion, 2*n_ion)
        """
        start = time.perf_counter()
        n_ion = self.Sample.n_dim
        n_sqw_prime = np.shape(self.sqw_prime_components)[0]
        _, qx, qy, qz = np.shape(self.q_mesh)
        matY = np.zeros(
            (n_sqw_prime,) + (qx, qy, qz) + (n_ion, n_ion),
            dtype="complex_",
        )
        matZ = np.zeros_like(matY, dtype="complex_")
        matV = np.zeros_like(matY, dtype="complex_")
        matW = np.zeros_like(matY, dtype="complex_")
        mat = np.zeros(
            (n_sqw_prime,) + (qx, qy, qz) + (2 * n_ion, 2 * n_ion),
            dtype="complex_",
        )
        for idx, sqw_comp in enumerate(self.sqw_prime_components):
            _, alpha, beta = sqw_comp
            for i in range(n_ion):  # number of ions
                ati = self.Sample.atoms[i]
                si = ati.s
                ui = ati.u
                gi = ati.g
                ti = ati.t * np.array(
                    [self.Sample.a_eff, self.Sample.b_eff, self.Sample.c_eff]
                )
                ff_i = Atoms.form_factor(self.q_mesh, ati.ff_params, gi)
                exp_i = Atoms.exp_ikt(self.q_mesh, ti)
                for j in range(n_ion):
                    atj = self.Sample.atoms[j]
                    sj = atj.s
                    uj = atj.u
                    gj = atj.g
                    tj = atj.t * np.array(
                        [self.Sample.a_eff, self.Sample.b_eff, self.Sample.c_eff]
                    )
                    ff_j = Atoms.form_factor(self.q_mesh, atj.ff_params, gj)
                    exp_j = Atoms.exp_ikt(self.q_mesh, tj)
                    if FORM_FACTOR:
                        prefactor = (
                            gi
                            * gj
                            / 4
                            * np.sqrt(si * sj)
                            * (ff_i * ff_j).real
                            * exp_i
                            * np.conj(exp_j)
                        )
                    else:  # no magnetic form factor, for testing
                        prefactor = (
                            np.sqrt(si * sj) * exp_i * np.conj(exp_j)
                        )  # test YZWV
                    matY[idx, :, :, :, i, j] = prefactor * ui[alpha] * np.conj(uj[beta])
                    matZ[idx, :, :, :, i, j] = prefactor * ui[alpha] * uj[beta]
                    matV[idx, :, :, :, i, j] = (
                        prefactor * np.conj(ui[alpha]) * np.conj(uj[beta])
                    )
                    matW[idx, :, :, :, i, j] = prefactor * np.conj(ui[alpha]) * uj[beta]
            # mat[idx, :, :, :, :, :] = np.block(
            #     [
            #         [matY[idx, :, :, :, :, :], matZ[idx, :, :, :, :, :]],
            #         [matV[idx, :, :, :, :, :], matW[idx, :, :, :, :, :]],
            #     ]
            # )
        mat = np.block([[matY, matZ], [matV, matW]])

        finish = time.perf_counter()
        print(QEspace.divider_str)
        print(
            "Magnetic form factors calculated in {} second(s)".format(
                round(finish - start, 4)
            )
        )
        print(QEspace.divider_str)
        return mat

    def sqw_prime_calc(self):
        """
        Calculation S'(q +/- tau,w)^{alpha, beta} * form_factor
        """
        n_dim = 2 * self.Sample.n_dim
        n_sqw_prime = np.shape(self.sqw_prime_components)[0]
        sz = (n_sqw_prime,) + np.shape(self.q_mesh)[1:] + (n_dim,)

        sqw_prime = np.zeros(sz, dtype="complex_")
        for idx, sqw_prime_comp in enumerate(self.sqw_prime_components):
            tau, _, _ = sqw_prime_comp

            # for i in range(n_dim):
            #     match tau:
            #         case 0:  # S'(q ,w)^{alpha, beta}
            #             sqw_prime[idx, :, :, :, i] += (
            #                 self.mat_Td
            #                 @ self.mag_form_factors[idx, :, :, :, :, :]
            #                 @ self.mat_T
            #             )[:, :, :, i, i]
            #         case 1:  # S'(q + tau ,w)^{alpha, beta}
            #             sqw_prime[idx, :, :, :, i] += (
            #                 self.mat_Td_plus_tau
            #                 @ self.mag_form_factors[idx, :, :, :, :, :]
            #                 @ self.mat_T_plus_tau
            #             )[:, :, :, i, i]
            #         case 2:  # S'(q - tau,w)^{alpha, beta}
            #             sqw_prime[idx, :, :, :, i] += (
            #                 self.mat_Td_minus_tau
            #                 @ self.mag_form_factors[idx, :, :, :, :, :]
            #                 @ self.mat_T_minus_tau
            #             )[:, :, :, i, i]
            # ----------------------------------------
            match tau:
                case 0:  # S'(q ,w)^{alpha, beta}
                    sqw_prime_i = (
                        self.mat_Td
                        @ self.mag_form_factors[idx, :, :, :, :, :]
                        @ self.mat_T
                    )
                    for i in range(self.Sample.n_dim):
                        sqw_prime[idx, :, :, :, i] = sqw_prime_i[:, :, :, i, i]
                        sqw_prime[idx, :, :, :, -1 - i] = sqw_prime_i[
                            :, :, :, -1 - i, -1 - i
                        ]

                case 1:  # S'(q + tau ,w)^{alpha, beta}
                    sqw_prime_i = (
                        self.mat_Td_plus_tau
                        @ self.mag_form_factors[idx, :, :, :, :, :]
                        @ self.mat_T_plus_tau
                    )
                    for i in range(self.Sample.n_dim):
                        sqw_prime[idx, :, :, :, i] = sqw_prime_i[:, :, :, i, i]
                        sqw_prime[idx, :, :, :, -1 - i] = sqw_prime_i[
                            :, :, :, -1 - i, -1 - i
                        ]

                case 2:  # S'(q - tau,w)^{alpha, beta}
                    sqw_prime_i = (
                        self.mat_Td_minus_tau
                        @ self.mag_form_factors[idx, :, :, :, :, :]
                        @ self.mat_T_minus_tau
                    )
                    for i in range(self.Sample.n_dim):
                        sqw_prime[idx, :, :, :, i] = sqw_prime_i[:, :, :, i, i]
                        sqw_prime[idx, :, :, :, -1 - i] = sqw_prime_i[
                            :, :, :, -1 - i, -1 - i
                        ]

        self.sqw_prime = sqw_prime / n_dim

    def sqw_calc(self):
        """
        Calculation S(q,w)^{alpha, beta}
        """
        if not self.Sample.tau == (0, 0, 0):  # AFM
            n_dim = (2 * self.Sample.n_dim, 3)
        else:  # FM
            n_dim = (2 * self.Sample.n_dim, 1)
        # n_dim = (2 * self.Sample.n_dim,)
        n_sqw_components = np.shape(self.sqw_components)[0]
        sz = (n_sqw_components,) + np.shape(self.q_mesh)[1:] + n_dim
        sqw = np.zeros(sz, dtype="complex_")

        self.energy_calc()
        self.sqw_prime_calc()

        for idx in range(n_sqw_components):
            for coeff in self.sqw_components_coeffs[idx]:
                tau_idx = self.sqw_prime_components[coeff[0]][0]
                sqw[idx, :, :, :, :, tau_idx] += (
                    float(sp.re(coeff[1])) + 1j * float(sp.im(coeff[1]))
                ) * self.sqw_prime[coeff[0], :, :, :, :]
        self.sqw = sqw.real

    def kin_lim_calc(self):
        """
        Calculate the kinematic boundary.
        Assuming a 1 deg inaccessible region due to beam stop.
        """
        beam_stop = 1
        # h_bar**2/(2*m) = 2.072138
        ki = np.sqrt(self.ei / 2.072138)
        qx, qy, qz = self.q_mesh
        q_sq = qx**2 + qy**2 + qz**2
        sintheta = np.sin(np.deg2rad(beam_stop))
        rt = q_sq - ki**2 * sintheta
        rt[rt < 0] = 0
        lim = 2.072138 * (-q_sq + 2 * ki * np.sqrt(rt) + 2 * (ki * sintheta) ** 2)
        return lim

    def inten_calc(self, mask=None):
        """
        Calculation S(q,w)^{alpha, beta} * dipolar factor
        initialize gamma with energy resolution step size
        """
        start = time.perf_counter()
        self.sqw_calc()

        # --------------------  map damping from enery to Q -----------------
        # default damping
        disp_list, gamma_list = LSWT.damping_factor_init(
            self.eng,
            step=np.mean(np.abs(np.diff(self.elist))),
            gamma_fcn=self.Sample.gamma_fnc,
        )
        # self.gamma_list
        self.gamma_mat = LSWT.damping_factor_mapping(self.eng, disp_list, gamma_list)
        if not self.Sample.tau == (0, 0, 0):  # AFM
            disp_list_plus_tau, gamma_list_plus_tau = LSWT.damping_factor_init(
                self.eng_plus_tau,
                step=np.mean(np.abs(np.diff(self.elist))),
                gamma_fcn=self.Sample.gamma_fnc,
            )
            disp_list_minus_tau, gamma_list_minus_tau = LSWT.damping_factor_init(
                self.eng_minus_tau,
                step=np.mean(np.abs(np.diff(self.elist))),
                gamma_fcn=self.Sample.gamma_fnc,
            )
            self.gamma_mat_plus_tau = LSWT.damping_factor_mapping(
                self.eng_plus_tau, disp_list_plus_tau, gamma_list_plus_tau
            )
            self.gamma_mat_minus_tau = LSWT.damping_factor_mapping(
                self.eng_minus_tau, disp_list_minus_tau, gamma_list_minus_tau
            )
        # ------------------------------------------------------------

        n_sqw, qx, qy, qz, n_dim, n_tau = np.shape(self.sqw)
        inten = np.zeros((qx, qy, qz, n_dim, n_tau), dtype="float_")

        for idx in range(n_sqw):
            dp = self.dipolar_factors[idx, :, :, :]
            for t in range(n_tau):
                for d in range(n_dim):
                    if np.any(self.sqw[idx, :, :, :, d, t]):
                        inten[:, :, :, d, t] += dp * self.sqw[idx, :, :, :, d, t].real
        self.inten = inten
        # -------------------- mask given branch ---------------------
        # if mask is None:
        #     self.inten = inten
        # else:
        #     n = self.Sample.n_dim * 2
        #     mask_branches = []
        #     for i in mask:
        #         mask_branches.append(i)
        #         mask_branches.append(n - i - 1)
        #     self.inten = np.delete(inten, mask_branches, axis=3)
        #     self.eng = np.delete(self.eng, mask_branches, axis=3)
        #     self.gamma_mat = np.delete(self.gamma_mat, mask_branches, axis=3)
        # ------------------------------------------------------------
        amp = LSWT.chi_bose(
            self.elist,
            self.eng,
            self.inten[:, :, :, :, 0],
            self.Sample.te,
            self.gamma_mat,
            self.BOSE_FACTOR,
        )
        if not self.Sample.tau == (0, 0, 0):  # AFM
            amp_plus_tau = LSWT.chi_bose(
                self.elist,
                self.eng_plus_tau,
                self.inten[:, :, :, :, 1],
                self.Sample.te,
                self.gamma_mat_plus_tau,
                self.BOSE_FACTOR,
            )
            amp_minus_tau = LSWT.chi_bose(
                self.elist,
                self.eng_minus_tau,
                self.inten[:, :, :, :, 2],
                self.Sample.te,
                self.gamma_mat_minus_tau,
                self.BOSE_FACTOR,
            )
            self.amp = amp + amp_plus_tau + amp_minus_tau
        else:  # FM
            self.amp = amp  # No resolution convolution

        # kinematic limit
        if self.ei is not None:
            kin_lim = self.kin_lim_calc()
            amp[self.elist[None, None, None, :] > kin_lim[:, :, :, None]] = None

        finish = time.perf_counter()
        print(
            "INS intensities calculated in {} second(s)".format(
                round(finish - start, 4)
            )
        )

    def slice(
        self, slice_ranges, plot_axes, aspect=None, PLOT=True, SIM=True, **kwargs
    ):
        x, y, slice, xlab, ylab, tit = super().slice(
            slice_ranges, plot_axes, aspect, PLOT, SIM, **kwargs
        )

        return x, y, slice, xlab, ylab, tit

    ##############################################################################
    #     # --------------------------------------------------------------
    #     start = time.perf_counter()
    #     sz = amp.shape
    #     amp_list = amp.reshape(-1, self.elist.size)
    #     amp_list_split = np.split(amp_list, self.processes)
    #     sig_conv_partial = partial(LSWT.sig_conv, self.elist)
    #     pool = Pool(self.processes)
    #     sqw_conv = pool.map(sig_conv_partial, amp_list_split)
    #     pool.close()
    #     self.sqw = np.reshape(sqw_conv, sz)
    #     finish = time.perf_counter()
    #     print("S(q,w) calculated in {} second(s)".format(round(finish - start, 4)))


if __name__ == "__main__":
    pass
