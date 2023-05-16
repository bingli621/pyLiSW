import numpy as np
from scipy.spatial.transform import Rotation as R


def test():
    return None


def gamma_fnc(x):
    """
    Customize damping FWHM gamma as a function of energy
    """
    # ARCS Ei=75 meV
    # fwhm = +3.3349e-07 * x**3 + 0.00011656 * x**2 - 0.030234 * x + 2.8092
    # CNCS Ei = 3.32 meV
    # fwhm = +0.00030354 * x**3 + 0.0039009 * x**2 - 0.040862 * x + 0.11303
    # CNCS Ei = 1.55 meV
    # fwhm = +0.00096336 * x**3 + 0.005865 * x**2 - 0.027999 * x + 0.038397
    # ARCS Ei=50 meV
    fwhm = +9.1948e-07 * x**3 + 0.00020961 * x**2 - 0.03571 * x + 1.9454
    # ARCS Ei=125 meV
    # fwhm = +1.4245e-07 * x**3 + 8.586e-05 * x**2 - 0.036398 * x + 5.5345
    fwhm = 1 * np.ones_like(x)
    return fwhm


def rot_ang(theta, phi):
    """
    Rotate angle theta about x axis and phi about z axis, in degrees
    """
    rp_mat = R.from_euler("xz", [theta, phi], degrees=True).as_matrix()
    # print(rp_mat)
    return np.array(rp_mat)


def rot_vec(theta, n):
    """
    Rotate theta (in radian) about unit vector n
    """
    if theta:
        n_norm = n / np.linalg.norm(n)
        r_mat = R.from_rotvec(theta * np.array(n_norm)).as_matrix()
    else:
        r_mat = np.identity(3)
    return np.array(r_mat)


def fold_slice(slice, err):
    """
    fold slice along x
    """
    amp = slice
    amp2 = np.flip(slice, axis=1)
    err_sq = err**2
    err_sq2 = np.flip(err, axis=1) ** 2
    cnt = np.nansum([~np.isnan(amp), ~np.isnan(amp2)], axis=0)
    # avoid warning of true_division
    np.seterr(divide="ignore", invalid="ignore")
    slice = np.nansum([amp, amp2], axis=0) / cnt
    err = np.sqrt(np.nansum([err_sq, err_sq2], axis=0)) / cnt
    np.seterr(divide="warn", invalid="warn")
    return slice, err


def add_data(data_list):
    dim = np.shape(data_list[0])
    cnt_list = []
    for data_set in data_list[0:]:
        if not np.shape(data_set) == dim:
            print("Dimensionality not matching.")
        else:
            cnt_list.append((~np.isnan(data_set)).astype(int))

    data_sum = np.nansum(data_list, axis=0)
    cnt_sum = np.nansum(cnt_list, axis=0)
    np.seterr(divide="ignore", invalid="ignore")
    data_avg = data_sum / cnt_sum
    np.seterr(divide="warn", invalid="warn")

    return data_avg
