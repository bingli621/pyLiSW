# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

params = {
    "legend.fontsize": "x-large",
    # "figure.figsize": (8, 6),
    "axes.labelsize": "xx-large",
    "axes.titlesize": "xx-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
}
plt.rcParams.update(params)


class QEspace(object):
    """
    Making slices and cuts
    Summing slices and cuts with proper error propagation
    -------------------------------------------------------------------------
    Attributes
    -------------------------------------------------------------------------
    sample          sample-dependent information
    axes
    proj_axes       projection axes for viewing
    axes_labels     axes labels
    bin_labels      axes labels for binning
    xlist           1d list of reciprocal x in r.l.u.
    ylist           1d list of reciprocal y in r.l.u.
    zlist           1d list of reciprocal z in r.l.u.
                    h_eff, k_eff, l_eff are orthogonal
    elist           1d list of energy transfer in meV
    h               3d q mesh in r.l.u.
    k               3d q mesh in r.l.u.
    l               3d q mesh in r.l.u.
    q_mesh          3d q mesh in inverse Angstrom
    amp             4d simulated neutron intensity
    data            4d intensity data
    err             4d error bar
    -------------------------------------------------------------------------
    Methods
    -------------------------------------------------------------------------
    load data
    slice
    cut
    -------------------------------------------------------------------------
    """

    divider_str = "-" * 150
    divider_str2 = "=" * 150

    def __init__(
        self,
        qe_ranges,
        Sample,
        proj_axes,
        axes,
    ):
        """
        Mesh 4D grid according to qe_ranges = [start, stop, step] * 4
        """
        self.Sample = Sample

        self.axes = axes  # High-symmetry directions for meshing
        p_mat = proj_axes
        x, y, z = self.axes
        self.axes_labels = np.array(
            [
                r"${}$ (r.l.u.)".format(x),
                r"${}$ (r.l.u.)".format(y),
                r"${}$ (r.l.u.)".format(z),
                "$E$ (meV)",
            ]
        )
        self.bin_labels = np.array(
            [
                r"${}$, $H$ = ".format(x),
                r"${}$, $K$ = ".format(y),
                r"${}$, $L$ = ".format(z),
                "$E$ = ",
            ]
        )

        if any(  # check for orthogonality
            [
                np.dot(proj_axes[0], proj_axes[1]),
                np.dot(proj_axes[1], proj_axes[2]),
                np.dot(proj_axes[2], proj_axes[0]),
            ]
        ):
            print("Warning! Pojection axes are not orthogonal.")
        else:
            print(
                QEspace.divider_str2
                + "\nInitializing reciprocal space with projection P={}\n".format(
                    proj_axes
                )
                + "{}, H={}, {}, K={}, {}, L={}".format(
                    self.axes[0],
                    qe_ranges[0],
                    self.axes[1],
                    qe_ranges[1],
                    self.axes[2],
                    qe_ranges[2],
                )
            )

        inv_p = np.linalg.inv(p_mat) * np.linalg.det(p_mat)

        qe_plot_list = [None] * 4
        for idx, qe_range in enumerate(qe_ranges):
            pstart, pend, pstep = qe_range
            # length of np.arange is not stable
            # qe_plot_list[idx] = np.arange(pstart, pend, pstep)
            num = int(round((pend - pstart) / pstep))
            qe_plot_list[idx] = np.linspace(pstart, pend, num, endpoint=False)

        self.xlist, self.ylist, self.zlist, self.elist = qe_plot_list
        self.x, self.y, self.z = np.meshgrid(
            self.xlist, self.ylist, self.zlist, indexing="ij"
        )
        self.h = inv_p[0, 0] * self.x + inv_p[0, 1] * self.y + inv_p[0, 2] * self.z
        self.k = inv_p[1, 0] * self.x + inv_p[1, 1] * self.y + inv_p[1, 2] * self.z
        self.l = inv_p[2, 0] * self.x + inv_p[2, 1] * self.y + inv_p[2, 2] * self.z

        self.q_mesh = (
            self.h * 2 * np.pi / Sample.a_eff,
            self.k * 2 * np.pi / Sample.b_eff,
            self.l * 2 * np.pi / Sample.c_eff,
        )

        sz = np.shape(self.h) + np.shape(self.elist)
        self.amp = np.zeros(sz)
        self.data = None
        self.err = None

    def load_data(self, datafiles, scalefactor=1, moveaxis=None):  # (2, 0)):
        """
        Load multiple ascii files
        Might need to change axes orders
        moveaxis=(2,0) for 0K0, (2,1) for HH0
        """
        sz = np.shape(self.amp)
        self.data = np.zeros(sz)
        self.err = np.zeros(sz)

        data = [None] * np.size(datafiles)
        err = [None] * np.size(datafiles)
        for i, datafile in enumerate(datafiles):
            with open(datafile, "r") as f:
                header = f.readlines()
                str = header[1][:-1]
                str = str.split(" ")
                dim = [int(s) for s in str[-1].split("x") if s.isdigit()]
                print(f"Loading data, dimension={dim}")
                # load data
                data_raw = np.loadtxt(datafile, skiprows=0)
                data_int = data_raw[:, 0] * scalefactor
                data_err = data_raw[:, 1] * scalefactor
                # data[i] = data_int.reshape(tuple(dim))
                # err[i] = (data_err.reshape(tuple(dim))) ** 2
                data[i] = data_int.reshape(tuple(sz))
                err[i] = (data_err.reshape(tuple(sz))) ** 2
                # --------------------------------------------
                if moveaxis is not None:  # move axis based on the axis from mantid 0K0
                    data[i] = np.moveaxis(data[i], moveaxis[0], moveaxis[1])
                    err[i] = np.moveaxis(err[i], moveaxis[0], moveaxis[1])

        if np.size(datafiles) == 1:
            self.data = data[0]
            self.err = np.sqrt(err[0]) / (~np.isnan(data[0]))
        else:
            np.seterr(divide="ignore", invalid="ignore")
            cnt = np.sum(~np.isnan(data), axis=0)
            self.data = np.nansum(data, axis=0) / cnt
            self.err = np.sqrt(np.nansum(err, axis=0)) / cnt
            np.seterr(divide="warn", invalid="warn")

        return self.data, self.err

    def slice(
        self, slice_ranges, plot_axes, aspect=None, PLOT=True, SIM=False, **kwargs
    ):
        """
        Make slice and generate a contour plot. Two axes not for plotting will
        be binned (averaged). Two axes for plotting can be rebinned if a
        different range/ step size is given. Plot data by default. Plot
        simulation if SIM=True
        """
        qe_ranges_list = [self.xlist, self.ylist, self.zlist, self.elist]
        bin_axes = list({0, 1, 2, 3} - set(plot_axes))

        # --------------- carve out the 4D chunck ----------------
        slice_idx = []
        for idx, slice_range in enumerate(slice_ranges):
            qe_range = np.round(qe_ranges_list[idx], 3)
            pstart = np.round(slice_range[0], 3)
            pend = np.round(slice_range[1], 3)
            idx1 = np.argmax(qe_range >= pstart)
            if any(qe_range >= pend):
                idx2 = np.argmax(qe_range >= pend) - 1
            else:
                idx2 = np.size(qe_range) - 1
            # print('idx = {0}, {1}'.format(idx1, idx2))
            # print('points = {0}, {1}'.format(qe_range[idx1], qe_range[idx2]))
            slice_idx.append((idx1, idx2 + 1))  # including last point
        idx_mat = (
            np.s_[slice_idx[0][0] : slice_idx[0][1]],
            np.s_[slice_idx[1][0] : slice_idx[1][1]],
            np.s_[slice_idx[2][0] : slice_idx[2][1]],
            np.s_[slice_idx[3][0] : slice_idx[3][1]],
        )
        # --------- average over two binning dirctions ------------

        if SIM:
            amp = self.amp[idx_mat]
            cnt = np.nansum(~np.isnan(amp), axis=tuple(bin_axes))
            slice = np.nansum(amp, axis=tuple(bin_axes)) / cnt

        else:
            amp = self.data[idx_mat]
            err_sq = self.err[idx_mat] ** 2
            cnt = np.nansum(~np.isnan(amp), axis=tuple(bin_axes))
            # avoid warning of true_division
            np.seterr(divide="ignore", invalid="ignore")
            slice = np.nansum(amp, axis=tuple(bin_axes)) / cnt
            err = np.sqrt(np.nansum(err_sq, axis=tuple(bin_axes))) / cnt
            np.seterr(divide="warn", invalid="warn")
        # ------- determine plot ranges based on slice_ranges ---------
        plot_ranges = []
        rebin_axes = []
        for plot_axis in list(plot_axes):
            slice_range = slice_ranges[plot_axis]
            qe_range = qe_ranges_list[plot_axis]
            # Adjust plot ranges if start/stop is different
            if slice_range[0] > qe_range[0]:
                REBIN = True
            elif slice_range[1] < qe_range[-1]:
                REBIN = True
            else:  # Check step size
                if np.shape(slice_range)[0] == 3:
                    if (
                        np.abs(slice_range[2] - np.mean(np.abs(np.diff(qe_range))))
                        < 1e-3
                    ):
                        REBIN = False  # same step size, no rebin
                    else:
                        REBIN = True
                else:  # Only start and stop is given, no rebin
                    REBIN = False
            if REBIN:
                rebin_axes.append(plot_axis)
                plot_range = np.arange(*slice_range)
            else:
                plot_range = qe_range
            plot_ranges.append(plot_range)
        # -------------- Rebin in the two plot directions -----------------
        # print('REBIN={}'.format(REBIN))
        if len(rebin_axes):
            sz_x = np.shape(plot_ranges[0])
            sz_y = np.shape(plot_ranges[1])
            # print('slice.shape={}'.format(slice.shape))
            for rebin_axis in rebin_axes:
                # --------------- which axis to rebin ---------------
                rebin_idx = plot_axes.index(rebin_axis)
                # print('rebin_idx = {}'.format(rebin_idx))
                if rebin_idx == 0:  # rebin x axis
                    slice_temp = np.zeros(sz_x + (slice.shape[1],))
                    cnt_temp = np.zeros(sz_x)
                elif rebin_idx == 1:  # rebin y axis
                    slice_temp = np.zeros((slice.shape[0],) + sz_y)
                    cnt_temp = np.zeros(sz_y)
                if not SIM:
                    err_temp = np.zeros_like(slice_temp)
                # print('slice_temp.shape={}'.format(slice_temp.shape))
                qe_range = np.round(
                    qe_ranges_list[rebin_axis][
                        np.s_[slice_idx[rebin_axis][0] : slice_idx[rebin_axis][1]]
                    ],
                    3,
                )
                plot_range = np.round(plot_ranges[rebin_idx], 3)
                # print(qe_range)
                # --------------- find bin box ---------------
                for i, q in enumerate(qe_range):
                    if any(plot_range > q):
                        idx = np.argmax(plot_range > q) - 1
                    else:
                        idx = np.size(plot_range) - 1
                    if rebin_idx == 0:  # rebin x axis
                        slice_temp[idx, :] += slice[i, :]
                        if not SIM:
                            err_temp[idx, :] += err[i, :]
                    elif rebin_idx == 1:  # rebin y axis
                        slice_temp[:, idx] += slice[:, i]
                        if not SIM:
                            err_temp[:, idx] += err[:, i]
                    cnt_temp[idx] += 1
                if rebin_idx == 0:  # rebin x axis
                    # slice_temp /= cnt_temp[cnt_temp > 0][:, None]
                    slice_temp /= cnt_temp[:, None]
                    if not SIM:
                        # err_temp /= cnt_temp[cnt_temp > 0][:, None]
                        err_temp /= cnt_temp[:, None]
                elif rebin_idx == 1:  # rebin y axis
                    # -----------------------------------------------
                    # slice_temp /= cnt_temp[cnt_temp > 0][None, :]
                    slice_temp /= cnt_temp[None, :]
                    if not SIM:
                        # err_temp /= cnt_temp[cnt_temp > 0][None, :]
                        err_temp /= cnt_temp[None, :]
                slice = slice_temp
                if not SIM:
                    err = err_temp
        # ---------------- Axes and title ----------------
        slice_xlabel = self.axes_labels[list(plot_axes)[0]]
        slice_ylabel = self.axes_labels[list(plot_axes)[1]]

        slice_title = "{}[{:.2f}, {:.2f}], \n{}[{:.2f}, {:.2f}]".format(
            self.bin_labels[bin_axes[0]],
            slice_ranges[bin_axes[0]][0],
            slice_ranges[bin_axes[0]][1],
            self.bin_labels[bin_axes[1]],
            slice_ranges[bin_axes[1]][0],
            slice_ranges[bin_axes[1]][1],
            # self.mag_field,
        )
        # ---------------- Plot ----------------
        if PLOT:
            fig, ax = plt.subplots()
            im = ax.pcolormesh(
                plot_ranges[0],
                plot_ranges[1],
                slice.transpose(),
                cmap="jet",
                shading="nearest",
                # shading="flat",
                **kwargs,
            )
            # if self.ei is not None and (3 in plot_axes):
            #     ax.plot(plot_ranges[0], kin_lim, "-w")
            # im = ax.imshow(slice.transpose(),**kwargs)
            # plt.gca().invert_yaxis()
            ax.grid(which="major", alpha=0.5)
            ax.set(xlabel=slice_xlabel, ylabel=slice_ylabel)
            ax.set(title=slice_title)
            # if plot_axes == (0, 1):
            #     ax.set_aspect(1/np.sqrt(3))
            ax.set_xlim(min(plot_ranges[0]), max(plot_ranges[0]))
            ax.set_ylim(min(plot_ranges[1]), max(plot_ranges[1]))
            if aspect:
                ax.set_aspect(aspect)
            cbar = fig.colorbar(im, ax=ax)
            # ax.set_ylim(bottom=-6)
            # ax.set_ylim(top=6)
            fig.tight_layout()
            # mpl.use('macosx')
            # plt.xticks(np.arange(0, 18, 2))
            fig.show()

        if SIM:
            return (
                plot_ranges[0],
                plot_ranges[1],
                slice.transpose(),
                slice_xlabel,
                slice_ylabel,
                slice_title,
            )
        else:
            return (
                plot_ranges[0],
                plot_ranges[1],
                slice.transpose(),
                err.transpose(),
                slice_xlabel,
                slice_ylabel,
                slice_title,
            )

    def cut(self, cut_ranges=None, plot_axis=None, PLOT=True, SIM=False, **kwargs):
        """
        Make cut and generate a line plot. Three axes not for plotting will be
        binned (averaged). One axis for plotting can be rebinned if a different
        step size is given. Plot data if SIM=False. Plot simulation if SIM=True
        but self.data is None. Overplot data and simulation if SIM=True.
        """

        qe_ranges_list = [self.xlist, self.ylist, self.zlist, self.elist]
        bin_axes = list({0, 1, 2, 3} - set([plot_axis]))

        # --------------- carve out the 4D chunck ----------------
        cut_idx = []
        for idx, cut_range in enumerate(cut_ranges):
            qe_range = np.round(qe_ranges_list[idx], 3)
            pstart = np.round(cut_range[0], 3)
            pend = np.round(cut_range[1], 3)
            idx1 = np.argmax(qe_range >= pstart)
            if any(qe_range >= pend):
                idx2 = np.argmax(qe_range >= pend) - 1
            else:
                idx2 = np.size(qe_range) - 1
            # print('idx = {0}, {1}'.format(idx1, idx2))
            # print('points = {0}, {1}'.format(qe_range[idx1], qe_range[idx2]))
            cut_idx.append((idx1, idx2 + 1))  # including last point
        idx_mat = (
            np.s_[cut_idx[0][0] : cut_idx[0][1]],
            np.s_[cut_idx[1][0] : cut_idx[1][1]],
            np.s_[cut_idx[2][0] : cut_idx[2][1]],
            np.s_[cut_idx[3][0] : cut_idx[3][1]],
        )
        # --------- average over three binning dirctions ------------
        if SIM:
            amp_sim = self.amp[idx_mat]
            cnt_sim = np.nansum(~np.isnan(amp_sim), axis=tuple(bin_axes))
            cut_sim = np.nansum(amp_sim, axis=tuple(bin_axes)) / cnt_sim
        if self.data is not None:
            amp_data = self.data[idx_mat]
            err_sq = self.err[idx_mat] ** 2
            cnt = np.nansum(~np.isnan(amp_data), axis=tuple(bin_axes))
            # suppress true_divide warning
            np.seterr(divide="ignore", invalid="ignore")
            cut = np.nansum(amp_data, axis=tuple(bin_axes)) / cnt
            err = np.sqrt(np.nansum(err_sq, axis=tuple(bin_axes))) / cnt
            np.seterr(divide="warn", invalid="warn")

        # ------- determine plot ranges based on slice_ranges ---------
        qe_range = qe_ranges_list[plot_axis]
        cut_range = cut_ranges[plot_axis]
        # Adjust plot ranges if start/stop is different
        if cut_range[0] > qe_range[0]:
            REBIN = True
        elif cut_range[1] < qe_range[-1]:
            REBIN = True
        else:  # Check step size
            if np.shape(cut_range)[0] == 3:
                if np.abs(cut_range[2] - np.mean(np.abs(np.diff(qe_range)))) < 1e-3:
                    REBIN = False  # same step size, no rebin
                else:
                    REBIN = True
            else:  # Only start and stop is given, no rebin
                REBIN = False
        if REBIN:
            plot_range = np.arange(*cut_range)
        else:
            plot_range = qe_range

        # -------------- Rebin in the two plot directions -----------------
        if REBIN:
            sz = np.shape(plot_range)
            if SIM:
                cut_sim_temp = np.zeros(sz)
                cnt_sim_temp = np.zeros(sz)
            if self.data is not None:
                cut_temp = np.zeros(sz)
                err_temp = np.zeros(sz)
                cnt_temp = np.zeros(sz)

            qe_range = np.round(
                qe_ranges_list[plot_axis][
                    np.s_[cut_idx[plot_axis][0] : cut_idx[plot_axis][1]]
                ],
                3,
            )
            plot_range = np.round(plot_range, 3)
            # print(qe_range)
            # --------------- find bin box ---------------
            for i, q in enumerate(qe_range):
                if any(plot_range > q):
                    idx = np.argmax(plot_range > q) - 1
                else:
                    idx = np.size(plot_range) - 1
                if SIM:
                    cut_sim_temp[idx] += cut_sim[i]
                    cnt_sim_temp[idx] += 1
                if self.data is not None:
                    cut_temp[idx] += cut[i]
                    err_temp[idx] += err[i] ** 2
                    cnt_temp[idx] += 1
            if SIM:
                cut_sim_temp /= cnt_sim_temp  # [cnt_sim_temp] ??
                cut_sim = cut_sim_temp
            if self.data is not None:
                np.seterr(divide="ignore", invalid="ignore")
                cut_temp /= cnt_temp
                cut = cut_temp
                err = np.sqrt(err_temp) / cnt_temp
                np.seterr(divide="warn", invalid="warn")

        # ---------------- Axes and title ----------------
        cut_xlabel = self.axes_labels[plot_axis]
        cut_ylabel = "Intensity (mBarn/meV/Sr/f.u.)"
        cut_title = (
            "{}[{:.3f}, {:.3f}],\n".format(
                self.bin_labels[bin_axes[0]],
                cut_ranges[bin_axes[0]][0],
                cut_ranges[bin_axes[0]][1],
            )
            + " {}[{:.3f}, {:.3f}],\n".format(
                self.bin_labels[bin_axes[1]],
                cut_ranges[bin_axes[1]][0],
                cut_ranges[bin_axes[1]][1],
            )
            + " {}[{:.3f}, {:.3f}]".format(
                self.bin_labels[bin_axes[2]],
                cut_ranges[bin_axes[2]][0],
                cut_ranges[bin_axes[2]][1],
            )
            # + "\nH = {} T".format(self.mag_field)
        )

        # ---------------- Plot ----------------
        if PLOT:
            fig, ax = plt.subplots()
            if SIM:
                if self.data is None:
                    im_sim = ax.plot(plot_range, cut_sim, "-o")
                else:
                    im = ax.errorbar(plot_range, cut, err, fmt="o")
                    im_sim = ax.plot(plot_range, cut_sim)
            else:
                im = ax.errorbar(plot_range, cut, err, fmt="o")

            ax.set(xlabel=cut_xlabel, ylabel=cut_ylabel)
            ax.set(title=cut_title)
            plt.ylim(bottom=0)
            # plt.ylim(top=0.1)
            plt.tight_layout()
            plt.grid(alpha=0.6)
            # mpl.use('macosx')
            fig.show()

        if SIM:
            return plot_range, cut_sim, cut_xlabel, cut_ylabel, cut_title
        else:
            return plot_range, cut, err, cut_xlabel, cut_ylabel, cut_title

    def plot_disp(self, q_axis=None, **kwargs):
        """
        Plot dispersion along given high symmetry direction x, y or z
        """
        nq0, nq1, nq2, n_dim = np.shape(self.eng)
        if q_axis == "x":
            plot_x = self.xlist
            label_x = self.axes_labels[0]
            idx1 = round((nq1-1) / 2)
            idx2 = round((nq2-1) / 2)
            plot_ys = self.eng[:, idx1, idx2, :]
            tit = "{}{:.2f}\n{}{:.2f}".format(
                self.bin_labels[1],
                self.ylist[idx1],
                self.bin_labels[2],
                self.zlist[idx2],
            )
        elif q_axis == "y":
            plot_x = self.ylist
            label_x = self.axes_labels[1]
            idx0 = round((nq0-1) / 2)
            idx2 = round((nq2-1) / 2)
            plot_ys = self.eng[idx0, :, idx2, :]
            tit = "{}{:.2f}\n{}{:.2f}".format(
                self.bin_labels[0],
                self.xlist[idx0],
                self.bin_labels[2],
                self.zlist[idx2],
            )
        elif q_axis == "z":
            plot_x = self.zlist
            label_x = self.axes_labels[2]
            idx0 = round((nq0-1) / 2)
            idx1 = round((nq1-1) / 2)
            plot_ys = self.eng[idx0, idx1, :, :]
            tit = "{}{:.2f}\n{}{:.2f}".format(
                self.bin_labels[0],
                self.xlist[idx0],
                self.bin_labels[1],
                self.ylist[idx1],
            )
        else:
            print("Unknown axis for ploting")

        shift = 0  # shift to distinguish in case of degeneracy
        fig, ax = plt.subplots()
        for i in range(n_dim):
            #  plot_y = list(itertools.chain(*self.eng[:, :, :, :]))
            plot_y = plot_ys[:, i]
            ax.plot(plot_x, plot_y + shift * i, "-o", label="{}".format(i))
        ax.legend()
        ax.grid(alpha=0.6)
        ax.set_ylabel(self.axes_labels[-1])
        ax.set_xlabel(label_x)
        ax.set_title(tit)
        plt.tight_layout()
        fig.show()


# -----------------------------------------------------------------------
# export slice to ascii file
# -----------------------------------------------------------------------
# def save2d(self, filename):
#     data = self.Sqw
#     # write file
#     header = "Intensity Error " + \
#         np.array2string(self.axes_labels[0]) + \
#         np.array2string(self.axes_labels[1]) + \
#         np.array2string(self.axes_labels[2]) + \
#         np.array2string(self.axes_labels[3])
#     print(np.shape(data))
#     # header += "\n shape: "+np.shape(data)[0]+"x" + \
#     #     np.shape(data)[1]+"x" + np.shape(data)[2]+"x" + \
#     #     np.shape(data)[3]
#     toPrint = np.c_[data]
#     np.savetxt(filename, toPrint, fmt='%.6e', header=header)
