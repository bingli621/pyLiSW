# pyLiSW (python Linear Spin Wave)
 Modeling and simulation of single-crystal spin wave data from inelastic neutron scattering experiments, based on linear spin wave theory, using Python.

Tutorial 1. 1D FM chain

![FM_1d](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/1_FM_1d.png)

Tutorial 2. 2D FM on a square lattice

![FM_2d](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/2_FM_2d_100.png "Along (100)")
![FM_2d](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/2_FM_2d_110.png "Along (110)")

Tutorial 3. Neel type AFM chain

![AFM_1d](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/3_AFM_1d_Neel.png)

Tutorial 4. Neel type AFM chain, modeled using a supercell (Miller indices are doubled)

![AFM_1d_s](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/4_AFM_1d_Neel_supercell.png)


Tutorial 10. Neel type AFM on a honeycomb lattice, modeled using a supercell (Miller indices are doubled)

![AFM_hc_disp_110](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/10_AFM_honeycomb_1.png)

![AFM_hc_inten_110](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/10_AFM_honeycomb_2.png)

![AFM_hc_disp_m110](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/10_AFM_honeycomb_3.png)

![AFM_hc_inten_m110](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/10_AFM_honeycomb_4.png)

Tutorial 11. 1D incommensurate cycloid chain. Moments rotates in xz-plane, propagation vector tau=(0.1, 0, 0). Cycloid due to competing exchange interation, FM J1 = -1 meV, AFM J2 = 0.309 meV

![cycloid_CEI_disp](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/11_AFM_1d_cycloid_CEI_1.png)
![cycloid_CEI](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/11_AFM_1d_cycloid_CEI_2.png)

Tutorial 12. 1D incommensurate cycloid chain. Moments rotates in xz-plane, propagation vector tau=(0.1, 0, 0). Cycloid due to competing exchange interation, FM J1 = -1 meV, AFM J2 = 0.309 meV. Calculated using a supercell containing 10 unit cells.

![cycloid_CEI_sup_disp](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/12_AFM_1d_cycloid_supercell_CEI_1.png)
![cycloid_sup_CEI](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/12_AFM_1d_cycloid_supercell_CEI_2.png)

Tutorial 13. 1D incommensurate cycloid chain. Moments rotates in xz-plane, propagation vector tau=(0.1, 0, 0). Cycloid due to DM interation, FM J1 = -1 meV, D_y = +0.727 meV

![cycloid_DMI_disp](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/13_AFM_1d_cycloid_DMI_1.png)
![cycloid_DMI](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/13_AFM_1d_cycloid_DMI_2.png)

Tutorial 14. 1D incommensurate cycloid chain. Moments rotates in xz-plane, propagation vector tau=(0.1, 0, 0). Cycloid due to DM interation, FM J1 = -1 meV, D_y = +0.727 meV. Calculated using a supercell containing 10 unit cells.

![cycloid_DMI_sup_disp](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/14_AFM_1d_cycloid_supercell_DMI_1.png)
![cycloid_sup_DMI](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/14_AFM_1d_cycloid_supercell_DMI_2.png)

Tutorial 15. Cmoparison of 1D incommensurate cycloid chains due to competeing exchange interactions (CEI) and DM interactions (DMI). Moments rotates in xz-plane, propagation vector tau=(0.2, 0, 0). FM J1 = -1 meV in both cases. In CEI case, AFM J2 = 0.809 meV. In DMI case, Dy = +3.078 meV.

![cycloid_CEI_DMI_disp](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/15_AFM_1d_cycloid_CEI_DMI_comparison_3.png)
![cycloid_CEI_DMI](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/15_AFM_1d_cycloid_CEI_DMI_comparison_4.png)

Tutorial 16. 1D Neel AFM chain in applied magnetic field.
![AFM_in_field](https://raw.githubusercontent.com/bingli621/pyLiSW/master/tutorials/16_AFM_1d_Neel_supercell_in_field_2.png)