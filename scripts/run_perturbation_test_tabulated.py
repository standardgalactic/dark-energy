#!/usr/bin/env python3
"""
Omitted dark-energy perturbations with the exact QCD-DE equation of state.

Extends run_perturbation_test.py. Instead of fitting a CPL fluid, the equation of state w(a) of the
QCD-DE component at the published best fit is read from the authors' own background run and fed to
CLASS as a tabulated fluid (fluid_equation_of_state = TAB, provided by class_qcd_experimental.patch).
The fluid then reproduces the QCD-DE background to numerical precision, so the perturbed and
unperturbed (fld_unperturbed) runs differ only through the dark-energy perturbations.

Output: how well the tabulated fluid reproduces the QCD-DE spectra when unperturbed (a check of the
background), the effect of perturbations for the tabulated w(z), and the same for the CPL fit for a
direct comparison.

Usage: run_perturbation_test_tabulated.py /path/to/class_qcd /path/to/workdir
"""
import os, subprocess, sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

cls, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(os.path.join(work, "out"), exist_ok=True)
COMMON = """h = 0.6774
omega_b = 0.02237
omega_cdm = 0.11965
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.06
T_ncdm = 0.71611
A_s = 2.1e-9
n_s = 0.9649
tau_reio = 0.051
"""
SPECTRA = "output = tCl,pCl,lCl,mPk\nlensing = yes\nl_max_scalars = 2500\nP_k_max_h/Mpc = 1.\nz_pk = 0\n"
QCD = "Omega_fld = 0\nOmega_scf = 0\nOmega_Lambda = 0\nOmega_qcd = -1\nz_qcd = 0.68\nDz_qcd = 0.82\nswitch_function_qcd = exp\n"

def run(name, body):
    path = os.path.join(work, name + ".ini")
    open(path, "w").write(COMMON + body + "root = %s/out/%s_\n" % (work, name))
    subprocess.run([os.path.join(cls, "class"), path], cwd=cls, check=True, stdout=subprocess.DEVNULL)

# 1. QCD-DE background and its effective equation of state
run("qcdbg", QCD + "write background = yes\n")
run("qcd", SPECTRA + QCD)
d = np.loadtxt(os.path.join(work, "out", "qcdbg_00_background.dat"))
d = d[np.argsort(d[:, 0])]
z, H, rq, rcrit = d[:, 0], d[:, 3], d[:, 16], d[:, 14]
lna = -np.log1p(z)                       # increasing with row index reversed below
order = np.argsort(lna)
good = rq[order] > 0
lna, rq, H, z = lna[order][good], rq[order][good], H[order][good], z[order][good]
d = d[order][good]
rest = H ** 2 - rq
# w from the density: w = -1 - (1/3) dln rho / dln a (cubic spline of ln rho in ln a)
sp = CubicSpline(lna, np.log(rq))
w = -1.0 - sp(lna, 1) / 3.0
# cross-check against p_qcd / rho_qcd of the background table
wcol = d[:, 17] / d[:, 16]
sel = (z > 0) & (z < 5)
print("w(z) from rho vs p/rho column, max |diff| for z<5: %.2e" % np.max(np.abs(w[sel] - wcol[sel])))
for zz in (0, 0.5, 1, 2, 3):
    i = np.argmin(abs(z - zz)); print("  z=%.1f  w=%.3f" % (zz, w[i]))
# Beyond z_cut the exact w(z) drives rho_DE below any representable number (rho ~ exp(-z/dz)); a double
# cannot carry the integral, so w is blended smoothly to -1 over z_cut/2 < z < z_cut, which freezes the
# already negligible density instead of letting it underflow. The dark-energy fraction there is below
# the printed bound.
Z_CUT = 30.0
x = np.clip((np.log1p(z) - np.log1p(Z_CUT / 2)) / (np.log1p(Z_CUT) - np.log1p(Z_CUT / 2)), 0, 1)
blend = x * x * (3 - 2 * x)
w = w * (1 - blend) + (-1.0) * blend
keep = z <= 5 * Z_CUT
lna_t, w_t = lna[keep], w[keep]
dw = CubicSpline(lna_t, w_t)(lna_t, 1)
integ = np.concatenate([[0], np.cumsum(0.5 * (3 * (1 + w_t[1:]) + 3 * (1 + w_t[:-1])) * np.diff(lna_t))])
lnrho_rel = -integ  # ln rho(a) up to a constant, from d ln rho = -3(1+w) d ln a
i30 = np.argmin(abs(z[keep] - Z_CUT))
lnr = lnrho_rel - lnrho_rel[np.argmin(abs(z[keep]))]
print("ln[rho_DE(z=%g)/rho_DE(0)] with the blended table: %.1f; exact: %.1f"
      % (Z_CUT, lnr[i30], np.log(rq[np.argmin(abs(z - Z_CUT))] / rq[np.argmin(z)])))
lna, w = lna_t, w_t
tab = os.path.join(work, "w_qcd_table.dat")
np.savetxt(tab, np.column_stack([lna, w, dw]), fmt="%.10e")
# check that the tabulated w reproduces rho_DE and H(z) to z = Z_CUT
rho_tab = rq[np.argmin(z)] * np.exp(lnr)
zk = z[keep]
dH = np.sqrt(rest[keep] + rho_tab) / H[keep] - 1
print("H(z) from tabulated w vs QCD-DE background: max |dH/H| for z<%g = %.2e" % (Z_CUT, np.max(np.abs(dH[zk < Z_CUT]))))
Om_de = rq[np.argmin(z)] / d[np.argmin(z), 14]
print("Omega_DE(z=0) = %.6f" % Om_de)

# 2. tabulated fluid, perturbed and unperturbed
FLDT = ("Omega_Lambda = 0\nOmega_scf = 0\nfluid_equation_of_state = TAB\nfld_table_file = %s\n"
        "cs2_fld = 1\nuse_ppf = yes\nc_gamma_over_c_fld = 0.4\n" % tab)
run("tab_pert", SPECTRA + FLDT + "fld_unperturbed = no\n")
run("tab_unpert", SPECTRA + FLDT + "fld_unperturbed = yes\n")

# 3. the CPL fit for comparison (as in run_perturbation_test.py)
def rho(w0, wa):
    return rq[-1] * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
def cost(p):
    dh = np.sqrt(rest + rho(*p)) / H - 1
    s = z <= 3
    return 1e3 * (np.sqrt(np.mean(dh[s] ** 2)) + 0.5 * np.max(np.abs(dh[s])))
best = min((minimize(cost, [a, b], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10))
            for a in (-1.1, -0.9, -0.7) for b in (-1, 0, 1)), key=lambda r: r.fun)
w0, wa = (float(x) for x in best.x)
FLDC = ("Omega_Lambda = 0\nOmega_scf = 0\nfluid_equation_of_state = CLP\nw0_fld = %r\nwa_fld = %r\n"
        "cs2_fld = 1\nuse_ppf = yes\nc_gamma_over_c_fld = 0.4\n" % (w0, wa))
run("cpl_pert", SPECTRA + FLDC + "fld_unperturbed = no\n")
run("cpl_unpert", SPECTRA + FLDC + "fld_unperturbed = yes\n")

def cl(name):
    return np.loadtxt(os.path.join(work, "out", name + "_00_cl_lensed.dat"))
def pk(name):
    return np.loadtxt(os.path.join(work, "out", name + "_00_pk.dat"))
def compare(a, b, label):
    print("\n" + label)
    l = a[:, 0]; s = l >= 2; cv = np.sqrt(2 / (2 * l + 1))
    for k, c in (("TT", 1), ("EE", 2), ("phiphi", 5)):
        r = (b[:, c] - a[:, c]) / a[:, c]
        i = np.argmax(np.where(s, np.abs(r) / cv, 0))
        print("  %-7s max |dC/C| = %.2e;  worst l = %d (%.2f of cosmic variance);  sum (dC/C / cv)^2 = %.1f"
              % (k, np.max(np.abs(r[s])), l[i], abs(r[i]) / cv[i], np.sum((r[s] / cv[s]) ** 2)))
def pkshift(a, b, label):
    ka, kb = pk(a)[:, 0], pk(b)[:, 0]
    k = ka[(ka > kb.min()) & (ka < kb.max())]
    r = np.interp(np.log(k), np.log(kb), pk(b)[:, 1]) / np.interp(np.log(k), np.log(ka), pk(a)[:, 1]) - 1
    print("  %s: P(k) change for k>0.01 = %+.2e on average; implied sigma8 shift %+.2f percent"
          % (label, r[k > 0.01].mean(), 50 * r[k > 0.01].mean()))

compare(cl("qcd"), cl("tab_unpert"), "Background check: tabulated fluid, unperturbed, against the QCD-DE run (should vanish)")
compare(cl("tab_unpert"), cl("tab_pert"), "Perturbation effect for the exact QCD w(z) (tabulated fluid)")
compare(cl("cpl_unpert"), cl("cpl_pert"), "Perturbation effect for the CPL fit (w0 = %.3f, wa = %.3f)" % (w0, wa))
print("\nMatter power spectrum")
pkshift("qcd", "tab_unpert", "background check")
pkshift("tab_unpert", "tab_pert", "exact w(z)")
pkshift("cpl_unpert", "cpl_pert", "CPL fit")
