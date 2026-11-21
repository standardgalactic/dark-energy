#!/usr/bin/env python3
"""
Quantifies the effect of omitting dark-energy perturbations on a QCD-DE-like background.

Prerequisites: a clone of dlehdgk/class_qcd with class_qcd_fld_unperturbed.patch applied
(git apply class_qcd_fld_unperturbed.patch) and built with `make class`.

Method
1. Run the authors' QCD-DE background at the published best fit (z_q = 0.68, dz_q = 0.82).
2. Fit a CPL fluid (w0, wa) to the QCD dark-energy density rho_qcd(z) over 0 <= z <= 3.
3. Run CLASS with that CPL fluid with PPF perturbations, and again with the fluid restricted
   to the background (fld_unperturbed = yes). Both share one background, so the difference
   isolates the dark-energy perturbations.
4. Run the actual QCD-DE model (background only, as published) and compare it to the
   unperturbed CPL run to see how well the CPL fluid mimics the QCD background.

Usage: run_perturbation_test.py /path/to/class_qcd /path/to/workdir
"""
import os, subprocess, sys
import numpy as np
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
    subprocess.run([os.path.join(cls, "class"), path], cwd=cls, check=True,
                   stdout=subprocess.DEVNULL)

# 1. background of the published model
run("qcdbg", QCD + "write background = yes\n")
run("qcd", SPECTRA + QCD)
d = np.loadtxt(os.path.join(work, "out", "qcdbg_00_background.dat"))
d = d[np.argsort(d[:, 0])]
z, H, rq = d[:, 0], d[:, 3], d[:, 16]
rest = H ** 2 - rq

# 2. CPL fit to the dark-energy density
def rho(w0, wa):
    return rq[0] * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
def cost(p):
    dh = np.sqrt(rest + rho(*p)) / H - 1
    s = z <= 3
    return 1e3 * (np.sqrt(np.mean(dh[s] ** 2)) + 0.5 * np.max(np.abs(dh[s])))
best = min((minimize(cost, [a, b], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10))
            for a in (-1.1, -0.9, -0.7) for b in (-1, 0, 1)), key=lambda r: r.fun)
w0, wa = (float(x) for x in best.x)
dh = np.sqrt(rest + rho(w0, wa)) / H - 1
print("CPL fit: w0 = %.4f, wa = %.4f, max |dH/H| (z<=3) = %.2e" % (w0, wa, np.max(np.abs(dh[z <= 3]))))

# 3. perturbed and unperturbed CPL fluid
FLD = ("Omega_Lambda = 0\nOmega_scf = 0\nfluid_equation_of_state = CLP\nw0_fld = %r\nwa_fld = %r\n"
       "cs2_fld = 1\nuse_ppf = yes\nc_gamma_over_c_fld = 0.4\n" % (w0, wa))
run("cpl_pert", SPECTRA + FLD + "fld_unperturbed = no\n")
run("cpl_unpert", SPECTRA + FLD + "fld_unperturbed = yes\n")

def cl(name):
    return np.loadtxt(os.path.join(work, "out", name + "_00_cl_lensed.dat"))
def compare(a, b, label):
    print("\n" + label)
    l = a[:, 0]; sel = l >= 2; cv = np.sqrt(2 / (2 * l + 1))
    for k, c in (("TT", 1), ("EE", 2), ("phiphi", 5)):
        r = (b[:, c] - a[:, c]) / a[:, c]
        i = np.argmax(np.where(sel, np.abs(r) / cv, 0))
        print("  %-7s max |dC/C| = %.2e;  worst l = %d (%.2f of cosmic variance);  sum (dC/C / cv)^2 over l = %.1f"
              % (k, np.max(np.abs(r[sel])), l[i], abs(r[i]) / cv[i], np.sum((r[sel] / cv[sel]) ** 2)))
compare(cl("cpl_unpert"), cl("cpl_pert"), "Effect of dark-energy perturbations (same background), ideal full-sky")
compare(cl("qcd"), cl("cpl_unpert"), "CPL mimic against the actual QCD-DE background (mimic quality)")

pu = np.loadtxt(os.path.join(work, "out", "cpl_unpert_00_pk.dat"))
pp = np.loadtxt(os.path.join(work, "out", "cpl_pert_00_pk.dat"))
r = pp[:, 1] / pu[:, 1] - 1
k = pu[:, 0]
print("\nMatter power spectrum, perturbed / unperturbed - 1:")
for lo, hi in ((1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0)):
    m = (k >= lo) & (k < hi)
    print("  k in [%g, %g) h/Mpc: %+.2e to %+.2e" % (lo, hi, r[m].min(), r[m].max()))
print("  implied shift in sigma8: about %+.2f percent" % (50 * r[k > 0.01].mean()))
