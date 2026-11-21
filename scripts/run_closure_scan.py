#!/usr/bin/env python3
"""
Sensitivity of the omitted-perturbation effect to the closure of the dark-energy fluid.

Uses the tabulated exact w(z) fluid (see run_perturbation_test_tabulated.py, which must have been run
first in the same workdir so that w_qcd_table.dat and tab_unpert exist). Varies the effective sound
speed cs2_fld and the PPF transition parameter c_gamma_over_c_fld, and also checks the full-range
agreement of the tabulated background with the QCD-DE background.

Usage: run_closure_scan.py /path/to/class_qcd /path/to/workdir
"""
import os, re, subprocess, sys
import numpy as np

cls, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
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
tab = os.path.join(work, "w_qcd_table.dat")
FLD = "Omega_Lambda = 0\nOmega_scf = 0\nfluid_equation_of_state = TAB\nfld_table_file = %s\nuse_ppf = yes\n" % tab

def ini(name, body):
    path = os.path.join(work, name + ".ini")
    open(path, "w").write(COMMON + body + "root = %s/out/%s_\n" % (work, name))
    return path

cases = {"scan_cs2_1e-2": (0.01, 0.4), "scan_cs2_1e-1": (0.1, 0.4),
         "scan_cg_0.2": (1.0, 0.2), "scan_cg_0.8": (1.0, 0.8)}
procs = []
for name, (cs2, cg) in ({} if os.environ.get("REUSE") else cases).items():
    p = ini(name, SPECTRA + FLD + "cs2_fld = %g\nc_gamma_over_c_fld = %g\nfld_unperturbed = no\n" % (cs2, cg))
    procs.append(subprocess.Popen([os.path.join(cls, "class"), p], cwd=cls, stdout=subprocess.DEVNULL))
bg = ini("tabbg", FLD + "cs2_fld = 1\nc_gamma_over_c_fld = 0.4\nwrite background = yes\n")
subprocess.run([os.path.join(cls, "class"), bg], cwd=cls, check=True, stdout=subprocess.DEVNULL)
for p in procs:
    p.wait()

def cl(n): return np.loadtxt(os.path.join(work, "out", n + "_00_cl_lensed.dat"))
def pk(n): return np.loadtxt(os.path.join(work, "out", n + "_00_pk.dat"))
ref, refpk = cl("tab_unpert"), pk("tab_unpert")
print("Closure scan: perturbed minus background-only fluid with the exact w(z)")
print("%-16s %10s %10s %10s %12s %12s" % ("case", "TT/cv", "EE max%", "phiphi%", "P(k) k>0.01", "sigma8 %"))
rows = [("tab_pert (cs2=1, cg=0.4)", "tab_pert")] + [(n, n) for n in cases]
for label, n in rows:
    c = cl(n); l = ref[:, 0]; s = l >= 2; cv = np.sqrt(2 / (2 * l + 1))
    tt = np.max(np.abs((c[:, 1] - ref[:, 1]) / ref[:, 1])[s] / cv[s])
    ee = np.max(np.abs((c[:, 2] - ref[:, 2]) / ref[:, 2])[s]) * 100
    pp = np.max(np.abs((c[:, 5] - ref[:, 5]) / ref[:, 5])[s]) * 100
    p = pk(n); k = refpk[:, 0]; kk = k[(k > p[:, 0].min()) & (k < p[:, 0].max()) & (k > 0.01)]
    r = np.interp(np.log(kk), np.log(p[:, 0]), p[:, 1]) / np.interp(np.log(kk), np.log(k), refpk[:, 1]) - 1
    print("%-16s %10.3f %10.2f %10.2f %12.2e %12.2f" % (label[:16], tt, ee, pp, r.mean(), 50 * r.mean()))

# full-range background agreement between the tabulated fluid and the QCD-DE background
def table(name):
    path = os.path.join(work, "out", name)
    head = [l for l in open(path).read().split("\n") if re.match(r"#\s+1:", l)][0]
    cols = [m.group(2).strip() for m in re.finditer(r"(\d+):(.+?)(?=\s+\d+:|$)", head)]
    d = np.loadtxt(path); return cols, d[np.argsort(d[:, 0])]
ca, a = table("tabbg_00_background.dat")
cb, b = table("qcdbg_00_background.dat")
ia = [i for i, c in enumerate(ca) if "rho_fld" in c][0]
ib = [i for i, c in enumerate(cb) if "rho_qcd" in c][0]
zz = b[:, 0]
Ha = np.interp(zz, a[:, 0], a[:, ca.index("H [1/Mpc]")])
Hb = b[:, cb.index("H [1/Mpc]")]
fa = np.interp(zz, a[:, 0], a[:, ia]) / Ha ** 2
fb = b[:, ib] / Hb ** 2
print("\nFull CLASS range (z = 0 to %.1e):" % zz.max())
print("  max |dH/H| = %.2e" % np.max(np.abs(Ha / Hb - 1)))
print("  max |difference in Omega_DE(z)| = %.2e" % np.max(np.abs(fa - fb)))
m = fb > 1e-30
print("  max |d rho_DE / rho_DE| = %.2e for z where the QCD-DE density exceeds 1e-30 of critical (z < %.0f)"
      % (np.max(np.abs(fa[m] / fb[m] - 1)), zz[m].max()))
m2 = zz < 15
print("  max |d rho_DE / rho_DE| for z < 15 = %.2e" % np.max(np.abs(fa[m2] / fb[m2] - 1)))
