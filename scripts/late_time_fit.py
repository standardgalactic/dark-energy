#!/usr/bin/env python3
"""
Late-time check of the published QCD-DE model with real data: DESI DR2 BAO plus DES-Dovekie
supernovae, using the authors' CLASS fork through Cobaya. Compares best-fit chi-square for
LCDM, CPL (w0, wa) and QCD-DE (logistic in z) with the paper's DD+DESI row of Table 4
(Delta chi2 = -7.19 for both w0wa and QCD-DE).

Assumptions that are mine, not the paper's: a Gaussian BBN prior on omega_b, free omega_cdm and
H0, no CMB, n_s and A_s fixed (irrelevant for background-only quantities). The chi-square
reported is the likelihood part only.

Usage: late_time_fit.py MODEL [DATA_DIR]      MODEL in lcdm | cpl | qcd
"""
import json
import sys
import numpy as np
from scipy.optimize import differential_evolution, minimize
from cobaya.model import get_model

model_name = sys.argv[1]
data = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/cobaya_packages/data"

extra = {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611,
         "A_s": 2.1e-9, "n_s": 0.9649, "tau_reio": 0.051}
params = {
    "H0": {"prior": {"min": 50, "max": 90}, "ref": 67.7},
    "omega_b": {"prior": {"dist": "norm", "loc": 0.02218, "scale": 0.00055}, "ref": 0.02218},
    "omega_cdm": {"prior": {"min": 0.05, "max": 0.20}, "ref": 0.12},
}
if model_name == "cpl":
    extra.update({"Omega_Lambda": 0, "Omega_scf": 0, "fluid_equation_of_state": "CLP",
                  "use_ppf": "yes", "cs2_fld": 1})
    params["w0_fld"] = {"prior": {"min": -3, "max": 1}, "ref": -0.9}
    params["wa_fld"] = {"prior": {"min": -3, "max": 2}, "ref": -0.3}
elif model_name == "qcd":
    extra.update({"Omega_fld": 0, "Omega_scf": 0, "Omega_Lambda": 0, "Omega_qcd": -1,
                  "switch_function_qcd": "exp"})
    params["z_qcd"] = {"prior": {"min": 0.0, "max": 3.0}, "ref": 0.7}
    params["Dz_qcd"] = {"prior": {"min": 0.05, "max": 3.0}, "ref": 0.8}

info = {
    "theory": {"classy": {"extra_args": dict(extra, output="")}},
    "likelihood": {
        "bao.desi_dr2.desi_bao_all": None,
        "sn.desdovekie": None,
    },
    "params": params,
    "packages_path": "/home/claude/cobaya_packages",
}
model = get_model(info)
names = list(model.parameterization.sampled_params())
lo = np.array([model.prior.bounds(confidence=1.0)[i][0] for i in range(len(names))])
hi = np.array([model.prior.bounds(confidence=1.0)[i][1] for i in range(len(names))])
# the BBN Gaussian prior has infinite bounds; restrict the search box to +-5 sigma
for i, n in enumerate(names):
    if n == "omega_b":
        lo[i], hi[i] = 0.02218 - 5 * 0.00055, 0.02218 + 5 * 0.00055


def chi2(x):
    try:
        lp = model.logposterior(dict(zip(names, x)))
        v = -2.0 * float(lp.logpost)
        return v if np.isfinite(v) else 1e9
    except Exception:
        return 1e9


if len(names) == 3:
    res = minimize(chi2, [67.7, 0.02218, 0.12], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-6, maxiter=2000))
else:
    res = differential_evolution(chi2, list(zip(lo, hi)), seed=3, popsize=12, maxiter=120,
                                 tol=1e-8, polish=False)
    res2 = minimize(chi2, res.x, method="Nelder-Mead",
                    options=dict(xatol=1e-7, fatol=1e-8, maxiter=3000))
    res = res2 if res2.fun < res.fun else res
per = dict(zip(names, [float(v) for v in res.x]))
ll = model.loglikes(per)[0]
out = dict(model=model_name, chi2_total=float(-2 * np.sum(ll)), minus2_log_posterior=float(res.fun), best_fit=per,
           chi2_by_likelihood={k: -2 * float(v) for k, v in zip(model.likelihood, ll)})
print(json.dumps(out, indent=2))
json.dump(out, open("/home/claude/out/late_time_%s.json" % model_name, "w"), indent=2)
