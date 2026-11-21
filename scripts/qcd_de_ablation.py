#!/usr/bin/env python3
"""
Background-level ablation of the H-linear vacuum term in QCD-induced dark energy.

Reference: Lee, van de Bruck, Di Valentino, Van Waerbeke, Zhitnitsky (2026),
rho_DE = beta(z) c_H Lambda_QCD^3 H,  H^2 = (8 pi G / 3) rho_t + Hbar H beta(z),
beta(z) = A_q / (1 + exp((z - z_q) / dz_q)),  A_q fixed by beta(z = -1) = 1.

Stage 1: numerical consistency checks of the published equations (Eq. 17, 20, 22, 23).
Stage 2: mimicry test. The published model at its published parameters is treated as
         noiseless synthetic data. Each alternative response is refit (with omega_m
         fixed, i.e. early-time physics unchanged) to the same background observables
         (BAO D_M/r_d and D_H/r_d, the CMB acoustic angle, binned SN distance moduli
         with a free offset). The minimum chi-square measures how distinguishable the
         alternative is under the assumed error model. This is an Asimov-style
         forecast, not a fit to real data, and it does not replace a recomputed
         Bayesian evidence under matched priors.

Models
  qcd_z      published logistic in z, H-linear           (z_q, dz_q)
  qcd_a      same H-linear form, logistic in a           (a_q, da_q)
  qcd_lna    same H-linear form, logistic in ln a        (ln a_q, dln_q)
  switch_noH switch-on without the H factor, logistic in z (z_q, dz_q)
  switch_noH_a / switch_noH_lna   same, logistic in a / ln a (matched pairs to qcd_a, qcd_lna)
  unsupp     H-linear with beta = 1 (no suppression)     ()
  relax      causal response  tau dR/dt + R = H          (tau)
  cpl        w0, wa                                      (w0, wa)
  lcdm       cosmological constant                       ()

Structural note: a switch-on without the H factor has rho_DE non-decreasing in a, so
w_eff <= -1 at all times. It cannot reproduce a quintessence-like branch (w > -1) at
low redshift, and its poor mimicry (switch_noH*) reflects that limit, not only the loss
of the H factor. The H factor supplies the declining part. CPL has no such restriction
and is the fair H-free comparison.

Units: H0 = 1 inside the dimensionless solver; distances in Mpc.
Requires numpy and scipy only.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, differential_evolution, minimize, minimize_scalar
from scipy.special import expit

C_KMS = 299792.458
OMEGA_R_H2 = 4.18e-5          # photons plus three massless neutrinos
Z_STAR = 1089.9
BAO_Z = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])

# Published best fit, CMB-SPA + DD + DESI (Lee et al. 2026)
REF = dict(h0=67.74, om=0.3109, zq=0.68, dzq=0.82)


# ----------------------------------------------------------------------------
# Background pieces
# ----------------------------------------------------------------------------
class Cosmo:
    """Flat background with omega_m = Om h^2 held fixed when h varies."""

    def __init__(self, h, omega_m_h2):
        self.h = h
        self.om = omega_m_h2 / h ** 2
        self.orad = OMEGA_R_H2 / h ** 2
        self.ode = 1.0 - self.om - self.orad

    def omega_t(self, z):
        return self.om * (1 + z) ** 3 + self.orad * (1 + z) ** 4


def logistic_z(z, zq, dzq):
    amp = 1.0 + np.exp((-1.0 - zq) / dzq)
    return amp * expit(-(z - zq) / dzq), amp


def beta_shape(kind, z, p):
    if kind == "z":
        return logistic_z(z, p[0], p[1])[0]
    a = 1.0 / (1.0 + z)
    if kind == "a":
        return expit((a - p[0]) / p[1])
    if kind == "lna":
        return expit((np.log(a) - p[0]) / p[1])
    if kind == "one":
        return np.ones_like(z)
    raise ValueError(kind)


def E_hlinear(cos, z, kind, p):
    """H^2 = H0^2 Omega_t + Hbar H beta, closed by flatness at z = 0."""
    beta = beta_shape(kind, z, p)
    beta0 = float(beta_shape(kind, np.array([0.0]), p)[0])
    hb = cos.ode / beta0
    x = 0.5 * hb * beta
    return x + np.sqrt(x * x + cos.omega_t(z)), dict(hbar_over_h0=hb, beta0=beta0)


def E_switch_noH(cos, z, p, kind="z"):
    """rho_DE proportional to beta only (no H factor); same beta families as E_hlinear."""
    beta = beta_shape(kind, z, p)
    beta0 = float(beta_shape(kind, np.array([0.0]), p)[0])
    return np.sqrt(cos.omega_t(z) + cos.ode * beta / beta0), {}


def E_cpl(cos, z, p):
    w0, wa = p
    f = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(cos.omega_t(z) + cos.ode * f), {}


def E_lcdm(cos, z, p):
    return np.sqrt(cos.omega_t(z) + cos.ode), {}


# ----------------------------------------------------------------------------
# Causal relaxation model: tau dR/dt + R = H, rho_DE proportional to R
#   Units H0 = 1, x = ln a:  dR/dx = (E - R) / (tau E),  E^2 = Omega_t + hb R
#   hb is found by shooting so that E(a = 1) = 1.
# ----------------------------------------------------------------------------
Z_INIT = 1.0e4


def relax_solve(cos, tau, ic="zero"):
    x0 = -np.log(1.0 + Z_INIT)

    def omega_t_x(x):
        a = np.exp(x)
        return cos.om * a ** -3 + cos.orad * a ** -4

    def run(hb):
        if ic == "zero":
            y0 = 0.0
        else:
            y0 = 0.5 * hb + np.sqrt(0.25 * hb * hb + omega_t_x(x0))

        def rhs(x, y):
            e = np.sqrt(omega_t_x(x) + hb * max(y[0], 0.0))
            return [(e - y[0]) / (tau * e)]

        return solve_ivp(rhs, (x0, 0.0), [y0], method="LSODA", rtol=1e-8,
                         atol=1e-12, dense_output=True)

    def resid(hb):
        return hb * run(hb).y[0, -1] - cos.ode

    lo, hi = 1e-8, 1e3
    try:
        if resid(lo) * resid(hi) > 0:
            return None
        hb = brentq(resid, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=100)
    except Exception:
        return None
    sol = run(hb)
    return hb, sol


def E_relax(cos, z, p, ic="zero"):
    out = relax_solve(cos, p[0], ic)
    if out is None:
        return None, {}
    hb, sol = out
    r = np.maximum(sol.sol(-np.log(1.0 + z))[0], 0.0)
    return np.sqrt(cos.omega_t(z) + hb * r), dict(hbar_over_h0=hb)


MODELS = {
    # name: (parameter names, bounds for shape parameters, builder)
    "qcd_z": (["zq", "dzq"], [(0.0, 3.0), (0.05, 3.0)],
              lambda c, z, p: E_hlinear(c, z, "z", p)),
    "qcd_a": (["aq", "daq"], [(0.1, 1.5), (0.02, 1.5)],
              lambda c, z, p: E_hlinear(c, z, "a", p)),
    "qcd_lna": (["lnaq", "dlnq"], [(-2.5, 0.5), (0.05, 2.0)],
                lambda c, z, p: E_hlinear(c, z, "lna", p)),
    "switch_noH": (["zq", "dzq"], [(0.0, 3.0), (0.05, 3.0)],
                   lambda c, z, p: E_switch_noH(c, z, p, "z")),
    "switch_noH_a": (["aq", "daq"], [(0.1, 1.5), (0.02, 1.5)],
                     lambda c, z, p: E_switch_noH(c, z, p, "a")),
    "switch_noH_lna": (["lnaq", "dlnq"], [(-2.5, 0.5), (0.05, 2.0)],
                       lambda c, z, p: E_switch_noH(c, z, p, "lna")),
    "unsupp": ([], [], lambda c, z, p: E_hlinear(c, z, "one", p)),
    "relax": (["tau"], [(-1.5, 1.5)], None),   # bounds in log10(tau H0)
    "cpl": (["w0", "wa"], [(-2.0, 0.0), (-3.0, 2.0)],
            lambda c, z, p: E_cpl(c, z, p)),
    "lcdm": ([], [], lambda c, z, p: E_lcdm(c, z, p)),
}


# ----------------------------------------------------------------------------
# Observables and chi-square
# ----------------------------------------------------------------------------
def make_grid():
    return np.concatenate([np.linspace(0.0, 5.0, 4001),
                           np.geomspace(5.0, Z_STAR, 1500)[1:]])


class Proxy:
    def __init__(self, args):
        self.rd = 147.0
        self.sig_dm = args.sig_dm
        self.sig_dh = args.sig_dh
        self.sig_cmb = args.sig_cmb
        self.sn_z = np.geomspace(0.01, 2.2, 40)
        self.sig_sn = args.sig_sn
        self.grid = make_grid()

    def observables(self, E, h):
        z = self.grid
        d_h = C_KMS / (100.0 * h)
        dm = d_h * np.concatenate([[0.0], np.cumsum(
            0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(z))])
        dm_bao = np.interp(BAO_Z, z, dm)
        dh_bao = d_h / np.interp(BAO_Z, z, E)
        dm_star = dm[-1]
        mu = 5.0 * np.log10((1 + self.sn_z) * np.interp(self.sn_z, z, dm)) + 25.0
        return dict(dm=dm_bao / self.rd, dh=dh_bao / self.rd, star=dm_star, mu=mu)

    def chi2(self, obs, ref):
        c = np.sum(((obs["dm"] - ref["dm"]) / (self.sig_dm * ref["dm"])) ** 2)
        c += np.sum(((obs["dh"] - ref["dh"]) / (self.sig_dh * ref["dh"])) ** 2)
        c += ((obs["star"] - ref["star"]) / (self.sig_cmb * ref["star"])) ** 2
        d = obs["mu"] - ref["mu"]
        d = d - d.mean()
        c += np.sum((d / self.sig_sn) ** 2)
        return float(c)


def model_E(name, h, shape, omega_m_h2, z):
    cos = Cosmo(h, omega_m_h2)
    if cos.ode <= 0:
        return None, {}
    if name == "relax":
        return E_relax(cos, z, [10.0 ** shape[0]])
    return MODELS[name][2](cos, z, shape)


def objective_factory(name, proxy, ref_obs, omega_m_h2):
    def f(theta):
        E, _ = model_E(name, theta[0], theta[1:], omega_m_h2, proxy.grid)
        if E is None or not np.all(np.isfinite(E)):
            return 1e9
        return proxy.chi2(proxy.observables(E, theta[0]), ref_obs)
    return f


def fit_model(name, proxy, ref_obs, omega_m_h2, seed=1, verbose=True):
    pnames, bounds, _ = MODELS[name]
    f = objective_factory(name, proxy, ref_obs, omega_m_h2)
    hb = (0.60, 0.75)
    if name == "relax":
        best = None
        for lt in np.linspace(bounds[0][0], bounds[0][1], 13):
            r = minimize_scalar(lambda h: f([h, lt]), bounds=hb, method="bounded",
                                options=dict(xatol=1e-5))
            if best is None or r.fun < best[0]:
                best = (r.fun, [r.x, lt])
        r = minimize(f, best[1], method="Nelder-Mead",
                     options=dict(xatol=1e-6, fatol=1e-8, maxiter=400))
        theta = r.x if r.fun < best[0] else np.array(best[1])
        val = min(r.fun, best[0])
        # report tau, not log10(tau)
        return val, [theta[0], 10.0 ** theta[1]]
    allb = [hb] + list(bounds)
    r = differential_evolution(f, allb, seed=seed, popsize=18, maxiter=250,
                               tol=1e-9, polish=True)
    return float(r.fun), list(r.x)


# ----------------------------------------------------------------------------
# Diagnostics on a fine z grid
# ----------------------------------------------------------------------------
def diagnostics(E, cos, z):
    """Return eps_H = -dlnH/dlna, Eq. 14 combination, w_eff and DE fraction."""
    lnE = np.log(E)
    dlnE_dz = np.gradient(lnE, z)
    eps = (1 + z) * dlnE_dz                       # |Hdot|/H^2 in an expanding model
    # (3/2)(H/Hbar)|1+w_tot| with 1+w_tot = (2/3) eps  ->  (H/Hbar) eps
    rho = E ** 2 - cos.omega_t(z)                 # proportional to rho_DE
    ok = rho > 0
    w = np.full_like(z, np.nan)
    dlnrho_dz = np.gradient(np.log(np.where(ok, rho, np.nan)), z)
    w[ok] = -1.0 + (1.0 / 3.0) * (1 + z[ok]) * dlnrho_dz[ok]
    frac = rho / E ** 2
    return eps, w, frac


def phantom_crossing(z, w):
    s = w + 1.0
    for i in range(1, len(z)):
        if np.isfinite(s[i]) and np.isfinite(s[i - 1]) and s[i - 1] * s[i] < 0:
            return float(z[i - 1] - s[i - 1] * (z[i] - z[i - 1]) / (s[i] - s[i - 1]))
    return None


# ----------------------------------------------------------------------------
# Stage 1: checks of the published equations
# ----------------------------------------------------------------------------
def published_checks():
    out = {}
    h = REF["h0"] / 100.0
    cos = Cosmo(h, REF["om"] * h ** 2)
    z = np.linspace(0.0, 3.0, 30001)
    zq, dz = REF["zq"], REF["dzq"]
    beta, amp = logistic_z(z, zq, dz)
    out["A_q"] = float(amp)
    out["beta_at_z_minus1"] = float(logistic_z(np.array([-1.0]), zq, dz)[0][0])
    E, info = E_hlinear(cos, z, "z", [zq, dz])
    hb = info["hbar_over_h0"]
    out["beta0"] = info["beta0"]
    out["Hbar_km_s_Mpc"] = hb * REF["h0"]
    out["Hbar_note"] = "paper quotes 58.8 (+3.5/-4.3) at its own best-fit chain point"

    # Eq 17 residual of the quadratic H^2 = H0^2 Omega_t + Hbar H beta
    res = E ** 2 - cos.omega_t(z) - hb * E * beta
    out["eq16_residual_max"] = float(np.max(np.abs(res)))
    out["closure_E_at_z0"] = float(E[0])

    # Eq 23 derivative: analytic with and without (1+z), against finite difference
    dbeta_dlna_num = -(1 + z) * np.gradient(beta, z)
    correct = (1 + z) * beta * (amp - beta) / (amp * dz)
    literal = beta * (amp - beta) / (amp * dz)
    sl = slice(5, -5)
    # Eq. 23 is written for a*dbeta/dlna. The factor a cancels the (1+z).
    a_dbeta_num = dbeta_dlna_num / (1 + z)
    out["eq23_vs_a_dbeta_dlna_max_rel_err"] = float(np.max(np.abs(
        literal[sl] - a_dbeta_num[sl]) / np.maximum(np.abs(a_dbeta_num[sl]), 1e-12)))
    out["dbeta_dlna_with_(1+z)_max_rel_err"] = float(np.max(np.abs(
        correct[sl] - dbeta_dlna_num[sl]) / np.maximum(np.abs(dbeta_dlna_num[sl]), 1e-12)))
    out["eq23_expression_misread_as_dbeta_dlna_max_rel_err"] = float(np.max(np.abs(
        literal[sl] - dbeta_dlna_num[sl]) / np.maximum(np.abs(dbeta_dlna_num[sl]), 1e-12)))

    # Eq 20 against the numerical Hdot, with each form of beta'
    src = 3 * cos.om * (1 + z) ** 3 + 4 * cos.orad * (1 + z) ** 4  # 8piG(rho+P)/H0^2
    hdot_num = -E * (1 + z) * np.gradient(E, z)
    for tag, bp in (("dbeta_dlna_with_(1+z)", correct), ("if_Eq23_misread_as_dbeta_dlna", literal)):
        hdot = E ** 2 / (2 * E ** 2 - hb * E * beta) * (hb * E * bp - src)
        out["eq20_vs_numeric_Hdot_max_rel_err_" + tag] = float(np.max(np.abs(
            hdot[sl] - hdot_num[sl]) / np.maximum(np.abs(hdot_num[sl]), 1e-12)))

    # Identity between the paper's two adiabatic criteria
    eps, w_num, frac = diagnostics(E, cos, z)
    eq14 = (E / hb) * eps
    wtot = -1.0 + (2.0 / 3.0) * eps
    eq14_direct = 1.5 * (E / hb) * np.abs(1 + wtot)
    out["eq14_identity_max_abs_err"] = float(np.max(np.abs(eq14 - eq14_direct)))
    for zz in (0.0, 0.3, 0.67, 1.0, 2.0):
        i = int(np.argmin(np.abs(z - zz)))
        out["eps_H_at_z=%.2f" % zz] = float(eps[i])
        out["eq14_combination_at_z=%.2f" % zz] = float(eq14[i])
        out["H_over_Hbar_at_z=%.2f" % zz] = float(E[i] / hb)

    # w_eff: analytic with each form of beta', and numeric from rho_DE
    dlnE_dlna = -(1 + z) * np.gradient(np.log(E), z)
    w_c = -1.0 - (dlnE_dlna + correct / beta) / 3.0
    w_l = -1.0 - (dlnE_dlna + literal / beta) / 3.0
    out["w_eff_analytic_vs_numeric_max_abs_err"] = float(np.nanmax(np.abs(w_c[10:-10] - w_num[10:-10])))
    out["phantom_crossing_z_correct"] = phantom_crossing(z, w_c)
    out["phantom_crossing_z_if_Eq23_misread_as_dbeta_dlna"] = phantom_crossing(z, w_l)
    m = z <= 2.0
    out["w_eff_max_abs_difference_literal_vs_correct_z<=2"] = float(np.max(np.abs(w_c[m] - w_l[m])))
    out["w0_correct"] = float(w_c[0])
    out["w0_if_Eq23_misread"] = float(w_l[0])
    # Form used in the authors' CLASS code: H' = a Hdot, with adbeta_dloga = a dbeta/dlna
    a = 1.0 / (1 + z)
    hp_code = 1.0 / (2.0 - hb * beta / E) * (hb * E * literal - a * src)
    hp_num = a * hdot_num
    out["class_qcd_form_H_prime_vs_numeric_max_rel_err"] = float(np.max(np.abs(
        hp_code[sl] - hp_num[sl]) / np.maximum(np.abs(hp_num[sl]), 1e-12)))
    return out


# ----------------------------------------------------------------------------
# Stage 2: mimicry and curves
# ----------------------------------------------------------------------------
def curve_rows(name, h, shape, omega_m_h2, zc):
    E, info = model_E(name, h, shape, omega_m_h2, zc)
    if E is None:
        return [], None
    cos = Cosmo(h, omega_m_h2)
    eps, w, frac = diagnostics(E, cos, zc)
    d_h = C_KMS / (100.0 * h)
    dm = d_h * np.concatenate([[0.0], np.cumsum(
        0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zc))])
    hb = info.get("hbar_over_h0")
    rows = []
    for i in range(0, len(zc), 20):
        rows.append(dict(model=name, z=float(zc[i]), H_km_s_Mpc=float(100 * h * E[i]),
                         eps_H=float(eps[i]),
                         eq14=float(E[i] / hb * eps[i]) if hb else "",
                         w_eff=float(w[i]) if np.isfinite(w[i]) else "",
                         DE_fraction=float(frac[i]), D_M_Mpc=float(dm[i]),
                         D_H_Mpc=float(d_h / E[i])))
    return rows, phantom_crossing(zc, w)


def early_fraction(name, h, shape, omega_m_h2, ic="zero"):
    zs = np.array([0.0, 1.0, 3.0, 10.0, 100.0, Z_STAR])
    cos = Cosmo(h, omega_m_h2)
    if name == "relax":
        E, _ = E_relax(cos, zs, [shape[0]], ic)
    else:
        E, _ = model_E(name, h, shape, omega_m_h2, zs)
    if E is None:
        return None
    return {("z=%g" % z): float((E[i] ** 2 - cos.omega_t(z)) / E[i] ** 2)
            for i, z in enumerate(zs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="qcd_de_ablation_out", help="output directory")
    ap.add_argument("--sig-dm", type=float, default=0.008, help="fractional error on D_M/r_d")
    ap.add_argument("--sig-dh", type=float, default=0.018, help="fractional error on D_H/r_d")
    ap.add_argument("--sig-cmb", type=float, default=3e-4, help="fractional error on D_M(z*)")
    ap.add_argument("--sig-sn", type=float, default=0.02, help="mag error per SN bin (40 bins)")
    ap.add_argument("--skip-relax", action="store_true")
    ap.add_argument("--checks-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    results = dict(settings=vars(args), reference=REF)
    results["published_checks"] = published_checks()
    print("== Stage 1: checks of the published equations ==")
    for k, v in results["published_checks"].items():
        print("  %-58s %s" % (k, v))
    if args.checks_only:
        with open(os.path.join(args.out, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        return

    proxy = Proxy(args)
    h_ref = REF["h0"] / 100.0
    omh2 = REF["om"] * h_ref ** 2
    E_ref, _ = model_E("qcd_z", h_ref, [REF["zq"], REF["dzq"]], omh2, proxy.grid)
    ref_obs = proxy.observables(E_ref, h_ref)

    print("\n== Stage 2: mimicry of the published model (synthetic, noiseless) ==")
    names = ["lcdm", "unsupp", "qcd_z", "qcd_a", "qcd_lna", "switch_noH",
             "switch_noH_a", "switch_noH_lna", "cpl"]
    if not args.skip_relax:
        names.append("relax")
    zc = np.linspace(0.0, 3.0, 601)
    fits, rows, crossing = {}, [], {}
    r0, _ = curve_rows("qcd_z", h_ref, [REF["zq"], REF["dzq"]], omh2, zc)
    rows += [dict(r, model="reference") for r in r0]
    for n in names:
        t1 = time.time()
        chi2, theta = fit_model(n, proxy, ref_obs, omh2)
        shape = theta[1:]
        entry = dict(chi2_min=chi2, h0_best=100 * theta[0],
                     params=dict(zip(MODELS[n][0], [float(s) for s in shape])),
                     n_shape_params=len(MODELS[n][0]), seconds=round(time.time() - t1, 1))
        if n == "relax":
            entry["params"] = {"tau_times_H0": float(theta[1])}
            shape_for_curve = [np.log10(theta[1])]
        else:
            shape_for_curve = shape
        entry["DE_fraction_vs_z"] = early_fraction(
            n, theta[0], [np.log10(theta[1])] if n == "relax" else shape, omh2)
        rr, pc = curve_rows(n, theta[0], shape_for_curve, omh2, zc)
        rows += rr
        entry["phantom_crossing_z"] = pc
        fits[n] = entry
        print("  %-11s chi2_min = %9.3f   h0 = %6.2f   %s   (%.0fs)" % (
            n, chi2, entry["h0_best"], entry["params"], entry["seconds"]))
    results["mimicry_fits"] = fits

    cal = fits["lcdm"]["chi2_min"]
    results["calibration"] = dict(
        proxy_expected_delta_chi2_of_reference_vs_lcdm=cal,
        published_delta_chi2_CMBSPA_DD_DESI=-11.21,
        note="If these differ by a large factor, the proxy error model has different "
             "information content from the real likelihoods; scale chi2 values accordingly.")
    print("\n  Proxy chi2 of LCDM against the reference: %.2f (published Delta chi2: 11.21)" % cal)

    if "relax" in fits:
        tau = fits["relax"]["params"]["tau_times_H0"]
        h = fits["relax"]["h0_best"] / 100.0
        results["relax_initial_condition_sensitivity"] = dict(
            zero_initial_R=early_fraction("relax", h, [tau], omh2, "zero"),
            tracking_initial_R=early_fraction("relax", h, [tau], omh2, "tracking"),
            note="DE fraction rho_DE/rho_total; the relaxation model suppresses early DE only "
                 "through its initial condition, which is an added assumption.")
        print("\n  Relaxation model, DE fraction vs initial condition:")
        print("   zero initial R :", results["relax_initial_condition_sensitivity"]["zero_initial_R"])
        print("   tracking R     :", results["relax_initial_condition_sensitivity"]["tracking_initial_R"])

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(args.out, "curves.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\nWrote results.json and curves.csv to %s  (total %.0fs)" % (args.out, time.time() - t0))


if __name__ == "__main__":
    main()
