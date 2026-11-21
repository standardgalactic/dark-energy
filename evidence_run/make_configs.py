#!/usr/bin/env python3
"""
Generate matched Cobaya configurations for a nested-sampling evidence comparison of
LCDM, CPL, the published QCD-DE (logistic in z) and an H-linear variant (logistic in ln a).

Every model uses the same theory block, base-parameter priors, likelihoods and sampler.
Only the dark-energy parameters and their priors differ. Prior widths for the transition
parameters come in two variants ("paper" and "narrow") so the dependence of ln B on prior
volume can be read off directly.

Requirements: the authors' CLASS fork (https://github.com/dlehdgk/class_qcd) with
class_qcd_experimental.patch applied, built and installed as the python module `classy`.
The CMB block is a template: Planck low-ell and a Planck high-ell lite likelihood are
included so the file runs on a stock installation. For ACT DR6 and SPT-3G D1 pass a YAML
snippet with --cmb-extra-yaml (a file whose top-level key is `likelihood:`); it is merged in.
The paper's "CMB-SPA" combination truncates Planck at high ell, which these defaults do not.

Usage:
  make_configs.py OUTDIR [--data late|cmb|all] [--sn dovekie|pantheonplus]
                  [--cmb-extra-yaml FILE] [--packages-path DIR] [--nlive-per-dim 25]
"""
import argparse
import os
import yaml

BASE_PARAMS = {
    "omega_b": {"prior": {"min": 0.019, "max": 0.026}, "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0002},
                "proposal": 0.0001, "latex": r"\Omega_\mathrm{b} h^2"},
    "omega_cdm": {"prior": {"min": 0.09, "max": 0.15}, "ref": {"dist": "norm", "loc": 0.120, "scale": 0.002},
                  "proposal": 0.001, "latex": r"\Omega_\mathrm{c} h^2"},
    "H0": {"prior": {"min": 55, "max": 80}, "ref": {"dist": "norm", "loc": 67.5, "scale": 1.0},
           "proposal": 0.5, "latex": "H_0"},
    "logA": {"prior": {"min": 2.5, "max": 3.5}, "ref": {"dist": "norm", "loc": 3.05, "scale": 0.01},
             "proposal": 0.01, "drop": True, "latex": r"\log(10^{10} A_\mathrm{s})"},
    "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": "A_\\mathrm{s}"},
    "n_s": {"prior": {"min": 0.88, "max": 1.06}, "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
            "proposal": 0.002, "latex": "n_\\mathrm{s}"},
    "tau_reio": {"prior": {"dist": "norm", "loc": 0.051, "scale": 0.006},
                 "ref": {"dist": "norm", "loc": 0.051, "scale": 0.005}, "proposal": 0.003,
                 "latex": r"\tau_\mathrm{reio}"},
}

CLASS_COMMON = {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611,
                "non linear": "halofit"}

# transition-parameter priors: (paper-like, narrow). Narrow boxes still contain the published posteriors.
DE_MODELS = {
    "lcdm": dict(extra={}, params={}),
    "cpl": dict(
        extra={"Omega_Lambda": 0, "Omega_scf": 0, "fluid_equation_of_state": "CLP",
               "use_ppf": "yes", "c_gamma_over_c_fld": 0.4, "cs2_fld": 1},
        params={
            "paper": {"w0_fld": (-3.0, 1.0, -0.9), "wa_fld": (-3.0, 2.0, -0.3)},
            "narrow": {"w0_fld": (-1.5, -0.5, -0.9), "wa_fld": (-2.0, 1.0, -0.3)},
        }),
    "qcd_z": dict(
        extra={"Omega_fld": 0, "Omega_scf": 0, "Omega_Lambda": 0, "Omega_qcd": -1,
               "switch_function_qcd": "exp"},
        params={
            "paper": {"z_qcd": (0.0, 3.0, 0.7), "Dz_qcd": (0.01, 3.0, 0.8)},
            "narrow": {"z_qcd": (0.0, 2.0, 0.7), "Dz_qcd": (0.2, 2.0, 0.8)},
        }),
    "qcd_lna": dict(
        extra={"Omega_fld": 0, "Omega_scf": 0, "Omega_Lambda": 0, "Omega_qcd": -1,
               "switch_function_qcd": "lna"},
        params={
            "paper": {"lna_qcd": (-2.5, 0.5, -0.7), "Dlna_qcd": (0.05, 2.0, 0.33)},
            "narrow": {"lna_qcd": (-1.5, 0.0, -0.7), "Dlna_qcd": (0.1, 1.0, 0.33)},
        }),
}


def likelihoods(data, sn, cmb_extra):
    like = {}
    if data in ("late", "all"):
        like["bao.desi_dr2.desi_bao_all"] = None
        like["sn.desdovekie" if sn == "dovekie" else "sn.pantheonplus"] = None
    if data in ("cmb", "all"):
        like["planck_2018_lowl.TT"] = None
        like["planck_2018_lowl.EE"] = None
        like["planck_2018_highl_plik.TTTEEE_lite_native"] = None
        like["planck_2018_lensing.native"] = None
        if cmb_extra:
            extra = yaml.safe_load(open(cmb_extra))
            like.update(extra.get("likelihood", {}))
    return like


def build(model, width, args):
    spec = DE_MODELS[model]
    params = {k: dict(v) for k, v in BASE_PARAMS.items()}
    if args.data == "late":
        # background-only data: perturbation parameters are irrelevant, fix them
        for k in ("logA", "A_s", "n_s", "tau_reio"):
            params.pop(k)
        params["A_s"] = 2.1e-9
        params["n_s"] = 0.9649
        params["tau_reio"] = 0.051
        params["omega_b"] = {"prior": {"dist": "norm", "loc": 0.02218, "scale": 0.00055},
                             "ref": 0.02218, "proposal": 0.0005, "latex": r"\Omega_\mathrm{b} h^2"}
    if model != "lcdm":
        for name, (lo, hi, ref) in spec["params"][width].items():
            params[name] = {"prior": {"min": lo, "max": hi}, "ref": ref,
                            "proposal": 0.05, "latex": name}
    extra = dict(CLASS_COMMON, **spec["extra"])
    if args.data == "late":
        extra.pop("non linear", None)
        extra["output"] = ""
    else:
        extra.update({"output": "tCl pCl lCl", "lensing": "yes", "l_max_scalars": 5000})
    n_dim = sum(1 for v in params.values() if isinstance(v, dict) and "prior" in v)
    info = {
        "theory": {"classy": {"extra_args": extra}},
        "likelihood": likelihoods(args.data, args.sn, args.cmb_extra_yaml),
        "params": params,
        "sampler": {"polychord": {"nlive": args.nlive_per_dim * n_dim,
                                  "num_repeats": 5 * n_dim, "precision_criterion": 0.001,
                                  "do_clustering": True}},
        "packages_path": args.packages_path,
        "output": "chains/%s_%s_%s" % (model, width, args.data),
    }
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--data", default="all", choices=["late", "cmb", "all"])
    ap.add_argument("--sn", default="dovekie", choices=["dovekie", "pantheonplus"])
    ap.add_argument("--cmb-extra-yaml", default=None)
    ap.add_argument("--packages-path", default="/path/to/cobaya_packages")
    ap.add_argument("--nlive-per-dim", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    written = []
    for model in DE_MODELS:
        for width in (("paper",) if model == "lcdm" else ("paper", "narrow")):
            info = build(model, width, args)
            path = os.path.join(args.outdir, "%s_%s_%s.yaml" % (model, width, args.data))
            with open(path, "w") as fh:
                yaml.safe_dump(info, fh, sort_keys=False)
            written.append(path)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
