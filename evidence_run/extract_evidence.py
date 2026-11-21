#!/usr/bin/env python3
"""
Summarize PolyChord evidence runs produced from make_configs.py.

For each config it reads log(Z) +/- error from the PolyChord .stats file, reports
ln B against the LCDM run, and compares the shift in ln Z between the "paper" and "narrow"
prior boxes with the pure prior-volume expectation ln(V_narrow / V_paper). If the posterior
is contained in both boxes, ln Z should rise by exactly that amount, so any remainder
indicates that the narrow box truncates the posterior or that the sampling error is large.

Usage: extract_evidence.py CONFIG_DIR CHAIN_DIR
"""
import glob
import math
import os
import re
import sys

import yaml

cfg_dir, chain_dir = sys.argv[1], sys.argv[2]
pat = re.compile(r"log\(Z\)\s*=\s*([-+0-9.eE]+)\s*\+/-\s*([-+0-9.eE]+)")


def read_stats(outprefix):
    base = os.path.basename(outprefix)
    for p in glob.glob(os.path.join(chain_dir, "**", base + "*.stats"), recursive=True):
        txt = open(p).read()
        m = pat.search(txt)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def box_volume(info):
    v = 0.0
    for name, p in info["params"].items():
        if isinstance(p, dict) and "prior" in p and "min" in p["prior"] and name not in (
                "omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio"):
            v += math.log(p["prior"]["max"] - p["prior"]["min"])
    return v


rows = {}
for f in sorted(glob.glob(os.path.join(cfg_dir, "*.yaml"))):
    info = yaml.safe_load(open(f))
    key = os.path.basename(f)[:-5]
    st = read_stats(info["output"])
    rows[key] = dict(stats=st, logvol=box_volume(info))

ref = next((v["stats"] for k, v in rows.items() if k.startswith("lcdm") and v["stats"]), None)
print("%-24s %12s %8s %10s" % ("run", "ln Z", "+/-", "ln B"))
for k, v in rows.items():
    if v["stats"] is None:
        print("%-24s %12s" % (k, "missing"))
        continue
    lz, err = v["stats"]
    lnb = "" if ref is None else "%+.2f +/- %.2f" % (lz - ref[0], math.hypot(err, ref[1]))
    print("%-24s %12.2f %8.2f %s" % (k, lz, err, lnb))
print("\nPrior-volume check (narrow minus paper):")
for model in sorted({k.rsplit("_", 2)[0] for k in rows if not k.startswith("lcdm")}):
    pk, nk = [k for k in rows if k.startswith(model + "_paper")], [k for k in rows if k.startswith(model + "_narrow")]
    if pk and nk and rows[pk[0]]["stats"] and rows[nk[0]]["stats"]:
        d = rows[nk[0]]["stats"][0] - rows[pk[0]]["stats"][0]
        e = rows[nk[0]]["logvol"] * -1 + rows[pk[0]]["logvol"]
        print("  %-10s delta ln Z = %+.2f ; expected from volume = %+.2f" % (model, d, e))
