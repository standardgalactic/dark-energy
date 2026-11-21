# QCD Vacuum Dark Energy: Technical Assessment

**Status: work in progress.** This repository contains an independent technical assessment of an evolving-dark-energy model motivated by the QCD vacuum, together with the scripts, model modifications, configuration files, and selected results used to investigate its background evolution, observational fits, and perturbative closure. Results and interpretations may change as tests are extended.

The central questions concern whether the proposed vacuum-response prescription is independently identifiable from phenomenological dark-energy parametrizations, whether its approximations hold over the fitted redshift range, and whether its perturbations admit a physically justified closure. Numerical agreement at the homogeneous-background level is not, by itself, evidence of a complete perturbative theory.

## Contents

| Path | Contents |
| --- | --- |
| [`essay/`](essay/) | LaTeX source and PDF of *Identifiability and Perturbation Closure in a QCD Vacuum Dark Energy Model*. |
| [`scripts/qcd_de_ablation.py`](scripts/qcd_de_ablation.py) | Background consistency checks and mimicry/ablation tests; NumPy and SciPy only. |
| [`scripts/late_time_fit.py`](scripts/late_time_fit.py) | Late-time DESI DR2 and DES-Dovekie fits using Cobaya and CLASS. |
| [`scripts/run_perturbation_test.py`](scripts/run_perturbation_test.py) | CPL-fit perturbation test. |
| [`scripts/run_perturbation_test_tabulated.py`](scripts/run_perturbation_test_tabulated.py) | Perturbation test with tabulated QCD-component equation of state, `w(z)`. |
| [`scripts/run_closure_scan.py`](scripts/run_closure_scan.py) | Sound-speed and PPF-parameter scans and full-range background checks. |
| [`results/`](results/) | Selected machine-readable results, run logs, and tabulated `w(z)`. |
| [`patches/class_qcd_fld_unperturbed.patch`](patches/class_qcd_fld_unperturbed.patch) | Initial CLASS patch; superseded by the experimental patch below. |
| [`evidence_run/`](evidence_run/) | Cobaya/PolyChord configuration generator, extraction script, configuration files, and experimental CLASS patch. |

## Dependencies and setup

The background-only ablation script uses Python with NumPy and SciPy. The CLASS-based experiments additionally require a local build of [`dlehdgk/class_qcd`](https://github.com/dlehdgk/class_qcd) and, for the corresponding fits and evidence runs, compatible Cobaya and PolyChord installations. Versions and full environment reproducibility should be recorded as the work progresses.

To prepare the experimental CLASS implementation in a **separate local checkout**, run:

```bash
git clone https://github.com/dlehdgk/class_qcd.git
cd class_qcd
git apply /path/to/dark-energy/evidence_run/class_qcd_experimental.patch
make class
```

Replace `/path/to/dark-energy` with the absolute path to this repository. The patch is experimental, and successful application depends on the upstream revision; check the accompanying [`evidence_run/README.md`](evidence_run/README.md) for run-specific details. The earlier patch in `patches/` is retained for provenance, not as the recommended implementation.

The `evidence_run/cfg/` and `evidence_run/cfg_late/` directories hold configuration inputs. Large PolyChord sampling chains and locally installed third-party source trees are deliberately excluded from version control. Selected derived outputs are retained in `results/`; their presence does not establish independent replication of the full evidence calculation.

## Research status and interpretation

This is an assessment of the model's theoretical and empirical support, not an endorsement of its physical interpretation. In particular, background fitting, perturbation closure, and identification of a QCD mechanism are separate claims requiring separate tests. Any results dependent on an unperturbed-fluid switch or phenomenological sound-speed/PPF prescription should be read with those assumptions explicit.

## License

Original repository code and documentation are made available under the MIT License in [`LICENSE`](LICENSE). Third-party projects, datasets, and software retain their own licenses. Computational dependencies include CLASS, CAMB, Cobaya, and PolyChord.
