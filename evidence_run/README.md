Matched-prior evidence run for QCD-induced dark energy

Contents. make_configs.py writes one Cobaya configuration per model and prior width. extract_evidence.py reads the PolyChord output and reports ln Z, ln B against LCDM, and a prior-volume consistency check. class_qcd_experimental.patch applies to dlehdgk/class_qcd and adds two switches, both off by default: switch_function_qcd = lna (a logistic in ln a for the H-linear model, with lna_qcd and Dlna_qcd) and fld_unperturbed = yes (a fluid that enters the background only).

Setup. Clone the authors' CLASS fork, apply the patch with git apply, build it, and install the python wrapper in the environment that runs Cobaya (pip install --no-build-isolation .). Install Cobaya, PolyChord, and the likelihood data. The configurations use bao.desi_dr2.desi_bao_all, sn.desdovekie, planck_2018_lowl, and a Planck high-ell lite likelihood. ACT DR6 and SPT-3G D1 are added through the --cmb-extra-yaml option, since their package settings differ between installations.

Run. python make_configs.py cfg --data all --packages-path /path/to/cobaya_packages, then cobaya-run cfg/lcdm_paper_all.yaml and the same for each other file, using mpirun for PolyChord. Afterwards python extract_evidence.py cfg chains prints the table.

What was checked offline. Every late-time configuration loads and evaluates with the fork, DESI DR2 and Dovekie. The ln a switch in CLASS agrees with an independent python implementation to about 1e-5 in H(z). The CMB configurations were generated but not evaluated, because the CMB data are not installed in the build environment.

Known differences from the paper. Planck is not truncated at high ell as in the paper's CMB-SPA combination. The lower bound of the QCD width prior is 0.01 instead of 0. The CPL runs use the standard CLASS fluid with PPF perturbations, while the QCD-DE runs treat dark energy as background only, exactly as the authors do.

Overwriting and resuming. The generated files no longer contain force: True. To overwrite an existing run use cobaya-run -f, and to continue an interrupted run use cobaya-run --resume. Remove any stale chains/*.locked file after an interrupted run. For files generated earlier, delete the line starting with force: from the YAML before using --resume.
