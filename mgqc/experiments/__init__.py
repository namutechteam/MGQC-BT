"""The experiments the manuscript reports, grouped by the question they answer.

`REGISTRY` maps the name used on the command line to the module that runs it,
what it does, and which manuscript items its output feeds. It is the index for
`python -m mgqc list`, and it is the list a reader should work from when
checking that a number in the paper came from somewhere.

Each entry names the module and the function that runs the experiment;
`python -m mgqc run <name>` calls that function with the remaining arguments.
Modules are imported lazily so that a missing optional dependency -- torch, for
instance, which only the classical baselines need -- cannot stop the rest from
running.
"""
import importlib
from collections import namedtuple

Experiment = namedtuple("Experiment", "module entry summary produces manuscript")

REGISTRY = {
    "accuracy": Experiment(
        "baselines", "accuracy_main",
        "Main comparison across the seven configurations",
        "E1_sevenconfig_*.csv, E1_alpha_learned_*.csv",
        "Table 1, Table 2, Figs 2-4, Supp. Tables 6, 7, 12, Supp. Data 1"),
    "classical": Experiment(
        "baselines", "classical_main",
        "Parameter-matched classical message-passing models",
        "E2_classical_equivalent_*.csv",
        "Methods, Supp. Table 5, Fig. 8"),
    "schnet": Experiment(
        "baselines", "schnet_main",
        "SchNet reference at 61,697 parameters",
        "H3_gnn_baseline.csv",
        "Supp. Table 5, Fig. 8"),
    "isomer": Experiment(
        "generalization", "isomer_main",
        "Within-isomer-group discrimination and top-1 selection",
        "H4_stratified.csv",
        "Figs 5, 6, Supp. Table 3"),
    "targets": Experiment(
        "generalization", "targets_main",
        "HOMO-LUMO gap and dipole moment",
        "N1_alt_targets_*.csv",
        "Fig. 7, Supp. Tables 4, 10"),
    "entanglement": Experiment(
        "quantum_structure", "entanglement_main",
        "Concurrence and mutual information of the trained circuit",
        "I1_entanglement_signal.csv",
        "Supp. Table 9"),
    "bondcut": Experiment(
        "grouping", "bondcut_main",
        "Bond-order threshold sensitivity, with a randomized control at each cut",
        "N2_bondcut_sensitivity.csv",
        "Supp. Table 8"),
    "bondcut-cv": Experiment(
        "grouping", "bondcut_cv_main",
        "Threshold selected by cross-validation inside the training split",
        "N2b_cut_cv.csv",
        "Methods, bond typing"),
    "tuning": Experiment(
        "baselines", "tuning_main",
        "Hyper-parameter grids for the circuit and both baselines",
        "N3_baseline_tuning_*.csv",
        "Supp. Table 11"),
    "optimizer": Experiment(
        "quantum_structure", "optimizer_main",
        "Adjoint + Adam against the simultaneous-perturbation loop",
        "G0_training_recovery.csv",
        "Discussion, training"),
    "entangler": Experiment(
        "quantum_structure", "entangler_main",
        "Alternative entangling generators: XY, XXZ, Trotterized transverse field",
        "G1_entangler.csv",
        "Results, Supp. Table 17"),
    "angles": Experiment(
        "quantum_structure", "angles_main",
        "Distribution of the coupling angle actually applied to each bond",
        "R5_theta_summary.csv",
        "Supp. Table 13"),
    "angle-scan": Experiment(
        "quantum_structure", "angle_scan_main",
        "Coupling angle scaled out of the perturbative regime",
        "R7_theta_scan_*.csv",
        "Discussion, Supp. Table 14, Supp. Data 2"),
    "chemistry": Experiment(
        "grouping", "chemistry_main",
        "Do the learned coefficients line up with electronegativity and bond enthalpy",
        "R11_chemistry.csv",
        "Results, Supp. Table 16"),
    "grouping": Experiment(
        "grouping", "grouping_main",
        "What the parameter-sharing groups are indexed by",
        "R10_grouping_*.csv",
        "Results, Supp. Table 15"),
    "grouping-verdict": Experiment(
        "grouping", "verdict_main",
        "Pre-registered decision rules for the grouping experiment",
        "R10_verdict.txt",
        "Results, chemical alignment"),
}


def load(name):
    """Return the function that runs a registry name."""
    if name not in REGISTRY:
        raise KeyError(name)
    experiment = REGISTRY[name]
    return getattr(importlib.import_module(f".{experiment.module}", __name__), experiment.entry)
