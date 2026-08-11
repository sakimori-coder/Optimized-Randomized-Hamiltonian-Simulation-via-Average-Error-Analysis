r"""Compare qDRIFT decompositions for generated molecular Hamiltonians.

Molecular-Hamiltonian generation lives in :mod:`hamiltonians.chemistry`.
This module contains only the qDRIFT experiment and command-line interface.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence

import numpy as np

from grouping import GroupingMethod, GroupNormMethod
from hamiltonians.chemistry import (
    H2_STO3G_JW,
    MOLECULAR_HAMILTONIANS,
    MolecularHamiltonianPreset,
)
from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from main import QDriftComparison, compare_pauli_and_grouped_qdrift
from diamond_distance.variance import BuiltInMethod as VarianceMethod


VARIANCE_METHODS: tuple[VarianceMethod, ...] = (
    "contraction_bound",
    "pauli_l1_bound",
    "anticommuting_bound",
    "sdp_bound",
    "exact",
)
GROUP_NORM_METHODS: tuple[GroupNormMethod, ...] = (
    "coefficient_l1",
    "frobenius",
    "lp",
    "exact",
)
GROUPING_METHODS: tuple[GroupingMethod, ...] = ("greedy", "chemistry")


def run_molecular_qdrift_experiment(
    preset: MolecularHamiltonianPreset,
    *,
    time: float = 1.0,
    epsilon: float = 0.01,
    variance_methods: Sequence[VarianceMethod] = ("sdp_bound",),
    group_norm_method: GroupNormMethod = "lp",
    grouping_method: GroupingMethod = "chemistry",
    max_group_size: int = 12,
    include_identity: bool = False,
) -> dict[str, QDriftComparison]:
    """Compare Pauli and grouped qDRIFT for one molecular preset."""
    hamiltonian = build_lch_from_lcp_unit_cost(
        preset.to_lcp(include_identity=include_identity)
    )
    comparisons: dict[str, QDriftComparison] = {}
    for method in variance_methods:
        comparisons[method] = compare_pauli_and_grouped_qdrift(
            hamiltonian,
            time=time,
            epsilon=epsilon,
            variance_method=method,
            group_norm_method=group_norm_method,
            grouping_method=grouping_method,
            max_group_size=max_group_size,
        )
    return comparisons


def _maximum_probability(probabilities: np.ndarray) -> float:
    return float(np.max(probabilities)) if probabilities.size else 0.0


def _ratio(grouped: float, pauli: float) -> float:
    if pauli == 0.0:
        return 0.0 if grouped == 0.0 else float("inf")
    return grouped / pauli


def _print_decomposition_summary(comparison: QDriftComparison) -> None:
    print("\nDecompositions")
    print(
        f"{'type':<10} {'samples':>8} {'lambda':>16} "
        f"{'max probability':>16} {'depth / step':>16}"
    )
    for name, cost in (("pauli", comparison.pauli), ("grouped", comparison.grouped)):
        print(
            f"{name:<10} "
            f"{cost.num_sampling_terms:>8d} "
            f"{cost.lambda_sum:>16.8e} "
            f"{_maximum_probability(cost.sampling_probabilities):>16.8e} "
            f"{cost.expected_depth_per_step:>16.8e}"
        )


def _print_variance_result(comparison: QDriftComparison) -> None:
    variance_ratio = _ratio(
        comparison.grouped.variance_constant,
        comparison.pauli.variance_constant,
    )
    second_order_ratio = _ratio(
        comparison.grouped.one_step_second_order_bound,
        comparison.pauli.one_step_second_order_bound,
    )
    finite_time_ratio = _ratio(
        comparison.grouped.one_step_certified_upper_bound,
        comparison.pauli.one_step_certified_upper_bound,
    )
    print(f"\nVariance method: {comparison.variance_method}")
    print(
        f"{'type':<10} {'normalized B':>16} {'lambda^2 B':>16} "
        f"{'steps':>12} {'total depth':>16}"
    )
    for name, cost in (("pauli", comparison.pauli), ("grouped", comparison.grouped)):
        print(
            f"{name:<10} "
            f"{cost.normalized_variance_bound:>16.8e} "
            f"{cost.variance_constant:>16.8e} "
            f"{cost.steps:>12d} "
            f"{cost.expected_total_depth:>16.8e}"
        )

    print("  one-step time-t channel error (normalized diamond distance)")
    print(
        f"  {'type':<8} {'second order':>16} {'Taylor remainder':>18} "
        f"{'finite-time bound':>18}"
    )
    for name, cost in (("pauli", comparison.pauli), ("grouped", comparison.grouped)):
        print(
            f"  {name:<8} "
            f"{cost.one_step_second_order_bound:>16.8e} "
            f"{cost.one_step_taylor_remainder_bound:>18.8e} "
            f"{cost.one_step_certified_upper_bound:>18.8e}"
        )
    print(
        "  grouped / pauli variance-constant ratio = "
        f"{variance_ratio:.8e}"
    )
    print(
        "  grouped / pauli second-order ratio      = "
        f"{second_order_ratio:.8e}"
    )
    print(
        "  grouped / pauli finite-time-bound ratio = "
        f"{finite_time_ratio:.8e}"
    )
    print(
        "  grouped / pauli total-depth ratio       = "
        f"{comparison.grouped_to_pauli_depth_ratio:.8e}"
    )


def _print_summary(comparisons: dict[str, QDriftComparison]) -> None:
    first_comparison = next(iter(comparisons.values()))
    _print_decomposition_summary(first_comparison)
    print("\nVariance estimates and qDRIFT costs")
    print("  normalized B = ||sum_j p_j (H / lambda - H_j)^2||_op")
    print("  lambda^2 B   = variance constant used for the step count")
    print("  second order = time^2 * lambda^2 B")
    print("  channel error uses 0.5 * ||E_t - U_t||_diamond in [0, 1]")
    print("  finite-time bound adds the Taylor remainder and is clipped at 1")
    for comparison in comparisons.values():
        _print_variance_result(comparison)


def main() -> None:
    parser = ArgumentParser(
        description="Compare Pauli and grouped qDRIFT for molecular Hamiltonians"
    )
    parser.add_argument(
        "--hamiltonian",
        choices=sorted(MOLECULAR_HAMILTONIANS),
        default=H2_STO3G_JW.name,
    )
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--variance-method",
        choices=("all", *VARIANCE_METHODS),
        default="all",
    )
    parser.add_argument(
        "--group-norm-method",
        choices=GROUP_NORM_METHODS,
        default="lp",
    )
    parser.add_argument(
        "--grouping-method",
        choices=GROUPING_METHODS,
        default="chemistry",
    )
    parser.add_argument("--max-group-size", type=int, default=12)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--print-groups", action="store_true")
    parser.add_argument("--print-probabilities", action="store_true")
    args = parser.parse_args()

    preset = MOLECULAR_HAMILTONIANS[args.hamiltonian]
    print(f"Hamiltonian: {preset.name}")
    print(preset.description)
    print("Generating the Hamiltonian with PySCF and OpenFermion...", flush=True)
    generated = preset.generate()
    variance_methods = (
        VARIANCE_METHODS
        if args.variance_method == "all"
        else (args.variance_method,)
    )
    comparisons = run_molecular_qdrift_experiment(
        preset,
        time=args.time,
        epsilon=args.epsilon,
        variance_methods=variance_methods,
        group_norm_method=args.group_norm_method,
        grouping_method=args.grouping_method,
        max_group_size=args.max_group_size,
        include_identity=args.include_identity,
    )

    print(
        f"generated_qubits={generated.num_qubits}, "
        f"generated_pauli_terms={len(generated.terms)}, "
        f"hartree_fock_energy={generated.hartree_fock_energy:.10f}"
    )
    print(f"include_identity={args.include_identity}")
    if not args.include_identity and generated.identity_coefficient != 0.0:
        print(
            "removed identity coefficient="
            f"{generated.identity_coefficient:.8f} (global phase)"
        )
    print(f"time={args.time}, epsilon={args.epsilon}")
    print(f"grouping_method={args.grouping_method}")
    print(f"group_norm_method={args.group_norm_method}")
    _print_summary(comparisons)

    selected = comparisons["sdp_bound"] if "sdp_bound" in comparisons else next(
        iter(comparisons.values())
    )

    if args.print_groups:
        print("\ncommuting groups")
        for index, group in enumerate(selected.groups):
            print(
                f"group[{index}]: weight={group.normalization_weight:.10f}, "
                f"depth={group.rotation_depth}, "
                f"paulis=[{', '.join(group.pauli_strings)}]"
            )
    if args.print_probabilities:
        print("\npauli probabilities=", selected.pauli.sampling_probabilities)
        print("grouped probabilities=", selected.grouped.sampling_probabilities)


if __name__ == "__main__":
    main()
