"""Compare Pauli and commuting-group qDRIFT using exact matrix squares."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np

from brute_force_qdrift_error import (
    BruteForceQDriftError,
    brute_force_qdrift_error,
    brute_force_qdrift_error_from_decomposition,
)
from commuting_groups import decompose_into_commuting_groups
from lch import LCH
from lcp import LCP
from rotation_depth import minimum_pauli_rotation_depth


@dataclass(frozen=True)
class BruteForceImplementationCost:
    """Direct error estimate and expected Pauli-rotation depth."""

    error: BruteForceQDriftError
    expected_depth_per_step: float
    expected_total_depth: float


@dataclass(frozen=True)
class BruteForceGroup:
    """One commuting group used as a normalized qDRIFT sample."""

    h_g: float
    rotation_depth: int
    pauli_strings: tuple[str, ...]


@dataclass(frozen=True)
class BruteForceQDriftComparison:
    """Direct-error cost comparison for two qDRIFT decompositions."""

    pauli: BruteForceImplementationCost
    grouped: BruteForceImplementationCost
    pauli_strings: tuple[str, ...]
    groups: tuple[BruteForceGroup, ...]

    @property
    def grouped_to_pauli_depth_ratio(self) -> float:
        if self.pauli.expected_total_depth == 0.0:
            return 0.0 if self.grouped.expected_total_depth == 0.0 else float("inf")
        return self.grouped.expected_total_depth / self.pauli.expected_total_depth


def compare_brute_force_qdrift_costs(
    hamiltonian: LCH,
    *,
    time: float,
    epsilon: float,
) -> BruteForceQDriftComparison:
    r"""Compare costs using ``||sum p_j(H/Lambda-H_j)^2||_op``.

    Standard qDRIFT samples signed Pauli strings.  Grouped qDRIFT samples
    ``H_g=G_g/||G_g||_op`` after a commuting partition.  The required number
    of steps for both decompositions is obtained from the directly
    constructed centered second-moment matrix.
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")
    lcp_hamiltonian = hamiltonian.lcp
    if any(coefficient.imag != 0 for coefficient in lcp_hamiltonian.terms.values()):
        raise ValueError("Hamiltonian coefficients must be real")

    active_terms = {
        pauli: coefficient
        for pauli, coefficient in lcp_hamiltonian.terms.items()
        if coefficient != 0
    }
    active_hamiltonian = LCP(
        active_terms,
        num_qubits=lcp_hamiltonian.num_qubits,
    )

    pauli_error = brute_force_qdrift_error(
        hamiltonian,
        time=time,
        epsilon=epsilon,
    )
    pauli_depths = np.asarray(
        [
            0.0 if all(symbol == "I" for symbol in pauli) else 1.0
            for pauli in active_terms
        ],
        dtype=float,
    )
    pauli_per_step_depth = (
        float(np.dot(pauli_error.sampling_probabilities, pauli_depths))
        if pauli_depths.size
        else 0.0
    )
    pauli_cost = BruteForceImplementationCost(
        error=pauli_error,
        expected_depth_per_step=pauli_per_step_depth,
        expected_total_depth=float(pauli_error.required_steps * pauli_per_step_depth),
    )

    group_weights: list[float] = []
    normalized_groups = []
    group_depths: list[int] = []
    group_records: list[BruteForceGroup] = []
    for group in decompose_into_commuting_groups(active_hamiltonian):
        h_g = group.operator_norm()
        if h_g == 0.0:
            continue
        depth = minimum_pauli_rotation_depth(group)
        group_weights.append(h_g)
        normalized_groups.append(group.to_csr() / h_g)
        group_depths.append(depth)
        group_records.append(
            BruteForceGroup(
                h_g=h_g,
                rotation_depth=depth,
                pauli_strings=tuple(group.terms),
            )
        )

    grouped_error = brute_force_qdrift_error_from_decomposition(
        group_weights,
        normalized_groups,
        time=time,
        epsilon=epsilon,
    )
    group_depth_array = np.asarray(group_depths, dtype=float)
    grouped_per_step_depth = (
        float(np.dot(grouped_error.sampling_probabilities, group_depth_array))
        if group_depth_array.size
        else 0.0
    )
    grouped_cost = BruteForceImplementationCost(
        error=grouped_error,
        expected_depth_per_step=grouped_per_step_depth,
        expected_total_depth=float(
            grouped_error.required_steps * grouped_per_step_depth
        ),
    )

    return BruteForceQDriftComparison(
        pauli=pauli_cost,
        grouped=grouped_cost,
        pauli_strings=tuple(active_terms),
        groups=tuple(group_records),
    )


def _print_implementation(name: str, cost: BruteForceImplementationCost) -> None:
    error = cost.error
    print(
        f"{name}: lambda={error.lambda_sum:.10f}, "
        f"normalized_variance={error.normalized_variance_constant:.10e}, "
        f"variance={error.variance_constant:.10e}, "
        f"required_steps={error.required_steps}, "
        f"theory_steps={error.theory_steps}, "
        f"expected_depth_per_step={cost.expected_depth_per_step:.10f}, "
        f"expected_total_depth={cost.expected_total_depth:.10f}"
    )


def main() -> None:
    parser = ArgumentParser(
        description="Compare qDRIFT costs using exact centered matrix squares"
    )
    parser.add_argument("--num-terms", type=int, default=6)
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-probabilities", action="store_true")
    args = parser.parse_args()

    from main import build_lch_from_lcp_unit_cost, build_random_lcp

    hamiltonian = build_lch_from_lcp_unit_cost(
        build_random_lcp(
            num_terms=args.num_terms,
            num_qubits=args.num_qubits,
            seed=args.seed,
        )
    )
    comparison = compare_brute_force_qdrift_costs(
        hamiltonian,
        time=args.time,
        epsilon=args.epsilon,
    )

    print(f"num_terms={args.num_terms}, num_qubits={args.num_qubits}")
    print(f"num_commuting_groups={len(comparison.groups)}")
    _print_implementation("pauli_qdrift", comparison.pauli)
    _print_implementation("grouped_qdrift", comparison.grouped)
    print(
        "grouped/pauli total-depth ratio="
        f"{comparison.grouped_to_pauli_depth_ratio:.10f}"
    )

    if args.print_probabilities:
        print("pauli sampling distribution:")
        for pauli, probability in zip(
            comparison.pauli_strings,
            comparison.pauli.error.sampling_probabilities,
        ):
            print(f"  {pauli}: probability={probability:.10f}")
        print("commuting-group sampling distribution:")
        for index, (group, probability) in enumerate(
            zip(comparison.groups, comparison.grouped.error.sampling_probabilities)
        ):
            print(
                f"  group[{index}]: probability={probability:.10f}, "
                f"h_g={group.h_g:.10f}, depth={group.rotation_depth}, "
                f"paulis=[{', '.join(group.pauli_strings)}]"
            )


if __name__ == "__main__":
    main()
