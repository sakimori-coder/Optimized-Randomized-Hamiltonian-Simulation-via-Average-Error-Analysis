r"""Compare Pauli-term and commuting-group qDRIFT costs."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from grouping import (
    GroupNormMethod,
    GroupingMethod,
    build_grouped_lch,
    minimum_pauli_rotation_depth,
)
from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from hamiltonians.random import build_random_lcp
from operators import LCH, LCP
from qdrift_cost import QDriftCost, qdrift_cost
from diamond_distance.variance import (
    VarianceEstimator,
)


ComparisonVarianceMethodName = Literal[
    "exact",
    "anticommuting_bound",
    "contraction_bound",
    "pauli_l1_bound",
    "sdp_bound",
]
ComparisonVarianceMethod = ComparisonVarianceMethodName | VarianceEstimator

_COMPARISON_VARIANCE_METHODS = {
    "exact",
    "anticommuting_bound",
    "contraction_bound",
    "pauli_l1_bound",
    "sdp_bound",
}
_GROUP_NORM_METHODS = {"coefficient_l1", "frobenius", "lp", "exact"}
_GROUPING_METHODS = {"greedy", "chemistry"}


@dataclass(frozen=True)
class QDriftImplementationCost:
    """qDRIFT error estimate and expected Pauli-rotation depth."""

    variance_method: str
    num_sampling_terms: int
    lambda_sum: float
    normalized_variance_bound: float
    variance_constant: float
    steps: int
    sampling_probabilities: NDArray[np.float64]
    rotation_depths: NDArray[np.float64]
    expected_depth_per_step: float
    expected_total_depth: float
    one_step_second_order_bound: float
    one_step_taylor_remainder_bound: float
    one_step_certified_upper_bound: float


@dataclass(frozen=True)
class CommutingGroup:
    """One normalized sample in commuting-group qDRIFT."""

    normalization_weight: float
    rotation_depth: int
    pauli_strings: tuple[str, ...]


@dataclass(frozen=True)
class QDriftComparison:
    """Pauli-term versus commuting-group qDRIFT cost."""

    variance_method: str
    group_norm_method: str
    grouping_method: str
    pauli: QDriftImplementationCost
    grouped: QDriftImplementationCost
    pauli_strings: tuple[str, ...]
    groups: tuple[CommutingGroup, ...]

    @property
    def grouped_to_pauli_depth_ratio(self) -> float:
        if self.pauli.expected_total_depth == 0.0:
            if self.grouped.expected_total_depth == 0.0:
                return 0.0
            return float("inf")
        return self.grouped.expected_total_depth / self.pauli.expected_total_depth


def compare_pauli_and_grouped_qdrift(
    hamiltonian: LCH,
    *,
    time: float,
    epsilon: float,
    variance_method: ComparisonVarianceMethod = "contraction_bound",
    group_norm_method: GroupNormMethod = "lp",
    grouping_method: GroupingMethod = "greedy",
    max_group_size: int = 12,
) -> QDriftComparison:
    r"""Compare both qDRIFT decompositions under one variance method.

    A commuting group ``G_g`` is written as ``h_g A_g``, where ``h_g`` is
    estimated with ``group_norm_method`` and ``A_g=G_g/h_g``.  Certified
    upper-bound methods make every ``A_g`` a Hermitian contraction.

    Variance method ``exact`` materializes ``2**n`` matrices.  In contrast,
    ``anticommuting_bound``, ``contraction_bound``, ``pauli_l1_bound``, and
    ``sdp_bound`` remain matrix-free when combined with ``coefficient_l1`` or
    ``lp`` group normalization.  Groups are built by a deterministic greedy
    partition.  ``max_group_size`` caps every generic-greedy group and every
    non-diagonal chemistry fragment; the chemistry method deliberately keeps
    all Z-only terms in one group even when that exceeds the cap.  Group-norm
    methods ``coefficient_l1`` and ``lp`` provide certified upper bounds.  The
    group-norm method ``exact`` is a sparse numerical calculation, so it is
    not a rigorous certificate.  ``grouping_method="greedy"`` uses the
    generic lexicographic partition, while ``"chemistry"`` first groups a
    Jordan--Wigner molecular Hamiltonian by non-diagonal orbital support and
    then merges fully commuting fragments by sorted insertion.

    The standard decomposition is rebuilt from the flattened Hamiltonian,
    assigning cost zero to identity and unit cost to every non-identity
    Pauli.  Grouped depths are computed by
    :func:`minimum_pauli_rotation_depth`.  Consequently, both sides depend
    only on the flattened Hamiltonian and not on the input LCH decomposition.
    """
    if not isinstance(hamiltonian, LCH):
        raise TypeError("hamiltonian must be an LCH")
    if (
        isinstance(variance_method, str)
        and variance_method not in _COMPARISON_VARIANCE_METHODS
    ):
        raise ValueError(
            "comparison variance_method must be 'exact', "
            "'anticommuting_bound', 'contraction_bound', 'pauli_l1_bound', "
            "or 'sdp_bound'"
        )
    if group_norm_method not in _GROUP_NORM_METHODS:
        raise ValueError(
            "group_norm_method must be 'coefficient_l1', 'frobenius', "
            "'lp', or 'exact'"
        )
    if not isinstance(max_group_size, int) or isinstance(max_group_size, bool):
        raise TypeError("max_group_size must be an integer")
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")

    flattened_hamiltonian = hamiltonian.lcp
    pauli_hamiltonian = build_lch_from_lcp_unit_cost(flattened_hamiltonian)

    active_terms: list[tuple[float, LCP, float]] = []
    for coefficient, operator, cost in pauli_hamiltonian.lcp_terms:
        if coefficient == 0:
            continue
        if not np.isfinite(coefficient):
            raise ValueError("Hamiltonian coefficients must be finite")
        active_terms.append((float(coefficient), operator, float(cost)))

    pauli_result = qdrift_cost(
        pauli_hamiltonian,
        time=time,
        epsilon=epsilon,
        variance_method=variance_method,
    )
    pauli_depths = np.asarray([cost for _, _, cost in active_terms], dtype=float)
    pauli_cost = _implementation_cost_from_lch_result(
        pauli_result,
        rotation_depths=pauli_depths,
    )

    active_lcp = LCP(
        {
            pauli: coefficient
            for pauli, coefficient in flattened_hamiltonian.terms.items()
            if coefficient != 0.0
        },
        num_qubits=hamiltonian.num_qubits,
    )
    grouped_hamiltonian = build_grouped_lch(
        active_lcp,
        grouping_method=grouping_method,
        group_norm_method=group_norm_method,
        max_group_size=max_group_size,
    )

    group_records: list[CommutingGroup] = []
    grouped_lch_terms: list[tuple[float, LCP, float]] = []
    group_depths: list[float] = []
    for weight, group in grouped_hamiltonian.terms:
        depth = minimum_pauli_rotation_depth(group)
        grouped_lch_terms.append(
            (weight, group, float(depth))
        )
        group_depths.append(float(depth))
        group_records.append(
            CommutingGroup(
                normalization_weight=weight,
                rotation_depth=depth,
                pauli_strings=tuple(group.terms),
            )
        )

    grouped_result = qdrift_cost(
        LCH(grouped_lch_terms, num_qubits=hamiltonian.num_qubits),
        time=time,
        epsilon=epsilon,
        variance_method=variance_method,
    )
    depth_array = np.asarray(group_depths, dtype=float)
    method_name = _method_name(variance_method)
    grouped_cost = _implementation_cost_from_lch_result(
        grouped_result,
        rotation_depths=depth_array,
    )

    return QDriftComparison(
        variance_method=method_name,
        group_norm_method=group_norm_method,
        grouping_method=grouping_method,
        pauli=pauli_cost,
        grouped=grouped_cost,
        pauli_strings=tuple(active_lcp.terms),
        groups=tuple(group_records),
    )


def _implementation_cost_from_lch_result(
    result: QDriftCost,
    *,
    rotation_depths: NDArray[np.float64],
) -> QDriftImplementationCost:
    return QDriftImplementationCost(
        variance_method=result.variance_method,
        num_sampling_terms=len(rotation_depths),
        lambda_sum=result.lambda_sum,
        normalized_variance_bound=result.normalized_variance_bound,
        variance_constant=result.variance_constant,
        steps=result.steps,
        sampling_probabilities=result.sampling_probabilities,
        rotation_depths=rotation_depths,
        expected_depth_per_step=result.per_step_cost,
        expected_total_depth=result.total_cost,
        one_step_second_order_bound=result.one_step_second_order_bound,
        one_step_taylor_remainder_bound=(
            result.one_step_taylor_remainder_bound
        ),
        one_step_certified_upper_bound=(
            result.one_step_certified_upper_bound
        ),
    )


def _method_name(method: ComparisonVarianceMethod) -> str:
    if isinstance(method, str):
        return method
    return getattr(method, "__name__", "custom")


def _print_cost(name: str, cost: QDriftImplementationCost) -> None:
    maximum_probability = (
        float(np.max(cost.sampling_probabilities))
        if cost.sampling_probabilities.size
        else 0.0
    )
    print(
        f"{name}: variance={cost.variance_method}, "
        f"lambda={cost.lambda_sum:.10f}, "
        f"normalized_variance_bound={cost.normalized_variance_bound:.10f}, "
        f"variance_constant={cost.variance_constant:.10f}, "
        "one_step_second_order_bound="
        f"{cost.one_step_second_order_bound:.10f}, "
        "one_step_certified_upper_bound="
        f"{cost.one_step_certified_upper_bound:.10f}, "
        f"steps={cost.steps}, "
        f"max_probability={maximum_probability:.10f}, "
        f"expected_depth_per_step={cost.expected_depth_per_step:.10f}, "
        f"expected_total_depth={cost.expected_total_depth:.10f}"
    )


def main() -> None:
    parser = ArgumentParser(description="Compare Pauli and grouped qDRIFT costs")
    parser.add_argument("--num-terms", type=int, default=12)
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variance-method",
        choices=sorted(_COMPARISON_VARIANCE_METHODS),
        default="contraction_bound",
    )
    parser.add_argument(
        "--group-norm-method",
        choices=sorted(_GROUP_NORM_METHODS),
        default="lp",
    )
    parser.add_argument(
        "--grouping-method",
        choices=sorted(_GROUPING_METHODS),
        default="greedy",
    )
    parser.add_argument("--max-group-size", type=int, default=12)
    parser.add_argument("--print-probabilities", action="store_true")
    parser.add_argument("--print-groups", action="store_true")
    args = parser.parse_args()

    if args.variance_method == "exact" and args.num_qubits > 12:
        print("Warning: exact variance constructs 2^n matrices.")
    if args.group_norm_method == "exact":
        warning = "Warning: exact group norm is not a certified upper bound."
        if args.num_qubits > 12:
            warning += " It also constructs a 2^n sparse matrix."
        print(warning)

    hamiltonian = build_lch_from_lcp_unit_cost(
        build_random_lcp(
            num_terms=args.num_terms,
            num_qubits=args.num_qubits,
            seed=args.seed,
        )
    )
    comparison = compare_pauli_and_grouped_qdrift(
        hamiltonian,
        time=args.time,
        epsilon=args.epsilon,
        variance_method=args.variance_method,
        group_norm_method=args.group_norm_method,
        grouping_method=args.grouping_method,
        max_group_size=args.max_group_size,
    )

    print(f"num_terms={args.num_terms}, num_qubits={args.num_qubits}")
    print(f"num_commuting_groups={len(comparison.groups)}")
    print(f"grouping_method={comparison.grouping_method}")
    print(f"group_norm_method={comparison.group_norm_method}")
    _print_cost("pauli_qdrift", comparison.pauli)
    _print_cost("grouped_qdrift", comparison.grouped)
    print(
        "grouped/pauli total-depth ratio="
        f"{comparison.grouped_to_pauli_depth_ratio:.10f}"
    )

    if args.print_probabilities:
        print(f"pauli probabilities={comparison.pauli.sampling_probabilities}")
        print(f"grouped probabilities={comparison.grouped.sampling_probabilities}")
    if args.print_groups:
        for index, group in enumerate(comparison.groups):
            print(
                f"group[{index}]: weight={group.normalization_weight:.10f}, "
                f"depth={group.rotation_depth}, "
                f"paulis=[{', '.join(group.pauli_strings)}]"
            )


if __name__ == "__main__":
    main()
