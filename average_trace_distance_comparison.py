r"""Compare Pauli and commuting-group qDRIFT trace distances."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass

from average_trace_distance.comparison import (
    estimate_qdrift_haar_average_trace_distances,
)
from average_trace_distance.density_operator import TraceDistanceMethod
from average_trace_distance.monte_carlo import (
    QDriftHaarAverageTraceDistanceEstimate,
)
from average_trace_distance.variance_bound import (
    QDriftHaarAverageTraceDistanceBound,
    qdrift_haar_average_trace_distance_bounds,
)
from diamond_distance import (
    QDriftDiamondDistanceBound,
    VarianceMethod,
    qdrift_diamond_distance_bounds,
)
from grouping import (
    GroupNormMethod,
    GroupingMethod,
    build_grouped_lch,
)
from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from operators import LCH

_GROUP_NORM_METHODS = {"coefficient_l1", "frobenius", "lp", "exact"}
_TRACE_DISTANCE_METHODS = {"low_rank", "dense"}
_GROUPING_METHODS = {"greedy", "chemistry"}
_OPERATOR_VARIANCE_METHODS = {
    "exact",
    "contraction_bound",
    "pauli_l1_bound",
    "anticommuting_bound",
    "sdp_bound",
}


@dataclass(frozen=True)
class AverageTraceDistanceComparison:
    """Pauli-term and commuting-group qDRIFT step comparison."""

    num_qubits: int
    num_initial_states: int
    total_time: float
    number_of_steps: int
    step_time: float
    group_norm_method: str
    grouping_method: str
    trace_distance_method: str
    operator_variance_method: str
    max_group_size: int
    num_pauli_terms: int
    groups: tuple[tuple[str, ...], ...]
    group_weights: tuple[float, ...]
    pauli_hamiltonian: LCH
    grouped_hamiltonian: LCH
    pauli: QDriftHaarAverageTraceDistanceEstimate
    grouped: QDriftHaarAverageTraceDistanceEstimate
    pauli_bound: QDriftHaarAverageTraceDistanceBound
    grouped_bound: QDriftHaarAverageTraceDistanceBound
    pauli_diamond_bound: QDriftDiamondDistanceBound
    grouped_diamond_bound: QDriftDiamondDistanceBound

    @property
    def grouped_to_pauli_mean_ratio(self) -> float:
        """Return the grouped mean trace distance divided by the Pauli mean."""
        if self.pauli.mean == 0.0:
            return 0.0 if self.grouped.mean == 0.0 else float("inf")
        return self.grouped.mean / self.pauli.mean


def compare_average_trace_distances(
    hamiltonian: LCH,
    *,
    total_time: float,
    number_of_steps: int,
    num_initial_states: int = 100,
    group_norm_method: GroupNormMethod = "lp",
    grouping_method: GroupingMethod = "greedy",
    max_group_size: int = 12,
    trace_distance_method: TraceDistanceMethod = "low_rank",
    operator_variance_method: VarianceMethod = "exact",
    seed: int | None = None,
) -> AverageTraceDistanceComparison:
    r"""Compare both qDRIFT decompositions on shared Haar-random states.

    For every initial state, this function evaluates the trace distance from
    the ideal output to the complete Pauli qDRIFT channel and commuting-group
    qDRIFT channel.  Both decompositions use exactly the same initial states
    and ideal evolutions, giving a paired Monte Carlo comparison.

    The returned comparison also contains the independent ``tau(V)`` and
    ``tau(V**2)`` Haar-average bounds and operator-norm-derived normalized
    diamond-distance bounds for both decompositions.  This is a single-
    channel-step comparison whose duration is
    ``total_time / number_of_steps``.
    """
    target_hamiltonian = hamiltonian.lcp
    pauli_hamiltonian = build_lch_from_lcp_unit_cost(target_hamiltonian)
    grouped_hamiltonian = build_grouped_lch(
        target_hamiltonian,
        grouping_method=grouping_method,
        group_norm_method=group_norm_method,
        max_group_size=max_group_size,
    )
    group_weights = tuple(
        weight for weight, _ in grouped_hamiltonian.terms
    )
    group_paulis = tuple(
        tuple(group.terms) for _, group in grouped_hamiltonian.terms
    )
    estimates = estimate_qdrift_haar_average_trace_distances(
        target_hamiltonian,
        total_time,
        number_of_steps,
        [pauli_hamiltonian, grouped_hamiltonian],
        num_initial_states=num_initial_states,
        trace_distance_method=trace_distance_method,
        seed=seed,
    )
    pauli_estimate, grouped_estimate = estimates
    average_bounds = qdrift_haar_average_trace_distance_bounds(
        target_hamiltonian,
        total_time,
        number_of_steps,
        [pauli_hamiltonian, grouped_hamiltonian],
    )
    pauli_bound, grouped_bound = average_bounds
    diamond_bounds = qdrift_diamond_distance_bounds(
        target_hamiltonian,
        total_time,
        number_of_steps,
        [pauli_hamiltonian, grouped_hamiltonian],
        variance_method=operator_variance_method,
    )
    pauli_diamond_bound, grouped_diamond_bound = diamond_bounds

    return AverageTraceDistanceComparison(
        num_qubits=hamiltonian.num_qubits,
        num_initial_states=num_initial_states,
        total_time=float(total_time),
        number_of_steps=number_of_steps,
        step_time=float(total_time) / number_of_steps,
        group_norm_method=group_norm_method,
        grouping_method=grouping_method,
        trace_distance_method=trace_distance_method,
        operator_variance_method=pauli_diamond_bound.variance_method,
        max_group_size=max_group_size,
        num_pauli_terms=len(target_hamiltonian.terms),
        groups=group_paulis,
        group_weights=group_weights,
        pauli_hamiltonian=pauli_hamiltonian,
        grouped_hamiltonian=grouped_hamiltonian,
        pauli=pauli_estimate,
        grouped=grouped_estimate,
        pauli_bound=pauli_bound,
        grouped_bound=grouped_bound,
        pauli_diamond_bound=pauli_diamond_bound,
        grouped_diamond_bound=grouped_diamond_bound,
    )


def main() -> None:
    """Run a random-Hamiltonian comparison from the command line."""
    parser = ArgumentParser(
        description=(
            "Compare one-step Pauli and grouped qDRIFT mean trace distances"
        )
    )
    parser.add_argument("--num-terms", type=int, default=12)
    parser.add_argument("--num-qubits", type=int, default=5)
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--number-of-steps", type=int, default=1)
    parser.add_argument("--num-initial-states", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--trace-distance-method",
        choices=sorted(_TRACE_DISTANCE_METHODS),
        default="low_rank",
    )
    parser.add_argument(
        "--operator-variance-method",
        choices=sorted(_OPERATOR_VARIANCE_METHODS),
        default="exact",
    )
    parser.add_argument("--print-groups", action="store_true")
    args = parser.parse_args()

    from hamiltonians.random import build_random_lcp

    hamiltonian = build_lch_from_lcp_unit_cost(
        build_random_lcp(
            num_terms=args.num_terms,
            num_qubits=args.num_qubits,
            seed=args.seed,
        )
    )
    comparison = compare_average_trace_distances(
        hamiltonian,
        total_time=args.time,
        number_of_steps=args.number_of_steps,
        num_initial_states=args.num_initial_states,
        group_norm_method=args.group_norm_method,
        grouping_method=args.grouping_method,
        max_group_size=args.max_group_size,
        trace_distance_method=args.trace_distance_method,
        operator_variance_method=args.operator_variance_method,
        seed=args.seed,
    )

    print(
        f"num_qubits={comparison.num_qubits}, "
        f"num_terms={comparison.num_pauli_terms}, "
        f"num_groups={len(comparison.groups)}, "
        f"num_initial_states={comparison.num_initial_states}, "
        f"total_time={comparison.total_time}, "
        f"number_of_steps={comparison.number_of_steps}, "
        f"step_time={comparison.step_time}, "
        f"grouping_method={comparison.grouping_method}"
    )
    print(
        "pauli_qdrift: "
        f"mean_trace_distance={comparison.pauli.mean:.10f}, "
        f"standard_error={comparison.pauli.standard_error:.10f}"
    )
    print(
        "grouped_qdrift: "
        f"mean_trace_distance={comparison.grouped.mean:.10f}, "
        f"standard_error={comparison.grouped.standard_error:.10f}"
    )
    print(
        "grouped/pauli mean ratio="
        f"{comparison.grouped_to_pauli_mean_ratio:.10f}"
    )
    _print_diamond_bound(
        comparison.pauli_diamond_bound,
        comparison.grouped_diamond_bound,
    )
    _print_bound("pauli_qdrift_bound", comparison.pauli_bound)
    _print_bound("grouped_qdrift_bound", comparison.grouped_bound)
    if args.print_groups:
        print("\ngrouped qDRIFT decomposition")
        print(comparison.grouped_hamiltonian.format_decomposition())


def _print_bound(
    name: str,
    bound: QDriftHaarAverageTraceDistanceBound,
) -> None:
    print(
        f"{name}: "
        f"tau_V={bound.tau_v:.10f}, "
        f"tau_V_squared={bound.tau_v_squared:.10f}, "
        f"C={bound.normalized_average_variance_bound:.10f}, "
        f"step_second_order_bound={bound.step_second_order_bound:.10f}, "
        "step_taylor_remainder_bound="
        f"{bound.step_taylor_remainder_bound:.10f}, "
        "step_average_trace_distance_upper_bound="
        f"{bound.step_average_trace_distance_upper_bound:.10f}"
    )


def _print_diamond_bound(
    pauli: QDriftDiamondDistanceBound,
    grouped: QDriftDiamondDistanceBound,
) -> None:
    print(
        "worst-case normalized-diamond-distance bound "
        "(from operator-norm variance): "
        f"variance_method={pauli.variance_method}"
    )
    print(
        "pauli_qdrift: "
        f"B={pauli.normalized_variance_bound:.10f}, "
        f"step_second_order_bound={pauli.step_second_order_bound:.10f}, "
        "step_taylor_remainder_bound="
        f"{pauli.step_taylor_remainder_bound:.10f}, "
        "step_diamond_distance_upper_bound="
        f"{pauli.step_diamond_distance_upper_bound:.10f}"
    )
    print(
        "grouped_qdrift: "
        f"B={grouped.normalized_variance_bound:.10f}, "
        "step_second_order_bound="
        f"{grouped.step_second_order_bound:.10f}, "
        "step_taylor_remainder_bound="
        f"{grouped.step_taylor_remainder_bound:.10f}, "
        "step_diamond_distance_upper_bound="
        f"{grouped.step_diamond_distance_upper_bound:.10f}"
    )
    print(
        "grouped/pauli second-order-bound ratio="
        f"{_safe_ratio(grouped.step_second_order_bound, pauli.step_second_order_bound):.10f}"
    )
    print(
        "grouped/pauli finite-time-bound ratio="
        f"{_safe_ratio(grouped.step_diamond_distance_upper_bound, pauli.step_diamond_distance_upper_bound):.10f}"
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


if __name__ == "__main__":
    main()
