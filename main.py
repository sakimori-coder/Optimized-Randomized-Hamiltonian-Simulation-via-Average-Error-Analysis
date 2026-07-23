r"""Compare Pauli-term and commuting-group qDRIFT costs."""

from __future__ import annotations

import math
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from commuting_operator_norm_estimator import (
    BuiltInMethod as GroupNormMethod,
    estimate_commuting_operator_norm_upper_bound,
)
from lch import LCH
from lcp import LCP
from qdrift_cost import QDriftCost, qdrift_cost
from qdrift_variance_estimator import (
    VarianceEstimator,
    anticommuting_bound_from_samples,
    estimate_centered_second_moment_norm,
    pauli_l1_bound_from_samples,
    pauli_moment_sdp_bound_from_samples,
    qdrift_decomposition_from_samples,
)
from rotation_depth import minimum_pauli_rotation_depth


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
_GROUP_NORM_METHODS = {"coefficient_l1", "lp", "exact"}


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


def _random_pauli_string(num_qubits: int, rng: np.random.Generator) -> str:
    paulis = np.array(["I", "X", "Y", "Z"])
    return "".join(rng.choice(paulis, size=num_qubits).tolist())


def build_random_lcp(
    num_terms: int,
    num_qubits: int,
    *,
    seed: int | None = None,
) -> LCP:
    """Build a random LCP with unique Pauli strings and real coefficients."""
    if not isinstance(num_terms, int) or num_terms < 0:
        raise ValueError("num_terms must be a non-negative integer")
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("num_qubits must be a non-negative integer")
    if num_terms > 4**num_qubits:
        raise ValueError("num_terms cannot exceed 4^num_qubits")

    rng = np.random.default_rng(seed)
    terms: dict[str, complex] = {}
    while len(terms) < num_terms:
        pauli = _random_pauli_string(num_qubits, rng)
        if pauli not in terms:
            terms[pauli] = complex(float(rng.normal()))
    return LCP(terms, num_qubits=num_qubits)


def build_lch_from_lcp_unit_cost(hamiltonian: LCP) -> LCH:
    """Convert an LCP to an LCH with unit Pauli-rotation depth."""
    return LCH(
        [
            (
                coefficient,
                pauli,
                0.0 if all(symbol == "I" for symbol in pauli) else 1.0,
            )
            for pauli, coefficient in hamiltonian.terms.items()
        ],
        num_qubits=hamiltonian.num_qubits,
    )


def compare_pauli_and_grouped_qdrift(
    hamiltonian: LCH,
    *,
    time: float,
    epsilon: float,
    variance_method: ComparisonVarianceMethod = "contraction_bound",
    group_norm_method: GroupNormMethod = "lp",
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
    partition and contain at most ``max_group_size`` Pauli terms.  Group-norm
    methods ``coefficient_l1`` and ``lp`` provide certified upper bounds.  The
    group-norm method ``exact`` is a sparse numerical calculation, so it is
    not a rigorous certificate.

    Pauli-term rotation depths are read from the costs stored in
    ``hamiltonian``; grouped depths are computed by
    :func:`minimum_pauli_rotation_depth`.
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
            "group_norm_method must be 'coefficient_l1', 'lp', or 'exact'"
        )
    if not isinstance(max_group_size, int) or isinstance(max_group_size, bool):
        raise TypeError("max_group_size must be an integer")
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")

    active_terms: list[tuple[float, str, float]] = []
    for coefficient, pauli, cost in hamiltonian.lcp_terms:
        if coefficient == 0:
            continue
        if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
            raise ValueError("Hamiltonian coefficients must be finite")
        if coefficient.imag != 0.0:
            raise ValueError("Hamiltonian coefficients must be real")
        active_terms.append((float(coefficient.real), pauli, float(cost)))

    pauli_result = qdrift_cost(
        hamiltonian,
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
        {pauli: coefficient for coefficient, pauli, _ in active_terms},
        num_qubits=hamiltonian.num_qubits,
    )
    grouped_hamiltonians = _greedy_commuting_groups(
        active_lcp,
        max_group_size=max_group_size,
    )

    group_records: list[CommutingGroup] = []
    normalized_groups: list[tuple[LCP, float]] = []
    group_depths: list[float] = []
    for group in grouped_hamiltonians:
        weight = estimate_commuting_operator_norm_upper_bound(
            group,
            method=group_norm_method,
        )
        if weight == 0.0:
            continue
        depth = minimum_pauli_rotation_depth(group)
        normalized_groups.append((group, weight))
        group_depths.append(float(depth))
        group_records.append(
            CommutingGroup(
                normalization_weight=weight,
                rotation_depth=depth,
                pauli_strings=tuple(group.terms),
            )
        )

    group_weights = [weight for _, weight in normalized_groups]
    group_probabilities, grouped_lambda = _probabilities_from_weights(group_weights)
    grouped_variance = _grouped_variance_bound(
        normalized_groups,
        method=variance_method,
    )
    grouped_variance_constant = (
        0.0
        if grouped_variance == 0.0
        else float(grouped_lambda * grouped_lambda * grouped_variance)
    )
    grouped_steps = _required_steps(
        lambda_sum=grouped_lambda,
        normalized_variance_bound=grouped_variance,
        time=pauli_result.time,
        epsilon=pauli_result.epsilon,
    )
    depth_array = np.asarray(group_depths, dtype=float)
    expected_depth = (
        float(np.dot(group_probabilities, depth_array))
        if group_probabilities.size
        else 0.0
    )
    method_name = _method_name(variance_method)
    grouped_cost = QDriftImplementationCost(
        variance_method=method_name,
        num_sampling_terms=len(group_records),
        lambda_sum=grouped_lambda,
        normalized_variance_bound=grouped_variance,
        variance_constant=grouped_variance_constant,
        steps=grouped_steps,
        sampling_probabilities=group_probabilities,
        rotation_depths=depth_array,
        expected_depth_per_step=expected_depth,
        expected_total_depth=float(grouped_steps * expected_depth),
    )

    return QDriftComparison(
        variance_method=method_name,
        group_norm_method=group_norm_method,
        pauli=pauli_cost,
        grouped=grouped_cost,
        pauli_strings=tuple(pauli for _, pauli, _ in active_terms),
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
    )


def _grouped_variance_bound(
    normalized_groups: list[tuple[LCP, float]],
    *,
    method: ComparisonVarianceMethod,
) -> float:
    if not normalized_groups:
        return 0.0

    weights = [weight for _, weight in normalized_groups]
    if method == "contraction_bound":
        return 1.0
    if method == "pauli_l1_bound":
        return pauli_l1_bound_from_samples(
            weights,
            [group * (1.0 / weight) for group, weight in normalized_groups],
        )
    if method == "anticommuting_bound":
        return anticommuting_bound_from_samples(
            weights,
            [group * (1.0 / weight) for group, weight in normalized_groups],
        )
    if method == "sdp_bound":
        return pauli_moment_sdp_bound_from_samples(
            weights,
            [group * (1.0 / weight) for group, weight in normalized_groups],
        )

    samples = [group.to_csr() / weight for group, weight in normalized_groups]
    decomposition = qdrift_decomposition_from_samples(weights, samples)
    return estimate_centered_second_moment_norm(
        decomposition,
        method=method,
    )


def _greedy_commuting_groups(
    hamiltonian: LCP,
    *,
    max_group_size: int,
) -> list[LCP]:
    grouped_terms: list[dict[str, complex]] = []
    terms = hamiltonian.terms
    for pauli in sorted(terms):
        coefficient = terms[pauli]
        compatible = [
            index
            for index, group in enumerate(grouped_terms)
            if len(group) < max_group_size
            and all(LCP._pauli_strings_commute(pauli, other) for other in group)
        ]
        if compatible:
            selected = max(
                compatible,
                key=lambda index: (len(grouped_terms[index]), -index),
            )
            grouped_terms[selected][pauli] = coefficient
        else:
            grouped_terms.append({pauli: coefficient})

    return [
        LCP(group, num_qubits=hamiltonian.num_qubits)
        for group in grouped_terms
    ]


def _probabilities_from_weights(
    weights: list[float],
) -> tuple[NDArray[np.float64], float]:
    if not weights:
        return np.array([], dtype=float), 0.0
    maximum = max(weights)
    scaled = np.asarray([weight / maximum for weight in weights], dtype=float)
    scaled_sum = math.fsum(float(weight) for weight in scaled)
    return scaled / scaled_sum, float(maximum * scaled_sum)


def _required_steps(
    *,
    lambda_sum: float,
    normalized_variance_bound: float,
    time: float,
    epsilon: float,
) -> int:
    if lambda_sum == 0.0 or time == 0.0:
        return 0
    if normalized_variance_bound == 0.0:
        return 1
    lambda_time = lambda_sum * abs(time)
    return max(
        1,
        math.ceil(
            2.0
            * normalized_variance_bound
            * lambda_time
            * lambda_time
            / epsilon
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
        max_group_size=args.max_group_size,
    )

    print(f"num_terms={args.num_terms}, num_qubits={args.num_qubits}")
    print(f"num_commuting_groups={len(comparison.groups)}")
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
