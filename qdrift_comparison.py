"""Compare Pauli-term and commuting-group qDRIFT implementation costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from commuting_groups import decompose_into_commuting_groups
from lch import LCH
from lcp import LCP
from qdrift_cost import qdrift_cost
from rotation_depth import minimum_pauli_rotation_depth


@dataclass(frozen=True)
class QDriftImplementationCost:
    """qDRIFT steps and expected Pauli-rotation depth."""

    num_sampling_terms: int
    lambda_sum: float
    steps: int
    sampling_probabilities: NDArray[np.float64]
    rotation_depths: NDArray[np.float64]
    expected_depth_per_step: float
    expected_total_depth: float


@dataclass(frozen=True)
class CommutingGroup:
    """One normalized term ``G_g = h_g H_g`` in grouped qDRIFT."""

    h_g: float
    rotation_depth: int
    pauli_strings: tuple[str, ...]


@dataclass(frozen=True)
class QDriftComparison:
    """Cost comparison for two decompositions of the same Hamiltonian."""

    pauli: QDriftImplementationCost
    grouped: QDriftImplementationCost
    pauli_strings: tuple[str, ...]
    groups: tuple[CommutingGroup, ...]

    @property
    def grouped_to_pauli_depth_ratio(self) -> float:
        if self.pauli.expected_total_depth == 0.0:
            return 0.0 if self.grouped.expected_total_depth == 0.0 else float("inf")
        return self.grouped.expected_total_depth / self.pauli.expected_total_depth


def compare_qdrift_implementations(
    hamiltonian: LCP,
    *,
    time: float,
    epsilon: float,
) -> QDriftComparison:
    """Compare standard and commuting-group qDRIFT under a depth cost model.

    Standard qDRIFT samples individual Pauli strings.  Grouped qDRIFT first
    writes ``H = sum_g G_g = sum_g h_g H_g``, where each ``G_g`` is a
    pairwise-commuting group, ``h_g = ||G_g||_op``, and ``||H_g||_op = 1``.

    Clifford basis changes are free.  The cost of evolving one commuting
    group is its exact minimum parallel Pauli-rotation depth.
    """
    if not isinstance(hamiltonian, LCP):
        raise TypeError("hamiltonian must be an LCP")
    if any(coefficient.imag != 0 for coefficient in hamiltonian.terms.values()):
        raise ValueError("Hamiltonian coefficients must be real")

    active_terms = {
        pauli: coefficient
        for pauli, coefficient in hamiltonian.terms.items()
        if coefficient != 0
    }
    active_hamiltonian = LCP(active_terms, num_qubits=hamiltonian.num_qubits)

    pauli_lch = LCH(
        [
            (
                coefficient,
                pauli,
                0.0 if all(symbol == "I" for symbol in pauli) else 1.0,
            )
            for pauli, coefficient in active_terms.items()
        ],
        num_qubits=hamiltonian.num_qubits,
    )
    pauli_result = qdrift_cost(
        pauli_lch,
        time=time,
        epsilon=epsilon,
        variance_method="contraction_bound",
    )
    pauli_depths = np.asarray(pauli_lch.evolution_costs, dtype=float)
    pauli_cost = QDriftImplementationCost(
        num_sampling_terms=len(active_terms),
        lambda_sum=pauli_result.lambda_sum,
        steps=pauli_result.steps,
        sampling_probabilities=pauli_result.sampling_probabilities,
        rotation_depths=pauli_depths,
        expected_depth_per_step=pauli_result.per_step_cost,
        expected_total_depth=pauli_result.total_cost,
    )

    group_records: list[CommutingGroup] = []
    for group in decompose_into_commuting_groups(active_hamiltonian):
        h_g = group.operator_norm()
        if h_g == 0.0:
            continue
        group_records.append(
            CommutingGroup(
                h_g=h_g,
                rotation_depth=minimum_pauli_rotation_depth(group),
                pauli_strings=tuple(group.terms),
            )
        )

    group_weights = np.asarray([group.h_g for group in group_records], dtype=float)
    group_depths = np.asarray(
        [group.rotation_depth for group in group_records],
        dtype=float,
    )
    grouped_lambda = float(group_weights.sum())
    if grouped_lambda == 0.0:
        group_probabilities = np.array([], dtype=float)
        grouped_steps = 0
        grouped_per_step_depth = 0.0
    else:
        group_probabilities = group_weights / grouped_lambda
        grouped_steps = int(
            np.ceil(2.0 * (grouped_lambda * abs(float(time))) ** 2 / float(epsilon))
        )
        grouped_per_step_depth = float(np.dot(group_probabilities, group_depths))

    grouped_cost = QDriftImplementationCost(
        num_sampling_terms=len(group_records),
        lambda_sum=grouped_lambda,
        steps=grouped_steps,
        sampling_probabilities=group_probabilities,
        rotation_depths=group_depths,
        expected_depth_per_step=grouped_per_step_depth,
        expected_total_depth=float(grouped_steps * grouped_per_step_depth),
    )

    return QDriftComparison(
        pauli=pauli_cost,
        grouped=grouped_cost,
        pauli_strings=tuple(active_terms),
        groups=tuple(group_records),
    )
