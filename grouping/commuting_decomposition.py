r"""Construct a normalized commuting-group decomposition from an LCP."""

from __future__ import annotations

from .commuting import GroupingMethod, group_commuting_terms
from .operator_norm import (
    GroupNormMethod,
    estimate_commuting_operator_norm_upper_bound,
)
from operators import LCH, LCP


def build_grouped_lch(
    hamiltonian: LCP,
    *,
    grouping_method: GroupingMethod = "greedy",
    group_norm_method: GroupNormMethod = "lp",
    max_group_size: int | None = None,
) -> LCH:
    r"""Return ``H = sum_g h_g H_g`` as a grouped LCH.

    First partition the input as ``H = sum_g G_g``, where every ``G_g`` is
    a commuting Pauli sum.  Then compute the selected norm ``h_g`` and
    normalize the sampled Hamiltonian as ``H_g = G_g / h_g``.  In
    particular, ``group_norm_method="frobenius"`` gives
    ``p_g proportional to ||G_g||_F``.

    Thus each returned LCH term is ``(h_g, H_g)`` and the flattened LCH is
    exactly the input Hamiltonian.
    """
    groups = group_commuting_terms(
        hamiltonian,
        method=grouping_method,
        max_group_size=max_group_size,
    )

    terms = []
    for group in groups:
        weight = estimate_commuting_operator_norm_upper_bound(
            group,
            method=group_norm_method,
        )
        terms.append((weight, group * (1.0 / weight)))

    return LCH(terms, num_qubits=hamiltonian.num_qubits)
