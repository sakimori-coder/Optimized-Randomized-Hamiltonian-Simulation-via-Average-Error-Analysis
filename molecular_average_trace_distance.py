r"""Compare Pauli and grouped qDRIFT for molecular Hamiltonians."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Literal

from average_trace_distance.comparison import (
    estimate_qdrift_haar_average_trace_distances,
)
from average_trace_distance.density_operator import TraceDistanceMethod
from average_trace_distance.monte_carlo import (
    QDriftHaarAverageTraceDistanceEstimate,
)
from average_trace_distance.statistics import TraceDistanceStatistics
from average_trace_distance.variance_bound import (
    QDriftHaarAverageTraceDistanceBound,
    qdrift_haar_average_trace_distance_bounds,
)
from grouping import GroupingMethod, GroupNormMethod, build_grouped_lch
from hamiltonians.chemistry import (
    H2_STO3G_JW,
    MOLECULAR_HAMILTONIANS,
    GeneratedMolecularHamiltonian,
    MolecularHamiltonianPreset,
    hydrogen_chain_preset,
)
from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from operators import LCH, LCP


CalculationMode = Literal["both", "monte_carlo", "bound"]
CALCULATION_MODES: tuple[CalculationMode, ...] = (
    "both",
    "monte_carlo",
    "bound",
)
GROUP_NORM_METHODS: tuple[GroupNormMethod, ...] = (
    "coefficient_l1",
    "frobenius",
    "lp",
    "exact",
)
GROUPING_METHODS: tuple[GroupingMethod, ...] = ("greedy", "chemistry")
TRACE_DISTANCE_METHODS: tuple[TraceDistanceMethod, ...] = (
    "low_rank",
    "dense",
)


@dataclass(frozen=True)
class MolecularQDriftDecompositions:
    """Target Hamiltonian and the two qDRIFT decompositions being compared."""

    preset: MolecularHamiltonianPreset
    generated_hamiltonian: GeneratedMolecularHamiltonian
    target_hamiltonian: LCP
    pauli_qdrift: LCH
    grouped_qdrift: LCH
    grouping_method: GroupingMethod
    group_norm_method: GroupNormMethod
    max_group_size: int

    @property
    def groups(self) -> tuple[tuple[str, ...], ...]:
        """Return Pauli strings in every commuting group."""
        return tuple(
            tuple(group.terms)
            for _, group in self.grouped_qdrift.terms
        )


@dataclass(frozen=True)
class MolecularAverageTraceDistanceExperiment:
    """All outputs requested for one molecular experiment."""

    decompositions: MolecularQDriftDecompositions
    total_time: float
    number_of_steps: int
    step_time: float
    mode: CalculationMode
    estimates: tuple[
        QDriftHaarAverageTraceDistanceEstimate,
        QDriftHaarAverageTraceDistanceEstimate,
    ] | None
    bounds: tuple[
        QDriftHaarAverageTraceDistanceBound,
        QDriftHaarAverageTraceDistanceBound,
    ] | None


def build_molecular_qdrift_decompositions(
    preset: MolecularHamiltonianPreset,
    *,
    group_norm_method: GroupNormMethod = "lp",
    grouping_method: GroupingMethod = "chemistry",
    max_group_size: int = 12,
) -> MolecularQDriftDecompositions:
    """Build Pauli and commuting-group qDRIFT LCHs from one preset."""
    generated = preset.generate()
    target_hamiltonian = generated.to_lcp(include_identity=False)
    pauli_qdrift = build_lch_from_lcp_unit_cost(target_hamiltonian)
    grouped_qdrift = build_grouped_lch(
        target_hamiltonian,
        group_norm_method=group_norm_method,
        grouping_method=grouping_method,
        max_group_size=max_group_size,
    )
    return MolecularQDriftDecompositions(
        preset=preset,
        generated_hamiltonian=generated,
        target_hamiltonian=target_hamiltonian,
        pauli_qdrift=pauli_qdrift,
        grouped_qdrift=grouped_qdrift,
        grouping_method=grouping_method,
        group_norm_method=group_norm_method,
        max_group_size=max_group_size,
    )


def run_molecular_average_trace_distance_experiment(
    preset: MolecularHamiltonianPreset,
    *,
    total_time: float,
    number_of_steps: int,
    num_initial_states: int = 100,
    group_norm_method: GroupNormMethod = "lp",
    grouping_method: GroupingMethod = "chemistry",
    max_group_size: int = 12,
    trace_distance_method: TraceDistanceMethod = "low_rank",
    seed: int | None = 42,
    mode: CalculationMode = "both",
) -> MolecularAverageTraceDistanceExperiment:
    """Run the requested Monte Carlo and analytic-bound calculations."""
    decompositions = build_molecular_qdrift_decompositions(
        preset,
        group_norm_method=group_norm_method,
        grouping_method=grouping_method,
        max_group_size=max_group_size,
    )
    qdrift_decompositions = [
        decompositions.pauli_qdrift,
        decompositions.grouped_qdrift,
    ]
    estimates = (
        tuple(
            estimate_qdrift_haar_average_trace_distances(
                decompositions.target_hamiltonian,
                total_time,
                number_of_steps,
                qdrift_decompositions,
                num_initial_states=num_initial_states,
                trace_distance_method=trace_distance_method,
                seed=seed,
            )
        )
        if mode in ("both", "monte_carlo")
        else None
    )
    bounds = (
        tuple(
            qdrift_haar_average_trace_distance_bounds(
                decompositions.target_hamiltonian,
                total_time,
                number_of_steps,
                qdrift_decompositions,
            )
        )
        if mode in ("both", "bound")
        else None
    )
    return MolecularAverageTraceDistanceExperiment(
        decompositions=decompositions,
        total_time=float(total_time),
        number_of_steps=number_of_steps,
        step_time=float(total_time) / number_of_steps,
        mode=mode,
        estimates=estimates,
        bounds=bounds,
    )


def _ratio(grouped: float, pauli: float) -> float:
    if pauli == 0.0:
        return 0.0 if grouped == 0.0 else float("inf")
    return grouped / pauli


def _print_hamiltonian_and_settings(
    decompositions: MolecularQDriftDecompositions,
    *,
    total_time: float,
    number_of_steps: int,
) -> None:
    generated = decompositions.generated_hamiltonian
    print("\nHamiltonian")
    print(f"  name                 = {decompositions.preset.name}")
    print(f"  description          = {decompositions.preset.description}")
    print(f"  qubits               = {generated.num_qubits}")
    print(f"  Pauli terms          = {len(decompositions.target_hamiltonian.terms)}")
    print(f"  commuting groups     = {len(decompositions.groups)}")
    print(f"  group sizes          = {[len(group) for group in decompositions.groups]}")
    print(f"  Hartree-Fock energy  = {generated.hartree_fock_energy:.10f}")
    print(f"  removed identity     = {generated.identity_coefficient:.10f}")

    print("\nSettings")
    print(f"  total time           = {total_time}")
    print(f"  number of steps      = {number_of_steps}")
    print(f"  step time            = {float(total_time) / number_of_steps}")
    print(f"  grouping method      = {decompositions.grouping_method}")
    print(f"  group norm method    = {decompositions.group_norm_method}")
    print(f"  max group size       = {decompositions.max_group_size}")


def _print_monte_carlo(
    pauli: QDriftHaarAverageTraceDistanceEstimate,
    grouped: QDriftHaarAverageTraceDistanceEstimate,
) -> None:
    paired_difference = TraceDistanceStatistics.from_values(
        grouped.values - pauli.values
    )
    print("\nMonte Carlo Haar-average trace distance")
    print(f"  Haar states          = {pauli.num_initial_states}")
    print(f"  trace-distance method = {pauli.trace_distance_method}")
    print(
        f"{'type':<10} {'mean':>16} {'sample std':>16} "
        f"{'standard error':>16}"
    )
    for name, estimate in (("pauli", pauli), ("grouped", grouped)):
        print(
            f"{name:<10} "
            f"{estimate.mean:>16.8e} "
            f"{estimate.sample_standard_deviation:>16.8e} "
            f"{estimate.standard_error:>16.8e}"
        )
    print(
        "  grouped / pauli mean ratio = "
        f"{_ratio(grouped.mean, pauli.mean):.8e}"
    )
    print(
        "  grouped - pauli paired mean = "
        f"{paired_difference.mean:.8e} "
        f"(standard error {paired_difference.standard_error:.8e})"
    )


def _bound_metrics(
    bound: QDriftHaarAverageTraceDistanceBound,
) -> tuple[float, ...]:
    return (
        bound.lambda_sum,
        bound.tau_v,
        bound.tau_v_squared,
        bound.normalized_average_variance_bound,
        bound.step_second_order_bound,
        bound.step_taylor_remainder_bound,
        bound.step_average_trace_distance_upper_bound,
    )


def _print_bounds(
    pauli: QDriftHaarAverageTraceDistanceBound,
    grouped: QDriftHaarAverageTraceDistanceBound,
) -> None:
    names = (
        "lambda",
        "tau(V)",
        "tau(V^2)",
        "C",
        "second-order term",
        "Taylor remainder",
        "one-step upper bound",
    )
    print("\nHaar-average trace-distance bounds")
    print(
        f"{'metric':<22} {'pauli':>16} {'grouped':>16} "
        f"{'grouped / pauli':>18}"
    )
    for name, pauli_value, grouped_value in zip(
        names,
        _bound_metrics(pauli),
        _bound_metrics(grouped),
    ):
        print(
            f"{name:<22} "
            f"{pauli_value:>16.8e} "
            f"{grouped_value:>16.8e} "
            f"{_ratio(grouped_value, pauli_value):>18.8e}"
        )


def _print_result(
    experiment: MolecularAverageTraceDistanceExperiment,
) -> None:
    """Print every result calculated by one experiment."""
    _print_hamiltonian_and_settings(
        experiment.decompositions,
        total_time=experiment.total_time,
        number_of_steps=experiment.number_of_steps,
    )
    if experiment.estimates is not None:
        _print_monte_carlo(*experiment.estimates)
    if experiment.bounds is not None:
        _print_bounds(*experiment.bounds)


def _preset_from_arguments(args) -> MolecularHamiltonianPreset:
    if args.hydrogen_chain is None:
        return MOLECULAR_HAMILTONIANS[args.hamiltonian]
    return hydrogen_chain_preset(
        args.hydrogen_chain,
        spacing=args.spacing,
    )


def main() -> None:
    """Run one molecular average trace-distance experiment."""
    parser = ArgumentParser(
        description=(
            "Compare Pauli and grouped qDRIFT Haar-average trace distances "
            "for one molecular Hamiltonian"
        )
    )
    hamiltonian_source = parser.add_mutually_exclusive_group()
    hamiltonian_source.add_argument(
        "--hamiltonian",
        choices=sorted(MOLECULAR_HAMILTONIANS),
        default=H2_STO3G_JW.name,
    )
    hamiltonian_source.add_argument(
        "--hydrogen-chain",
        type=int,
        metavar="NUM_ATOMS",
        help="generate an equally spaced hydrogen chain",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="hydrogen-chain spacing in angstrom",
    )
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--number-of-steps", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=CALCULATION_MODES,
        default="both",
    )
    parser.add_argument("--num-initial-states", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--trace-distance-method",
        choices=TRACE_DISTANCE_METHODS,
        default="low_rank",
    )
    parser.add_argument("--print-groups", action="store_true")
    args = parser.parse_args()

    preset = _preset_from_arguments(args)
    print("Generating the molecular Hamiltonian...", flush=True)
    experiment = run_molecular_average_trace_distance_experiment(
        preset,
        total_time=args.time,
        number_of_steps=args.number_of_steps,
        num_initial_states=args.num_initial_states,
        group_norm_method=args.group_norm_method,
        grouping_method=args.grouping_method,
        max_group_size=args.max_group_size,
        trace_distance_method=args.trace_distance_method,
        seed=args.seed,
        mode=args.mode,
    )
    _print_result(experiment)

    if args.print_groups:
        print("\nGrouped qDRIFT decomposition")
        print(
            experiment.decompositions.grouped_qdrift.format_decomposition()
        )


if __name__ == "__main__":
    main()
