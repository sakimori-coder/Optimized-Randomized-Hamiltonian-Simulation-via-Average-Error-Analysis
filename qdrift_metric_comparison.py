r"""Single-purpose Pauli-versus-grouped qDRIFT metric experiment."""

from __future__ import annotations

import json
import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grouping import build_chemistry_depth1_frobenius_lch
from hamiltonians.chemistry import (
    H2_STO3G_JW,
    MOLECULAR_HAMILTONIANS,
    GeneratedMolecularHamiltonian,
    MolecularHamiltonianPreset,
)
from hamiltonians.pauli import build_lch_from_lcp_unit_cost
from hamiltonians.syk import GeneratedSykHamiltonian, SykHamiltonianPreset
from operators import LCH, LCP
from qdrift_trajectory_metrics import (
    QDriftTrajectoryMetricEstimate,
    ScalarSampleStatistics,
    TauMEstimate,
    calculate_tau_m,
    estimate_qdrift_haar_trajectory_metrics,
)


DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 1)
PARTIAL_FORMAT_VERSION = 1
HamiltonianPreset = MolecularHamiltonianPreset | SykHamiltonianPreset
GeneratedHamiltonian = GeneratedMolecularHamiltonian | GeneratedSykHamiltonian


@dataclass(frozen=True)
class QDriftDecompositions:
    preset: HamiltonianPreset
    generated: GeneratedHamiltonian
    target: LCP
    pauli: LCH
    grouped: LCH
    max_group_size: int


@dataclass(frozen=True)
class QDriftMetricExperiment:
    decompositions: QDriftDecompositions
    total_time: float
    number_of_steps: int
    num_initial_states: int
    state_indices: tuple[int, ...]
    num_trajectories: int
    num_workers: int
    trajectory_chunks_per_state: int
    tau_m: tuple[TauMEstimate, TauMEstimate]
    trajectory_metrics: tuple[
        QDriftTrajectoryMetricEstimate,
        QDriftTrajectoryMetricEstimate,
    ]

    @property
    def tau_m_ratio(self) -> float:
        pauli, grouped = self.tau_m
        return _ratio(pauli.tau_m, grouped.tau_m)

    @property
    def infidelity_ratio(self) -> float:
        pauli, grouped = self.trajectory_metrics
        return _ratio(pauli.mean_infidelity, grouped.mean_infidelity)

    @property
    def qpe_signal_error_ratio(self) -> float:
        pauli, grouped = self.trajectory_metrics
        return _ratio(
            pauli.mean_absolute_qpe_signal_error,
            grouped.mean_absolute_qpe_signal_error,
        )


@dataclass(frozen=True)
class QDriftMetricPartial:
    """One PBS array shard, partitioned by complete Haar input states."""

    metadata: dict[str, object]
    shard_index: int
    state_indices: np.ndarray
    tau_m: np.ndarray
    infidelities: np.ndarray
    qpe_signal_errors: np.ndarray


@dataclass(frozen=True)
class MergedQDriftMetricResult:
    """Validated aggregate of all PBS array shards."""

    metadata: dict[str, object]
    tau_m: tuple[float, float]
    infidelity: tuple[ScalarSampleStatistics, ScalarSampleStatistics]
    qpe_signal_error: tuple[
        ScalarSampleStatistics,
        ScalarSampleStatistics,
    ]


def build_decompositions(
    preset: HamiltonianPreset,
    *,
    max_group_size: int,
) -> QDriftDecompositions:
    """Generate the target and the two fixed qDRIFT decompositions."""
    generated = preset.generate()
    target = generated.to_lcp(include_identity=False)
    return QDriftDecompositions(
        preset=preset,
        generated=generated,
        target=target,
        pauli=build_lch_from_lcp_unit_cost(target),
        grouped=build_chemistry_depth1_frobenius_lch(
            target,
            max_group_size=max_group_size,
        ),
        max_group_size=max_group_size,
    )


def run_experiment(
    preset: HamiltonianPreset,
    *,
    total_time: float,
    number_of_steps: int,
    num_initial_states: int = 20,
    num_trajectories: int = 200,
    max_group_size: int = 12,
    seed: int | None = 42,
    num_workers: int = DEFAULT_NUM_WORKERS,
    trajectory_chunks_per_state: int | None = None,
    initial_state_indices: tuple[int, ...] | None = None,
) -> QDriftMetricExperiment:
    """Compute the three requested Pauli/grouped ratios."""
    decompositions = build_decompositions(
        preset,
        max_group_size=max_group_size,
    )
    return run_experiment_from_decompositions(
        decompositions,
        total_time=total_time,
        number_of_steps=number_of_steps,
        num_initial_states=num_initial_states,
        num_trajectories=num_trajectories,
        seed=seed,
        num_workers=num_workers,
        trajectory_chunks_per_state=trajectory_chunks_per_state,
        initial_state_indices=initial_state_indices,
    )


def run_experiment_from_decompositions(
    decompositions: QDriftDecompositions,
    *,
    total_time: float,
    number_of_steps: int,
    num_initial_states: int = 20,
    num_trajectories: int = 200,
    seed: int | None = 42,
    num_workers: int = DEFAULT_NUM_WORKERS,
    trajectory_chunks_per_state: int | None = None,
    initial_state_indices: tuple[int, ...] | None = None,
) -> QDriftMetricExperiment:
    """Run metrics from already generated/grouped decompositions."""
    pair = (decompositions.pauli, decompositions.grouped)
    trajectory_metrics = tuple(
        estimate_qdrift_haar_trajectory_metrics(
            decompositions.target,
            total_time,
            number_of_steps,
            pair,
            num_initial_states=num_initial_states,
            num_trajectories=num_trajectories,
            seed=seed,
            num_workers=num_workers,
            trajectory_chunks_per_state=trajectory_chunks_per_state,
            initial_state_indices=initial_state_indices,
        )
    )
    effective_chunks = (
        1
        if num_workers == 1
        else min(
            num_trajectories,
            num_workers
            if trajectory_chunks_per_state is None
            else trajectory_chunks_per_state,
        )
    )
    return QDriftMetricExperiment(
        decompositions=decompositions,
        total_time=float(total_time),
        number_of_steps=number_of_steps,
        num_initial_states=num_initial_states,
        state_indices=(
            tuple(range(num_initial_states))
            if initial_state_indices is None
            else tuple(initial_state_indices)
        ),
        num_trajectories=num_trajectories,
        num_workers=num_workers,
        trajectory_chunks_per_state=effective_chunks,
        tau_m=(calculate_tau_m(pair[0]), calculate_tau_m(pair[1])),
        trajectory_metrics=trajectory_metrics,
    )


def _ratio(pauli: float, grouped: float) -> float:
    if grouped == 0.0:
        return 0.0 if pauli == 0.0 else float("inf")
    return pauli / grouped


def print_experiment(experiment: QDriftMetricExperiment) -> None:
    decompositions = experiment.decompositions
    generated = decompositions.generated
    pauli_tau, grouped_tau = experiment.tau_m
    pauli_metrics, grouped_metrics = experiment.trajectory_metrics
    group_sizes = tuple(
        len(operator.terms)
        for _, operator in decompositions.grouped.terms
    )

    print("\nHamiltonian")
    print(f"  name                 = {decompositions.preset.name}")
    print(f"  description          = {decompositions.preset.description}")
    print(f"  qubits               = {generated.num_qubits}")
    print(f"  Pauli terms          = {len(decompositions.target)}")
    print(f"  depth-1 groups       = {len(group_sizes)}")
    if group_sizes:
        print(
            "  group size min/mean/max = "
            f"{min(group_sizes)}/{np.mean(group_sizes):.3f}/{max(group_sizes)}"
        )
    if isinstance(generated, GeneratedMolecularHamiltonian):
        print(f"  Hartree-Fock energy  = {generated.hartree_fock_energy:.10f}")
        print(f"  removed identity     = {generated.identity_coefficient:.10f}")
    else:
        print(f"  Majorana modes       = {generated.num_majoranas}")
        print(f"  disorder seed        = {generated.seed}")

    print("\nFixed method")
    print("  grouping             = chemistry_depth1")
    print("  group sampling weight = sqrt(sum_P a_P^2) [Frobenius]")
    print(f"  max group size       = {decompositions.max_group_size}")
    print(f"  total time           = {experiment.total_time}")
    print(f"  qDRIFT steps R       = {experiment.number_of_steps}")
    print(f"  shared Haar states K = {experiment.num_initial_states}")
    if len(experiment.state_indices) != experiment.num_initial_states:
        print(
            "  states in this shard = "
            f"{len(experiment.state_indices)} "
            f"(global indices {experiment.state_indices[0]}.."
            f"{experiment.state_indices[-1]})"
        )
    print(f"  trajectories/state S = {experiment.num_trajectories}")
    print(f"  process workers      = {experiment.num_workers}")
    print(f"  trajectory chunks    = {experiment.trajectory_chunks_per_state}")
    print("  Qulacs threads/worker = 1")

    print("\nThree requested metrics")
    print(
        f"  {'metric':<30} {'pauli':>16} {'grouped':>16} "
        f"{'ratio (pauli/grouped)':>24}"
    )
    rows = (
        ("tau(M_p)", pauli_tau.tau_m, grouped_tau.tau_m),
        (
            "average infidelity",
            pauli_metrics.mean_infidelity,
            grouped_metrics.mean_infidelity,
        ),
        (
            "average QPE signal error",
            pauli_metrics.mean_absolute_qpe_signal_error,
            grouped_metrics.mean_absolute_qpe_signal_error,
        ),
    )
    for name, pauli_value, grouped_value in rows:
        print(
            f"  {name:<30} {pauli_value:>16.8e} "
            f"{grouped_value:>16.8e} "
            f"{_ratio(pauli_value, grouped_value):>24.8e}"
        )

    print("\nMonte Carlo standard errors")
    print(f"  {'metric':<30} {'pauli':>16} {'grouped':>16}")
    print(
        f"  {'average infidelity':<30} "
        f"{pauli_metrics.infidelity.standard_error:>16.8e} "
        f"{grouped_metrics.infidelity.standard_error:>16.8e}"
    )
    print(
        f"  {'average QPE signal error':<30} "
        f"{pauli_metrics.absolute_qpe_signal_error.standard_error:>16.8e} "
        f"{grouped_metrics.absolute_qpe_signal_error.standard_error:>16.8e}"
    )


def save_partial_experiment(
    experiment: QDriftMetricExperiment,
    output_directory: str | Path,
    *,
    shard_index: int,
    num_shards: int,
    seed: int,
) -> Path:
    """Atomically save the per-input observations from one array subjob."""
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    expected_indices = tuple(
        range(shard_index, experiment.num_initial_states, num_shards)
    )
    if experiment.state_indices != expected_indices:
        raise ValueError("experiment state indices do not match this shard")

    metadata = _partial_metadata(experiment, num_shards=num_shards, seed=seed)
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"partial_{shard_index:05d}.npz"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp.npz",
        dir=directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    pauli_metrics, grouped_metrics = experiment.trajectory_metrics
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            shard_index=np.asarray(shard_index, dtype=np.int64),
            state_indices=np.asarray(
                experiment.state_indices,
                dtype=np.int64,
            ),
            tau_m=np.asarray(
                [experiment.tau_m[0].tau_m, experiment.tau_m[1].tau_m],
                dtype=np.float64,
            ),
            infidelities=np.vstack(
                [
                    pauli_metrics.infidelity.values,
                    grouped_metrics.infidelity.values,
                ]
            ),
            qpe_signal_errors=np.vstack(
                [
                    pauli_metrics.absolute_qpe_signal_error.values,
                    grouped_metrics.absolute_qpe_signal_error.values,
                ]
            ),
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_partial_experiment(path: str | Path) -> QDriftMetricPartial:
    """Load one shard without permitting pickled Python objects."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "shard_index",
                "state_indices",
                "tau_m",
                "infidelities",
                "qpe_signal_errors",
            }
            if set(archive.files) != required:
                raise ValueError("partial file has unexpected fields")
            metadata = json.loads(str(archive["metadata_json"].item()))
            partial = QDriftMetricPartial(
                metadata=metadata,
                shard_index=int(archive["shard_index"].item()),
                state_indices=np.array(archive["state_indices"], copy=True),
                tau_m=np.array(archive["tau_m"], copy=True),
                infidelities=np.array(archive["infidelities"], copy=True),
                qpe_signal_errors=np.array(
                    archive["qpe_signal_errors"],
                    copy=True,
                ),
            )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read partial result {source}: {error}") from error
    _validate_partial(partial, source)
    return partial


def merge_partial_experiments(
    input_directory: str | Path,
) -> MergedQDriftMetricResult:
    """Validate complete shard coverage and combine per-input observations."""
    directory = Path(input_directory).expanduser().resolve()
    files = sorted(directory.glob("partial_*.npz"))
    if not files:
        raise ValueError(f"no partial_*.npz files found in {directory}")
    partials = [load_partial_experiment(path) for path in files]
    metadata = partials[0].metadata
    if any(partial.metadata != metadata for partial in partials[1:]):
        raise ValueError("partial-result metadata do not match")

    num_shards = int(metadata["num_shards"])
    shard_indices = [partial.shard_index for partial in partials]
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError("duplicate shard indices were found")
    expected_shards = set(range(num_shards))
    actual_shards = set(shard_indices)
    if actual_shards != expected_shards:
        missing = sorted(expected_shards - actual_shards)
        unexpected = sorted(actual_shards - expected_shards)
        raise ValueError(
            f"incomplete shard set: missing={missing}, unexpected={unexpected}"
        )

    reference_tau = partials[0].tau_m
    if any(
        not np.allclose(
            partial.tau_m,
            reference_tau,
            rtol=1e-12,
            atol=1e-14,
        )
        for partial in partials[1:]
    ):
        raise ValueError("tau(M_p) values do not match across shards")

    state_indices = np.concatenate(
        [partial.state_indices for partial in partials]
    )
    infidelities = np.concatenate(
        [partial.infidelities for partial in partials],
        axis=1,
    )
    qpe_errors = np.concatenate(
        [partial.qpe_signal_errors for partial in partials],
        axis=1,
    )
    order = np.argsort(state_indices)
    sorted_indices = state_indices[order]
    expected_indices = np.arange(
        int(metadata["num_initial_states"]),
        dtype=np.int64,
    )
    if not np.array_equal(sorted_indices, expected_indices):
        raise ValueError(
            "partial files do not contain each global input-state index once"
        )
    infidelities = infidelities[:, order]
    qpe_errors = qpe_errors[:, order]
    return MergedQDriftMetricResult(
        metadata=metadata,
        tau_m=(float(reference_tau[0]), float(reference_tau[1])),
        infidelity=(
            ScalarSampleStatistics.from_values(infidelities[0]),
            ScalarSampleStatistics.from_values(infidelities[1]),
        ),
        qpe_signal_error=(
            ScalarSampleStatistics.from_values(qpe_errors[0]),
            ScalarSampleStatistics.from_values(qpe_errors[1]),
        ),
    )


def print_merged_result(result: MergedQDriftMetricResult) -> None:
    """Print the same three ratios after combining all array subjobs."""
    metadata = result.metadata
    print("\nMerged PBS result")
    print(f"  Hamiltonian          = {metadata['hamiltonian_name']}")
    print(f"  qubits               = {metadata['num_qubits']}")
    print(f"  Pauli terms          = {metadata['num_pauli_terms']}")
    print(f"  depth-1 groups       = {metadata['num_groups']}")
    print(f"  completed shards     = {metadata['num_shards']}")
    print(f"  shared Haar states K = {metadata['num_initial_states']}")
    print(f"  trajectories/state S = {metadata['num_trajectories']}")
    print(f"  qDRIFT steps R       = {metadata['number_of_steps']}")

    print("\nThree requested metrics")
    print(
        f"  {'metric':<30} {'pauli':>16} {'grouped':>16} "
        f"{'ratio (pauli/grouped)':>24}"
    )
    rows = (
        ("tau(M_p)", *result.tau_m),
        (
            "average infidelity",
            result.infidelity[0].mean,
            result.infidelity[1].mean,
        ),
        (
            "average QPE signal error",
            result.qpe_signal_error[0].mean,
            result.qpe_signal_error[1].mean,
        ),
    )
    for name, pauli_value, grouped_value in rows:
        print(
            f"  {name:<30} {pauli_value:>16.8e} "
            f"{grouped_value:>16.8e} "
            f"{_ratio(pauli_value, grouped_value):>24.8e}"
        )

    print("\nMonte Carlo standard errors (all input states)")
    print(f"  {'metric':<30} {'pauli':>16} {'grouped':>16}")
    print(
        f"  {'average infidelity':<30} "
        f"{result.infidelity[0].standard_error:>16.8e} "
        f"{result.infidelity[1].standard_error:>16.8e}"
    )
    print(
        f"  {'average QPE signal error':<30} "
        f"{result.qpe_signal_error[0].standard_error:>16.8e} "
        f"{result.qpe_signal_error[1].standard_error:>16.8e}"
    )


def _partial_metadata(
    experiment: QDriftMetricExperiment,
    *,
    num_shards: int,
    seed: int,
) -> dict[str, object]:
    decompositions = experiment.decompositions
    group_sizes = [
        len(operator.terms)
        for _, operator in decompositions.grouped.terms
    ]
    return {
        "format_version": PARTIAL_FORMAT_VERSION,
        "hamiltonian_name": decompositions.preset.name,
        "hamiltonian_description": decompositions.preset.description,
        "num_qubits": decompositions.generated.num_qubits,
        "num_pauli_terms": len(decompositions.target),
        "num_groups": len(group_sizes),
        "group_size_min": min(group_sizes) if group_sizes else None,
        "group_size_max": max(group_sizes) if group_sizes else None,
        "group_size_mean": float(np.mean(group_sizes)) if group_sizes else None,
        "total_time": experiment.total_time,
        "number_of_steps": experiment.number_of_steps,
        "num_initial_states": experiment.num_initial_states,
        "num_trajectories": experiment.num_trajectories,
        "max_group_size": decompositions.max_group_size,
        "trajectory_chunks_per_state": experiment.trajectory_chunks_per_state,
        "seed": seed,
        "num_shards": num_shards,
        "grouping": "chemistry_depth1",
        "group_weight": "frobenius",
    }


def _validate_partial(partial: QDriftMetricPartial, path: Path) -> None:
    metadata = partial.metadata
    required_metadata = {
        "format_version",
        "hamiltonian_name",
        "num_qubits",
        "num_pauli_terms",
        "num_groups",
        "total_time",
        "number_of_steps",
        "num_initial_states",
        "num_trajectories",
        "max_group_size",
        "trajectory_chunks_per_state",
        "seed",
        "num_shards",
    }
    if not isinstance(metadata, dict) or not required_metadata <= metadata.keys():
        raise ValueError(f"partial file has incomplete metadata: {path}")
    if metadata["format_version"] != PARTIAL_FORMAT_VERSION:
        raise ValueError(f"unsupported partial format in {path}")
    if partial.state_indices.ndim != 1 or partial.state_indices.size == 0:
        raise ValueError(f"invalid state_indices in {path}")
    local_count = partial.state_indices.size
    if partial.tau_m.shape != (2,):
        raise ValueError(f"invalid tau_m shape in {path}")
    if partial.infidelities.shape != (2, local_count):
        raise ValueError(f"invalid infidelities shape in {path}")
    if partial.qpe_signal_errors.shape != (2, local_count):
        raise ValueError(f"invalid qpe_signal_errors shape in {path}")
    if not np.issubdtype(partial.state_indices.dtype, np.integer):
        raise ValueError(f"state_indices must be integers in {path}")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            partial.tau_m,
            partial.infidelities,
            partial.qpe_signal_errors,
        )
    ):
        raise ValueError(f"partial file contains non-finite values: {path}")
    num_shards = int(metadata["num_shards"])
    if not 0 <= partial.shard_index < num_shards:
        raise ValueError(f"invalid shard index in {path}")
    expected = np.arange(
        partial.shard_index,
        int(metadata["num_initial_states"]),
        num_shards,
        dtype=np.int64,
    )
    if not np.array_equal(partial.state_indices, expected):
        raise ValueError(f"state indices do not match shard index in {path}")


def _preset_from_arguments(args) -> HamiltonianPreset:
    if args.syk_qubits is not None:
        return SykHamiltonianPreset(
            num_qubits=args.syk_qubits,
            coupling_scale=args.syk_coupling_scale,
            seed=args.seed if args.syk_seed is None else args.syk_seed,
        )
    return MOLECULAR_HAMILTONIANS[args.hamiltonian]


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Compare tau(M_p), average infidelity, and average QPE signal "
            "error for Pauli and chemistry_depth1/Frobenius qDRIFT"
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--hamiltonian",
        choices=sorted(MOLECULAR_HAMILTONIANS),
        default=H2_STO3G_JW.name,
    )
    source.add_argument("--syk-qubits", type=int, metavar="NUM_QUBITS")
    parser.add_argument("--syk-coupling-scale", type=float, default=1.0)
    parser.add_argument("--syk-seed", type=int)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--number-of-steps", type=int, default=100)
    parser.add_argument("--num-initial-states", type=int, default=20)
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--max-group-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--trajectory-chunks-per-state",
        type=int,
        help="default: num-workers",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="zero-based PBS shard index",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        help="total number of PBS array shards",
    )
    parser.add_argument(
        "--partial-output-dir",
        type=Path,
        help="directory for partial_XXXXX.npz shard files",
    )
    parser.add_argument(
        "--merge-partials",
        type=Path,
        metavar="DIRECTORY",
        help="validate and merge a completed PBS result directory",
    )
    args = parser.parse_args()

    if args.merge_partials is not None:
        print_merged_result(merge_partial_experiments(args.merge_partials))
        return

    shard_arguments = (
        args.shard_index,
        args.num_shards,
        args.partial_output_dir,
    )
    if any(value is not None for value in shard_arguments) and any(
        value is None for value in shard_arguments
    ):
        parser.error(
            "--shard-index, --num-shards, and --partial-output-dir "
            "must be specified together"
        )
    if args.num_shards is not None:
        if args.num_shards <= 0:
            parser.error("--num-shards must be positive")
        if not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must satisfy 0 <= index < num-shards")
        if args.num_shards > args.num_initial_states:
            parser.error("--num-shards cannot exceed --num-initial-states")
        state_indices = tuple(
            range(
                args.shard_index,
                args.num_initial_states,
                args.num_shards,
            )
        )
    else:
        state_indices = None

    print("Generating Hamiltonian and fixed qDRIFT decompositions...", flush=True)
    experiment = run_experiment(
        _preset_from_arguments(args),
        total_time=args.time,
        number_of_steps=args.number_of_steps,
        num_initial_states=args.num_initial_states,
        num_trajectories=args.num_trajectories,
        max_group_size=args.max_group_size,
        seed=args.seed,
        num_workers=args.num_workers,
        trajectory_chunks_per_state=args.trajectory_chunks_per_state,
        initial_state_indices=state_indices,
    )
    print_experiment(experiment)
    if args.partial_output_dir is not None:
        destination = save_partial_experiment(
            experiment,
            args.partial_output_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            seed=args.seed,
        )
        print(f"\nPartial result saved to {destination}")


if __name__ == "__main__":
    main()
