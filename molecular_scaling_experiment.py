r"""Molecular qDRIFT benchmark for Pauli/grouped improvement factors.

The fixed benchmark contains full-orbital STO-3G and cc-pVDZ molecules plus
the Reiher FeMoco CAS(54e,54o) FCIDUMP Hamiltonian.  Every system gets the
exact coefficient-time ``tau(M_p)`` comparison.  Haar/state-vector estimates
of average infidelity and average QPE signal error are computed only through
the configured qubit cutoff (16 qubits by default, including NH3/STO-3G).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from grouping import chemistry_depth_one_groups
from hamiltonians.chemistry import (
    BEH2_STO3G_FULL_JW,
    CH4_CCPVDZ_FULL_JW,
    CH4_STO3G_FULL_JW,
    H2O_CCPVDZ_FULL_JW,
    H2O_STO3G_FULL_JW,
    H2_STO3G_JW,
    LIH_STO3G_FULL_JW,
    N2_STO3G_FULL_JW,
    NH3_STO3G_FULL_JW,
    MolecularHamiltonianPreset,
)
from hamiltonians.femoco import (
    FEMOCO_NAME,
    ReiherFeMocoHamiltonianPreset,
)
from qdrift_metric_comparison import (
    DEFAULT_NUM_WORKERS,
    run_experiment_from_decompositions,
)
from qdrift_trajectory_metrics import TauMEstimate


RESULT_FORMAT_VERSION = 1
METRICS = (
    "tau_m_factor",
    "infidelity_factor",
    "qpe_signal_error_factor",
)


@dataclass(frozen=True)
class MolecularBenchmarkSpec:
    system_id: str
    molecule: str
    basis: str
    expected_qubits: int
    preset: MolecularHamiltonianPreset | None


MOLECULAR_BENCHMARKS = (
    MolecularBenchmarkSpec(
        H2_STO3G_JW.name, "H2", "STO-3G", 4, H2_STO3G_JW
    ),
    MolecularBenchmarkSpec(
        LIH_STO3G_FULL_JW.name,
        "LiH",
        "STO-3G",
        12,
        LIH_STO3G_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        BEH2_STO3G_FULL_JW.name,
        "BeH2",
        "STO-3G",
        14,
        BEH2_STO3G_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        H2O_STO3G_FULL_JW.name,
        "H2O",
        "STO-3G",
        14,
        H2O_STO3G_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        NH3_STO3G_FULL_JW.name,
        "NH3",
        "STO-3G",
        16,
        NH3_STO3G_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        CH4_STO3G_FULL_JW.name,
        "CH4",
        "STO-3G",
        18,
        CH4_STO3G_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        N2_STO3G_FULL_JW.name,
        "N2",
        "STO-3G",
        20,
        N2_STO3G_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        H2O_CCPVDZ_FULL_JW.name,
        "H2O",
        "cc-pVDZ",
        48,
        H2O_CCPVDZ_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        CH4_CCPVDZ_FULL_JW.name,
        "CH4",
        "cc-pVDZ",
        68,
        CH4_CCPVDZ_FULL_JW,
    ),
    MolecularBenchmarkSpec(
        FEMOCO_NAME,
        "FeMoco",
        "Reiher CAS(54e,54o)",
        108,
        None,
    ),
)
BENCHMARK_BY_ID = {
    benchmark.system_id: benchmark for benchmark in MOLECULAR_BENCHMARKS
}
DEFAULT_SYSTEMS = tuple(BENCHMARK_BY_ID)


@dataclass(frozen=True)
class MolecularSweepConfig:
    systems: tuple[str, ...] = DEFAULT_SYSTEMS
    femoco_fcidump: str | None = None
    statevector_max_qubits: int = 16
    number_of_steps: int = 100
    num_initial_states: int = 20
    num_trajectories: int = 200
    max_group_size: int = 0
    coefficient_tolerance: float = 1e-12
    base_seed: int = 42
    num_workers: int = DEFAULT_NUM_WORKERS
    trajectory_chunks_per_state: int = DEFAULT_NUM_WORKERS

    def __post_init__(self) -> None:
        systems = tuple(self.systems)
        object.__setattr__(self, "systems", systems)
        if not systems:
            raise ValueError("systems must not be empty")
        if len(set(systems)) != len(systems):
            raise ValueError("systems must not contain duplicates")
        unknown = set(systems) - BENCHMARK_BY_ID.keys()
        if unknown:
            raise ValueError(f"unknown molecular systems: {sorted(unknown)}")
        if FEMOCO_NAME in systems:
            if not self.femoco_fcidump:
                raise ValueError(
                    "femoco_fcidump is required when FeMoco is selected"
                )
            object.__setattr__(
                self,
                "femoco_fcidump",
                str(Path(self.femoco_fcidump).expanduser().resolve()),
            )
        positive_integers = {
            "number_of_steps": self.number_of_steps,
            "num_initial_states": self.num_initial_states,
            "num_trajectories": self.num_trajectories,
            "num_workers": self.num_workers,
            "trajectory_chunks_per_state": self.trajectory_chunks_per_state,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.statevector_max_qubits, int):
            raise ValueError("statevector_max_qubits must be an integer")
        if self.num_trajectories < 2:
            raise ValueError("num_trajectories must be at least 2")
        if self.max_group_size < 0:
            raise ValueError("max_group_size must be non-negative")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        tolerance = float(self.coefficient_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "coefficient_tolerance must be finite and non-negative"
            )

    @property
    def total_tasks(self) -> int:
        return len(self.systems)

    @property
    def config_id(self) -> str:
        serialized = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> MolecularSweepConfig:
        normalized = dict(values)
        normalized["systems"] = tuple(normalized["systems"])
        return cls(**normalized)

    def effective_max_group_size(self, num_qubits: int) -> int:
        return num_qubits if self.max_group_size == 0 else self.max_group_size


@dataclass(frozen=True)
class MolecularSweepTask:
    task_id: int
    spec: MolecularBenchmarkSpec


@dataclass(frozen=True)
class TauOnlyResult:
    pauli: TauMEstimate
    grouped: TauMEstimate
    num_pauli_terms: int
    num_groups: int
    group_size_min: int
    group_size_mean: float
    group_size_max: int


def iter_sweep_tasks(
    config: MolecularSweepConfig,
) -> tuple[MolecularSweepTask, ...]:
    return tuple(
        MolecularSweepTask(task_id, BENCHMARK_BY_ID[system_id])
        for task_id, system_id in enumerate(config.systems)
    )


def run_sweep_shard(
    config: MolecularSweepConfig,
    output_directory: str | Path,
    *,
    task_index: int = 0,
    num_tasks: int = 1,
    overwrite: bool = False,
    max_tasks: int | None = None,
) -> tuple[int, int]:
    """Run one local/PBS shard and checkpoint every molecule atomically."""
    if not isinstance(num_tasks, int) or num_tasks <= 0:
        raise ValueError("num_tasks must be a positive integer")
    if not isinstance(task_index, int) or not 0 <= task_index < num_tasks:
        raise ValueError("task_index must satisfy 0 <= index < num_tasks")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when specified")

    output = Path(output_directory).expanduser().resolve()
    result_directory = output / "systems"
    result_directory.mkdir(parents=True, exist_ok=True)
    _ensure_configuration(output, config)
    assigned = tuple(
        task
        for task in iter_sweep_tasks(config)
        if task.task_id % num_tasks == task_index
    )
    if max_tasks is not None:
        assigned = assigned[:max_tasks]

    completed = 0
    skipped = 0
    print(
        f"Molecular shard {task_index + 1}/{num_tasks}: "
        f"{len(assigned)} systems",
        flush=True,
    )
    for position, task in enumerate(assigned, start=1):
        destination = _result_path(result_directory, task)
        if destination.exists() and not overwrite:
            _validate_result(_load_json(destination), config, task, destination)
            skipped += 1
            print(
                f"[{position}/{len(assigned)}] {task.spec.system_id}: skip",
                flush=True,
            )
            continue
        print(
            f"[{position}/{len(assigned)}] {task.spec.system_id}: run",
            flush=True,
        )
        result = run_molecular_system(config, task)
        _atomic_write_json(destination, result)
        completed += 1
        _print_result(result)
    return completed, skipped


def run_molecular_system(
    config: MolecularSweepConfig,
    task: MolecularSweepTask,
) -> dict[str, object]:
    start = time.perf_counter()
    preset = _preset_for_task(config, task)
    generated = preset.generate(
        coefficient_tolerance=config.coefficient_tolerance
    )
    if generated.num_qubits != task.spec.expected_qubits:
        raise ValueError(
            f"{task.spec.system_id} generated {generated.num_qubits} qubits; "
            f"expected {task.spec.expected_qubits}"
        )
    max_group_size = config.effective_max_group_size(generated.num_qubits)
    actual_metrics_computed = (
        generated.num_qubits <= config.statevector_max_qubits
    )

    if actual_metrics_computed:
        # Reuse the generated object rather than running PySCF a second time.
        target = generated.to_lcp(include_identity=False)
        from hamiltonians.pauli import build_lch_from_lcp_unit_cost
        from grouping import build_chemistry_depth1_frobenius_lch
        from qdrift_metric_comparison import QDriftDecompositions

        decompositions = QDriftDecompositions(
            preset=preset,
            generated=generated,
            target=target,
            pauli=build_lch_from_lcp_unit_cost(target),
            grouped=build_chemistry_depth1_frobenius_lch(
                target, max_group_size=max_group_size
            ),
            max_group_size=max_group_size,
        )
        pauli_lambda = decompositions.pauli.coefficient_one_norm()
        grouped_lambda = decompositions.grouped.coefficient_one_norm()
        evolution_time = _lambda_normalized_time(pauli_lambda)
        experiment = run_experiment_from_decompositions(
            decompositions,
            total_time=evolution_time,
            number_of_steps=config.number_of_steps,
            num_initial_states=config.num_initial_states,
            num_trajectories=config.num_trajectories,
            seed=_indexed_seed(config.base_seed, task.task_id),
            num_workers=config.num_workers,
            trajectory_chunks_per_state=config.trajectory_chunks_per_state,
        )
        pauli_tau, grouped_tau = experiment.tau_m
        pauli_metrics, grouped_metrics = experiment.trajectory_metrics
        group_sizes = tuple(
            len(operator.terms)
            for _, operator in decompositions.grouped.terms
        )
        tau_result = TauOnlyResult(
            pauli_tau,
            grouped_tau,
            len(target),
            len(group_sizes),
            min(group_sizes),
            float(np.mean(group_sizes)),
            max(group_sizes),
        )
        infidelity_pauli = pauli_metrics.mean_infidelity
        infidelity_grouped = grouped_metrics.mean_infidelity
        qpe_pauli = pauli_metrics.mean_absolute_qpe_signal_error
        qpe_grouped = grouped_metrics.mean_absolute_qpe_signal_error
        infidelity_se_pauli = pauli_metrics.infidelity.standard_error
        infidelity_se_grouped = grouped_metrics.infidelity.standard_error
        qpe_se_pauli = (
            pauli_metrics.absolute_qpe_signal_error.standard_error
        )
        qpe_se_grouped = (
            grouped_metrics.absolute_qpe_signal_error.standard_error
        )
    else:
        tau_result = _calculate_tau_only_from_generated(
            generated, max_group_size=max_group_size
        )
        pauli_tau = tau_result.pauli
        grouped_tau = tau_result.grouped
        pauli_lambda = pauli_tau.lambda_sum
        grouped_lambda = grouped_tau.lambda_sum
        evolution_time = _lambda_normalized_time(pauli_lambda)
        infidelity_pauli = None
        infidelity_grouped = None
        qpe_pauli = None
        qpe_grouped = None
        infidelity_se_pauli = None
        infidelity_se_grouped = None
        qpe_se_pauli = None
        qpe_se_grouped = None

    return {
        "format_version": RESULT_FORMAT_VERSION,
        "config_id": config.config_id,
        "task_id": task.task_id,
        "system_id": task.spec.system_id,
        "molecule": task.spec.molecule,
        "basis": task.spec.basis,
        "description": preset.description,
        "num_qubits": generated.num_qubits,
        "actual_metrics_computed": actual_metrics_computed,
        "max_group_size": max_group_size,
        "coefficient_tolerance": config.coefficient_tolerance,
        "num_pauli_terms": tau_result.num_pauli_terms,
        "num_groups": tau_result.num_groups,
        "group_size_min": tau_result.group_size_min,
        "group_size_mean": tau_result.group_size_mean,
        "group_size_max": tau_result.group_size_max,
        "hartree_fock_energy": generated.hartree_fock_energy,
        "removed_identity": generated.identity_coefficient,
        "evolution_time": evolution_time,
        "number_of_steps": config.number_of_steps,
        "step_time": evolution_time / config.number_of_steps,
        "pauli_lambda": pauli_lambda,
        "grouped_lambda": grouped_lambda,
        "pauli_lambda_time": pauli_lambda * evolution_time,
        "grouped_lambda_time": grouped_lambda * evolution_time,
        "tau_m_pauli": pauli_tau.tau_m,
        "tau_m_grouped": grouped_tau.tau_m,
        "tau_m_factor": _ratio(pauli_tau.tau_m, grouped_tau.tau_m),
        "infidelity_pauli": infidelity_pauli,
        "infidelity_grouped": infidelity_grouped,
        "infidelity_factor": _optional_ratio(
            infidelity_pauli, infidelity_grouped
        ),
        "infidelity_se_pauli": infidelity_se_pauli,
        "infidelity_se_grouped": infidelity_se_grouped,
        "qpe_signal_error_pauli": qpe_pauli,
        "qpe_signal_error_grouped": qpe_grouped,
        "qpe_signal_error_factor": _optional_ratio(qpe_pauli, qpe_grouped),
        "qpe_signal_error_se_pauli": qpe_se_pauli,
        "qpe_signal_error_se_grouped": qpe_se_grouped,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _preset_for_task(
    config: MolecularSweepConfig,
    task: MolecularSweepTask,
) -> MolecularHamiltonianPreset | ReiherFeMocoHamiltonianPreset:
    if task.spec.system_id == FEMOCO_NAME:
        assert config.femoco_fcidump is not None
        return ReiherFeMocoHamiltonianPreset(config.femoco_fcidump)
    assert task.spec.preset is not None
    return task.spec.preset


def _calculate_tau_only_from_generated(
    generated: object,
    *,
    max_group_size: int,
) -> TauOnlyResult:
    """Compute both trace factors without constructing state-vector objects."""
    target = generated.to_lcp(include_identity=False)
    coefficient_squares = math.fsum(
        coefficient * coefficient for _, coefficient in target.term_items()
    )
    pauli_lambda = math.fsum(
        abs(coefficient) for _, coefficient in target.term_items()
    )
    pauli_tau = TauMEstimate(
        lambda_sum=pauli_lambda,
        tau_h_squared=coefficient_squares,
        tau_sample_second_moment=pauli_lambda * pauli_lambda,
        tau_m=max(0.0, pauli_lambda * pauli_lambda - coefficient_squares),
    )
    groups = chemistry_depth_one_groups(
        target, max_group_size=max_group_size
    )
    group_sizes = tuple(len(group.terms) for group in groups)
    grouped_lambda = math.fsum(
        math.sqrt(
            math.fsum(value * value for value in group.terms.values())
        )
        for group in groups
    )
    grouped_tau = TauMEstimate(
        lambda_sum=grouped_lambda,
        tau_h_squared=coefficient_squares,
        tau_sample_second_moment=grouped_lambda * grouped_lambda,
        tau_m=max(0.0, grouped_lambda * grouped_lambda - coefficient_squares),
    )
    return TauOnlyResult(
        pauli=pauli_tau,
        grouped=grouped_tau,
        num_pauli_terms=len(target),
        num_groups=len(groups),
        group_size_min=min(group_sizes),
        group_size_mean=float(np.mean(group_sizes)),
        group_size_max=max(group_sizes),
    )


def aggregate_sweep(
    output_directory: str | Path,
    *,
    allow_incomplete: bool = False,
    make_plot: bool = True,
) -> tuple[Path, Path | None]:
    output = Path(output_directory).expanduser().resolve()
    config = _load_configuration(output)
    records: list[dict[str, object]] = []
    missing: list[MolecularSweepTask] = []
    result_directory = output / "systems"
    for task in iter_sweep_tasks(config):
        path = _result_path(result_directory, task)
        if not path.exists():
            missing.append(task)
            continue
        record = _load_json(path)
        _validate_result(record, config, task, path)
        records.append(record)
    if missing and not allow_incomplete:
        names = ", ".join(task.spec.system_id for task in missing)
        raise ValueError(
            f"{len(missing)} of {config.total_tasks} systems are missing "
            f"({names}); rerun the PBS array or use --allow-incomplete"
        )
    if not records:
        raise ValueError("no completed molecular results were found")

    csv_path = output / "molecular_results.csv"
    _atomic_write_csv(csv_path, list(records[0]), records)
    plot_path = None
    if make_plot:
        plot_path = output / "molecular_improvement_factors.png"
        _plot_results(
            records,
            plot_path,
            output / "molecular_improvement_factors.pdf",
        )
    print(
        f"Aggregated {len(records)}/{config.total_tasks} systems; "
        f"missing={len(missing)}",
        flush=True,
    )
    return csv_path, plot_path


def _plot_results(
    records: list[dict[str, object]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    matplotlib_directory = png_path.parent / ".matplotlib"
    matplotlib_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_directory))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.labelsize": 20,
            "xtick.labelsize": 13,
            "ytick.labelsize": 16,
        }
    )
    import matplotlib.pyplot as plt

    labels = {
        "tau_m_factor": r"$I_M$: Error upper bound",
        "infidelity_factor": r"$I_r$: Average infidelity",
        "qpe_signal_error_factor": (
            r"$I_{\mathrm{QPE}}$: Average QPE signal error"
        ),
    }
    colors = {
        "tau_m_factor": "#1f77b4",
        "infidelity_factor": "#d95f02",
        "qpe_signal_error_factor": "#2ca02c",
    }
    x_values = np.arange(len(records))
    figure, axis = plt.subplots(figsize=(12.0, 6.4), constrained_layout=True)
    for metric in METRICS:
        selected = [
            (index, float(record[metric]))
            for index, record in enumerate(records)
            if record[metric] is not None
            and math.isfinite(float(record[metric]))
        ]
        if selected:
            axis.plot(
                [item[0] for item in selected],
                [item[1] for item in selected],
                marker="o",
                linewidth=1.7,
                color=colors[metric],
                label=labels[metric],
            )
    tick_labels = [
        f"{record['molecule']}\n{record['basis']}\n({record['num_qubits']} q)"
        for record in records
    ]
    axis.set_xticks(x_values, tick_labels)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Molecular system")
    axis.set_ylabel("Improvement factor")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="upper left", fontsize=15, frameon=False)
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)


def _configuration_path(output: Path) -> Path:
    return output / "configuration.json"


def _ensure_configuration(
    output: Path,
    config: MolecularSweepConfig,
) -> None:
    path = _configuration_path(output)
    if path.exists():
        if _load_configuration(output) != config:
            raise ValueError(
                f"output directory has a different configuration: {path}"
            )
        return
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        path,
        {
            "format_version": RESULT_FORMAT_VERSION,
            "config_id": config.config_id,
            "config": config.to_dict(),
        },
    )
    if _load_configuration(output) != config:
        raise ValueError("configuration changed concurrently")


def _load_configuration(output: Path) -> MolecularSweepConfig:
    path = _configuration_path(output)
    payload = _load_json(path)
    if payload.get("format_version") != RESULT_FORMAT_VERSION:
        raise ValueError(f"unsupported configuration format: {path}")
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"invalid configuration: {path}")
    config = MolecularSweepConfig.from_dict(raw_config)
    if payload.get("config_id") != config.config_id:
        raise ValueError(f"configuration hash mismatch: {path}")
    return config


def _result_path(
    directory: Path,
    task: MolecularSweepTask,
) -> Path:
    return directory / f"{task.task_id:02d}_{task.spec.system_id}.json"


def _validate_result(
    result: dict[str, object],
    config: MolecularSweepConfig,
    task: MolecularSweepTask,
    path: Path,
) -> None:
    expected = {
        "format_version": RESULT_FORMAT_VERSION,
        "config_id": config.config_id,
        "task_id": task.task_id,
        "system_id": task.spec.system_id,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(
                f"result field {name!r} does not match in {path}: "
                f"expected {value!r}, got {result.get(name)!r}"
            )
    for metric in METRICS:
        if metric not in result:
            raise ValueError(f"result is missing {metric!r}: {path}")


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _indexed_seed(root_seed: int, task_id: int) -> int:
    sequence = np.random.SeedSequence(root_seed, spawn_key=(task_id,))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _ratio(pauli: float, grouped: float) -> float:
    if grouped == 0.0:
        return 0.0 if pauli == 0.0 else math.inf
    return pauli / grouped


def _optional_ratio(
    pauli: float | None,
    grouped: float | None,
) -> float | None:
    if pauli is None or grouped is None:
        return None
    return _ratio(pauli, grouped)


def _lambda_normalized_time(pauli_lambda: float) -> float:
    if not math.isfinite(pauli_lambda) or pauli_lambda <= 0.0:
        raise ValueError("Pauli lambda must be positive and finite")
    return 1.0 / pauli_lambda


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _print_result(result: dict[str, object]) -> None:
    print(
        f"  qubits={result['num_qubits']}, terms={result['num_pauli_terms']}, "
        f"groups={result['num_groups']}",
        flush=True,
    )
    print(
        "  factors: "
        f"I_M={_format_optional(result['tau_m_factor'])}, "
        f"I_r={_format_optional(result['infidelity_factor'])}, "
        f"I_QPE={_format_optional(result['qpe_signal_error_factor'])}, "
        f"elapsed={float(result['elapsed_seconds']):.2f}s",
        flush=True,
    )


def _add_config_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(BENCHMARK_BY_ID),
        default=list(DEFAULT_SYSTEMS),
    )
    parser.add_argument("--femoco-fcidump", type=Path)
    parser.add_argument("--statevector-max-qubits", type=int, default=16)
    parser.add_argument("--number-of-steps", type=int, default=100)
    parser.add_argument("--num-initial-states", type=int, default=20)
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=0,
        help="0 (default) uses the current qubit count",
    )
    parser.add_argument("--coefficient-tolerance", type=float, default=1e-12)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--trajectory-chunks-per-state",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )


def _config_from_arguments(arguments: Namespace) -> MolecularSweepConfig:
    return MolecularSweepConfig(
        systems=tuple(arguments.systems),
        femoco_fcidump=(
            None
            if arguments.femoco_fcidump is None
            else str(arguments.femoco_fcidump)
        ),
        statevector_max_qubits=arguments.statevector_max_qubits,
        number_of_steps=arguments.number_of_steps,
        num_initial_states=arguments.num_initial_states,
        num_trajectories=arguments.num_trajectories,
        max_group_size=arguments.max_group_size,
        coefficient_tolerance=arguments.coefficient_tolerance,
        base_seed=arguments.base_seed,
        num_workers=arguments.num_workers,
        trajectory_chunks_per_state=arguments.trajectory_chunks_per_state,
    )


def main() -> None:
    parser = ArgumentParser(
        description="Run the fixed molecular Pauli/grouped qDRIFT benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one local/PBS shard")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--task-index", type=int, default=0)
    run_parser.add_argument("--num-tasks", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--max-tasks", type=int)
    _add_config_arguments(run_parser)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="validate checkpoints and generate CSV/plots"
    )
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser.add_argument("--allow-incomplete", action="store_true")
    aggregate_parser.add_argument("--no-plot", action="store_true")
    arguments = parser.parse_args()

    if arguments.command == "run":
        completed, skipped = run_sweep_shard(
            _config_from_arguments(arguments),
            arguments.output_dir,
            task_index=arguments.task_index,
            num_tasks=arguments.num_tasks,
            overwrite=arguments.overwrite,
            max_tasks=arguments.max_tasks,
        )
        print(f"Shard finished: completed={completed}, skipped={skipped}")
    else:
        csv_path, plot_path = aggregate_sweep(
            arguments.output_dir,
            allow_incomplete=arguments.allow_incomplete,
            make_plot=not arguments.no_plot,
        )
        print(f"CSV:  {csv_path}")
        if plot_path is not None:
            print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
