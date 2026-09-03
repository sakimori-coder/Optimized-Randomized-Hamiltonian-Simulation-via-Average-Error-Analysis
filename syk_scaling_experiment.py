r"""SYK scaling sweep for Pauli/grouped qDRIFT improvement factors.

For every qubit count and disorder realization this script always computes
the exact coefficient-time ``tau(M_p)`` factor.  State-vector estimates of
average infidelity and average QPE signal error are computed only up to the
configured exact-simulation cutoff.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from grouping import chemistry_depth_one_groups
from hamiltonians.syk import SykHamiltonianPreset
from qdrift_metric_comparison import DEFAULT_NUM_WORKERS, run_experiment
from qdrift_trajectory_metrics import TauMEstimate


RESULT_FORMAT_VERSION = 1
METRICS = (
    "tau_m_factor",
    "infidelity_factor",
    "qpe_signal_error_factor",
)


@dataclass(frozen=True)
class SykSweepConfig:
    min_qubits: int = 5
    max_qubits: int = 50
    num_realizations: int = 50
    statevector_max_qubits: int = 15
    coupling_scale: float = 1.0
    total_time: float = 0.3
    number_of_steps: int = 20
    num_initial_states: int = 20
    num_trajectories: int = 200
    max_group_size: int = 0
    base_seed: int = 42
    num_workers: int = DEFAULT_NUM_WORKERS
    trajectory_chunks_per_state: int = DEFAULT_NUM_WORKERS

    def __post_init__(self) -> None:
        positive_integers = {
            "min_qubits": self.min_qubits,
            "max_qubits": self.max_qubits,
            "num_realizations": self.num_realizations,
            "number_of_steps": self.number_of_steps,
            "num_initial_states": self.num_initial_states,
            "num_trajectories": self.num_trajectories,
            "num_workers": self.num_workers,
            "trajectory_chunks_per_state": self.trajectory_chunks_per_state,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_qubits < 2:
            raise ValueError("min_qubits must be at least 2")
        if self.max_qubits < self.min_qubits:
            raise ValueError("max_qubits must not be smaller than min_qubits")
        if not isinstance(self.statevector_max_qubits, int):
            raise ValueError("statevector_max_qubits must be an integer")
        if self.num_trajectories < 2:
            raise ValueError("num_trajectories must be at least 2")
        if self.max_group_size < 0:
            raise ValueError("max_group_size must be non-negative")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        for name, value in {
            "coupling_scale": self.coupling_scale,
            "total_time": self.total_time,
        }.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.coupling_scale < 0.0:
            raise ValueError("coupling_scale must be non-negative")

    @property
    def total_tasks(self) -> int:
        return (
            self.max_qubits - self.min_qubits + 1
        ) * self.num_realizations

    @property
    def config_id(self) -> str:
        serialized = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> SykSweepConfig:
        return cls(**values)

    def effective_max_group_size(self, num_qubits: int) -> int:
        return num_qubits if self.max_group_size == 0 else self.max_group_size


@dataclass(frozen=True)
class SykSweepTask:
    task_id: int
    num_qubits: int
    realization: int


def iter_sweep_tasks(config: SykSweepConfig) -> tuple[SykSweepTask, ...]:
    tasks: list[SykSweepTask] = []
    for num_qubits in range(config.min_qubits, config.max_qubits + 1):
        for realization in range(config.num_realizations):
            tasks.append(
                SykSweepTask(
                    task_id=len(tasks),
                    num_qubits=num_qubits,
                    realization=realization,
                )
            )
    return tuple(tasks)


def run_sweep_shard(
    config: SykSweepConfig,
    output_directory: str | Path,
    *,
    task_index: int = 0,
    num_tasks: int = 1,
    overwrite: bool = False,
    max_tasks: int | None = None,
    tau_workers: int = 1,
) -> tuple[int, int]:
    """Run one local/PBS shard and checkpoint every realization atomically."""
    if not isinstance(num_tasks, int) or num_tasks <= 0:
        raise ValueError("num_tasks must be a positive integer")
    if not isinstance(task_index, int) or not 0 <= task_index < num_tasks:
        raise ValueError("task_index must satisfy 0 <= index < num_tasks")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive when specified")
    if not isinstance(tau_workers, int) or tau_workers <= 0:
        raise ValueError("tau_workers must be a positive integer")

    output = Path(output_directory).expanduser().resolve()
    result_directory = output / "realizations"
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
        f"Sweep shard {task_index + 1}/{num_tasks}: "
        f"{len(assigned)} realization tasks",
        flush=True,
    )
    pending_actual: list[SykSweepTask] = []
    pending_tau: list[SykSweepTask] = []
    for position, task in enumerate(assigned, start=1):
        destination = _result_path(result_directory, task)
        if destination.exists() and not overwrite:
            _validate_existing_result(destination, config, task)
            skipped += 1
            print(
                f"[{position}/{len(assigned)}] "
                f"q={task.num_qubits}, realization={task.realization}: skip",
                flush=True,
            )
            continue
        if task.num_qubits <= config.statevector_max_qubits:
            pending_actual.append(task)
        else:
            pending_tau.append(task)

    for position, task in enumerate(pending_actual, start=1):
        print(
            f"[state-vector {position}/{len(pending_actual)}] "
            f"q={task.num_qubits}, realization={task.realization}: run",
            flush=True,
        )
        result = run_syk_realization(config, task)
        _atomic_write_json(_result_path(result_directory, task), result)
        completed += 1
        _print_result_factors(result)

    if pending_tau:
        print(
            f"Starting {len(pending_tau)} tau-only tasks with "
            f"{min(tau_workers, len(pending_tau))} realization workers",
            flush=True,
        )
        if tau_workers == 1:
            tau_results = (
                (task, run_syk_realization(config, task))
                for task in pending_tau
            )
            for task, result in tau_results:
                _atomic_write_json(_result_path(result_directory, task), result)
                completed += 1
                print(
                    f"[tau {completed}/{len(assigned) - skipped}] "
                    f"q={task.num_qubits}, realization={task.realization}",
                    flush=True,
                )
                _print_result_factors(result)
        else:
            with ProcessPoolExecutor(
                max_workers=min(tau_workers, len(pending_tau))
            ) as executor:
                futures = {
                    executor.submit(run_syk_realization, config, task): task
                    for task in pending_tau
                }
                for future in as_completed(futures):
                    task = futures[future]
                    result = future.result()
                    _atomic_write_json(
                        _result_path(result_directory, task),
                        result,
                    )
                    completed += 1
                    print(
                        f"[tau {completed}/{len(assigned) - skipped}] "
                        f"q={task.num_qubits}, realization={task.realization}",
                        flush=True,
                    )
                    _print_result_factors(result)
    return completed, skipped


def run_syk_realization(
    config: SykSweepConfig,
    task: SykSweepTask,
) -> dict[str, object]:
    """Compute one seeded disorder realization."""
    start = time.perf_counter()
    syk_seed = _indexed_seed(
        config.base_seed,
        task.num_qubits,
        task.realization,
        0,
    )
    monte_carlo_seed = _indexed_seed(
        config.base_seed,
        task.num_qubits,
        task.realization,
        1,
    )
    preset = SykHamiltonianPreset(
        num_qubits=task.num_qubits,
        coupling_scale=config.coupling_scale,
        seed=syk_seed,
    )
    max_group_size = config.effective_max_group_size(task.num_qubits)

    if task.num_qubits <= config.statevector_max_qubits:
        experiment = run_experiment(
            preset,
            total_time=config.total_time,
            number_of_steps=config.number_of_steps,
            num_initial_states=config.num_initial_states,
            num_trajectories=config.num_trajectories,
            max_group_size=max_group_size,
            seed=monte_carlo_seed,
            num_workers=config.num_workers,
            trajectory_chunks_per_state=(
                config.trajectory_chunks_per_state
            ),
        )
        pauli_tau, grouped_tau = experiment.tau_m
        pauli_metrics, grouped_metrics = experiment.trajectory_metrics
        num_pauli_terms = len(experiment.decompositions.target)
        num_groups = len(experiment.decompositions.grouped.terms)
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
        (
            pauli_tau,
            grouped_tau,
            num_pauli_terms,
            num_groups,
        ) = _calculate_tau_only(preset, max_group_size=max_group_size)
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
        "num_qubits": task.num_qubits,
        "realization": task.realization,
        "syk_seed": syk_seed,
        "monte_carlo_seed": monte_carlo_seed,
        "actual_metrics_computed": (
            task.num_qubits <= config.statevector_max_qubits
        ),
        "max_group_size": max_group_size,
        "num_pauli_terms": num_pauli_terms,
        "num_groups": num_groups,
        "tau_m_pauli": pauli_tau.tau_m,
        "tau_m_grouped": grouped_tau.tau_m,
        "tau_m_factor": _ratio(pauli_tau.tau_m, grouped_tau.tau_m),
        "infidelity_pauli": infidelity_pauli,
        "infidelity_grouped": infidelity_grouped,
        "infidelity_factor": _optional_ratio(
            infidelity_pauli,
            infidelity_grouped,
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


def aggregate_sweep(
    output_directory: str | Path,
    *,
    allow_incomplete: bool = False,
    make_plot: bool = True,
) -> tuple[Path, Path, Path | None]:
    """Validate results, write raw/summary CSVs, and plot factor distributions."""
    output = Path(output_directory).expanduser().resolve()
    config = _load_configuration(output)
    records: list[dict[str, object]] = []
    missing: list[SykSweepTask] = []
    result_directory = output / "realizations"
    for task in iter_sweep_tasks(config):
        path = _result_path(result_directory, task)
        if not path.exists():
            missing.append(task)
            continue
        record = _load_json(path)
        _validate_result(record, config, task, path)
        records.append(record)
    if missing and not allow_incomplete:
        preview = ", ".join(
            f"q={task.num_qubits}/r={task.realization}"
            for task in missing[:8]
        )
        raise ValueError(
            f"{len(missing)} of {config.total_tasks} results are missing "
            f"({preview}); rerun the PBS array or use --allow-incomplete"
        )
    if not records:
        raise ValueError("no completed realization results were found")

    raw_path = output / "syk_realizations.csv"
    summary_path = output / "syk_summary.csv"
    _write_raw_csv(raw_path, records)
    summary_rows = _summary_rows(config, records)
    _write_summary_csv(summary_path, summary_rows)
    plot_path = None
    if make_plot:
        plot_path = output / "syk_improvement_factors.png"
        _plot_summary(
            summary_rows,
            plot_path,
            output / "syk_improvement_factors.pdf",
        )
    print(
        f"Aggregated {len(records)}/{config.total_tasks} realizations; "
        f"missing={len(missing)}",
        flush=True,
    )
    return raw_path, summary_path, plot_path


def _calculate_tau_only(
    preset: SykHamiltonianPreset,
    *,
    max_group_size: int,
) -> tuple[TauMEstimate, TauMEstimate, int, int]:
    """Avoid Pauli-LCH and state-vector construction above the cutoff."""
    generated = preset.generate()
    target = generated.to_lcp(include_identity=False)
    del generated
    coefficient_squares = math.fsum(
        coefficient * coefficient
        for _, coefficient in target.term_items()
    )
    pauli_lambda = math.fsum(
        abs(value) for _, value in target.term_items()
    )
    pauli_tau = TauMEstimate(
        lambda_sum=pauli_lambda,
        tau_h_squared=coefficient_squares,
        tau_sample_second_moment=pauli_lambda * pauli_lambda,
        tau_m=max(0.0, pauli_lambda * pauli_lambda - coefficient_squares),
    )

    groups = chemistry_depth_one_groups(
        target,
        max_group_size=max_group_size,
    )
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
    return pauli_tau, grouped_tau, len(target), len(groups)


def _summary_rows(
    config: SykSweepConfig,
    records: list[dict[str, object]],
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for num_qubits in range(config.min_qubits, config.max_qubits + 1):
        qubit_records = [
            record
            for record in records
            if int(record["num_qubits"]) == num_qubits
        ]
        for metric in METRICS:
            values = np.asarray(
                [
                    float(record[metric])
                    for record in qubit_records
                    if record[metric] is not None
                    and math.isfinite(float(record[metric]))
                ],
                dtype=np.float64,
            )
            row: dict[str, int | float | str] = {
                "num_qubits": num_qubits,
                "metric": metric,
                "expected_realizations": config.num_realizations,
                "completed_realizations": len(qubit_records),
                "finite_count": int(values.size),
            }
            if values.size:
                sample_std = (
                    0.0 if values.size == 1 else float(np.std(values, ddof=1))
                )
                quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
                positive = values[values > 0.0]
                row.update(
                    {
                        "mean": float(np.mean(values)),
                        "sample_std": sample_std,
                        "standard_error": sample_std / math.sqrt(values.size),
                        "geometric_mean": (
                            float(np.exp(np.mean(np.log(positive))))
                            if positive.size == values.size
                            else math.nan
                        ),
                        "q10": float(quantiles[0]),
                        "q25": float(quantiles[1]),
                        "median": float(quantiles[2]),
                        "q75": float(quantiles[3]),
                        "q90": float(quantiles[4]),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )
            else:
                row.update(
                    {
                        name: math.nan
                        for name in (
                            "mean",
                            "sample_std",
                            "standard_error",
                            "geometric_mean",
                            "q10",
                            "q25",
                            "median",
                            "q75",
                            "q90",
                            "minimum",
                            "maximum",
                        )
                    }
                )
            rows.append(row)
    return rows


def _write_raw_csv(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = list(records[0])
    _atomic_write_csv(path, fieldnames, records)


def _write_summary_csv(
    path: Path,
    rows: list[dict[str, int | float | str]],
) -> None:
    _atomic_write_csv(path, list(rows[0]), rows)


def _atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _plot_summary(
    rows: list[dict[str, int | float | str]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    matplotlib_directory = png_path.parent / ".matplotlib"
    matplotlib_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_directory))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "tau_m_factor": r"$\tau(M_p)$",
        "infidelity_factor": "average infidelity",
        "qpe_signal_error_factor": "average QPE signal error",
    }
    colors = {
        "tau_m_factor": "#1f77b4",
        "infidelity_factor": "#d95f02",
        "qpe_signal_error_factor": "#2ca02c",
    }
    figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    for metric in METRICS:
        selected = [
            row
            for row in rows
            if row["metric"] == metric
            and int(row["finite_count"]) > 0
            and float(row["median"]) > 0.0
        ]
        if not selected:
            continue
        qubits = np.asarray(
            [int(row["num_qubits"]) for row in selected],
            dtype=np.int64,
        )
        median = np.asarray([float(row["median"]) for row in selected])
        q10 = np.asarray([float(row["q10"]) for row in selected])
        q90 = np.asarray([float(row["q90"]) for row in selected])
        axis.plot(
            qubits,
            median,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color=colors[metric],
            label=f"{labels[metric]} median",
        )
        axis.fill_between(
            qubits,
            q10,
            q90,
            color=colors[metric],
            alpha=0.16,
            linewidth=0.0,
            label=f"{labels[metric]} 10–90%",
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_yscale("log")
    axis.set_xlabel("Number of qubits")
    axis.set_ylabel("Improvement factor (Pauli / grouped)")
    axis.set_title("SYK qDRIFT improvement over disorder realizations")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)


def _configuration_path(output: Path) -> Path:
    return output / "configuration.json"


def _ensure_configuration(output: Path, config: SykSweepConfig) -> None:
    path = _configuration_path(output)
    if path.exists():
        existing = _load_configuration(output)
        if existing != config:
            raise ValueError(
                f"output directory already has a different configuration: {path}"
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
    existing = _load_configuration(output)
    if existing != config:
        raise ValueError("configuration changed concurrently in output directory")


def _load_configuration(output: Path) -> SykSweepConfig:
    path = _configuration_path(output)
    payload = _load_json(path)
    if payload.get("format_version") != RESULT_FORMAT_VERSION:
        raise ValueError(f"unsupported configuration format: {path}")
    config_value = payload.get("config")
    if not isinstance(config_value, dict):
        raise ValueError(f"invalid configuration: {path}")
    config = SykSweepConfig.from_dict(config_value)
    if payload.get("config_id") != config.config_id:
        raise ValueError(f"configuration hash mismatch: {path}")
    return config


def _result_path(directory: Path, task: SykSweepTask) -> Path:
    return directory / (
        f"q{task.num_qubits:03d}_r{task.realization:03d}.json"
    )


def _validate_existing_result(
    path: Path,
    config: SykSweepConfig,
    task: SykSweepTask,
) -> None:
    _validate_result(_load_json(path), config, task, path)


def _validate_result(
    result: dict[str, object],
    config: SykSweepConfig,
    task: SykSweepTask,
    path: Path,
) -> None:
    expected = {
        "format_version": RESULT_FORMAT_VERSION,
        "config_id": config.config_id,
        "task_id": task.task_id,
        "num_qubits": task.num_qubits,
        "realization": task.realization,
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
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
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


def _indexed_seed(root_seed: int, *coordinates: int) -> int:
    sequence = np.random.SeedSequence(root_seed, spawn_key=coordinates)
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


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _print_result_factors(result: dict[str, object]) -> None:
    print(
        "  factors: "
        f"tau={_format_optional(result['tau_m_factor'])}, "
        f"infidelity={_format_optional(result['infidelity_factor'])}, "
        f"QPE={_format_optional(result['qpe_signal_error_factor'])}, "
        f"elapsed={float(result['elapsed_seconds']):.2f}s",
        flush=True,
    )


def _add_config_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--min-qubits", type=int, default=5)
    parser.add_argument("--max-qubits", type=int, default=50)
    parser.add_argument("--num-realizations", type=int, default=50)
    parser.add_argument("--statevector-max-qubits", type=int, default=15)
    parser.add_argument("--coupling-scale", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=0.3)
    parser.add_argument("--number-of-steps", type=int, default=20)
    parser.add_argument("--num-initial-states", type=int, default=20)
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=0,
        help="0 (default) uses the current qubit count",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--trajectory-chunks-per-state",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )


def _config_from_arguments(arguments: Namespace) -> SykSweepConfig:
    return SykSweepConfig(
        min_qubits=arguments.min_qubits,
        max_qubits=arguments.max_qubits,
        num_realizations=arguments.num_realizations,
        statevector_max_qubits=arguments.statevector_max_qubits,
        coupling_scale=arguments.coupling_scale,
        total_time=arguments.time,
        number_of_steps=arguments.number_of_steps,
        num_initial_states=arguments.num_initial_states,
        num_trajectories=arguments.num_trajectories,
        max_group_size=arguments.max_group_size,
        base_seed=arguments.base_seed,
        num_workers=arguments.num_workers,
        trajectory_chunks_per_state=arguments.trajectory_chunks_per_state,
    )


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Sweep SYK disorder realizations and compare Pauli/grouped "
            "qDRIFT improvement factors"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one local/PBS shard")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--task-index", type=int, default=0)
    run_parser.add_argument("--num-tasks", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument(
        "--tau-workers",
        type=int,
        default=1,
        help="parallel realizations above statevector-max-qubits",
    )
    run_parser.add_argument(
        "--max-tasks",
        type=int,
        help="pilot/debug limit for this shard",
    )
    _add_config_arguments(run_parser)

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="validate checkpoints and generate CSV/plots",
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
            tau_workers=arguments.tau_workers,
        )
        print(f"Shard finished: completed={completed}, skipped={skipped}")
    else:
        raw, summary, plot = aggregate_sweep(
            arguments.output_dir,
            allow_incomplete=arguments.allow_incomplete,
            make_plot=not arguments.no_plot,
        )
        print(f"Raw CSV:     {raw}")
        print(f"Summary CSV: {summary}")
        if plot is not None:
            print(f"Plot:        {plot}")


if __name__ == "__main__":
    main()
