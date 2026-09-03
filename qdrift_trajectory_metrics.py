r"""State-vector Monte Carlo metrics for complete qDRIFT trajectories.

This module samples both the input state and the independently resampled
qDRIFT sequence.  For an input ``|psi>`` and ideal final state
``|phi> = exp(-i t H)|psi>``, a sampled trajectory produces
``|chi_s> = V_s|psi>``.  The two per-input quantities estimated here are

``infidelity = 1 - E_s |<phi|chi_s>|**2``

and

``QPE signal error = |<psi|phi> - E_s <psi|chi_s>|``.

In particular, the absolute value in the QPE metric is taken *after* the
complex trajectory signals have been averaged.
"""

from __future__ import annotations

import os

# Outer Monte Carlo tasks are process-parallel.  Disable nested Qulacs/OpenMP
# and BLAS threading before importing NumPy, SciPy, or Qulacs.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState

from operators import LCH, LCP
from hamiltonians.pauli import SinglePauliLCH
from statevector_simulation import (
    PreparedQDriftBranches,
    haar_random_state,
    ideal_time_evolved_state,
    prepare_qdrift_branches,
)


_INPUT_WORKER_TARGET: LCP | None = None
_INPUT_WORKER_TIME = 0.0
_TRAJECTORY_INITIAL_STATES: tuple[NDArray[np.complex128], ...] = ()
_TRAJECTORY_IDEAL_STATES: tuple[NDArray[np.complex128], ...] = ()
_TRAJECTORY_SAMPLERS: tuple[
    PreparedQDriftTrajectorySampler | None,
    ...,
] = ()


@dataclass(frozen=True)
class ScalarSampleStatistics:
    """Mean and between-input-state uncertainty of scalar observations."""

    mean: float
    sample_standard_deviation: float
    standard_error: float
    values: NDArray[np.float64]

    @classmethod
    def from_values(
        cls,
        values: NDArray[np.float64],
    ) -> ScalarSampleStatistics:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("values must be a non-empty one-dimensional array")
        immutable = np.array(array, dtype=np.float64, copy=True)
        immutable.setflags(write=False)
        sample_std = (
            0.0
            if immutable.size == 1
            else float(np.std(immutable, ddof=1))
        )
        return cls(
            mean=float(np.mean(immutable)),
            sample_standard_deviation=sample_std,
            standard_error=sample_std / math.sqrt(immutable.size),
            values=immutable,
        )


@dataclass(frozen=True)
class PreparedQDriftTrajectorySampler:
    """Qulacs circuits and global phases for one qDRIFT decomposition."""

    channel: PreparedQDriftBranches
    global_phases: NDArray[np.complex128]

    def sample_overlaps(
        self,
        initial_state: NDArray[np.complex128],
        ideal_state: NDArray[np.complex128],
        number_of_steps: int,
        num_trajectories: int,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Return trajectory fidelities and complex QPE signals.

        The sampled state vectors are consumed immediately; no density
        matrices or ``num_trajectories * 2**num_qubits`` array is stored.
        """
        branch_indices = rng.choice(
            len(self.channel.circuits),
            size=(num_trajectories, number_of_steps),
            p=self.channel.probabilities,
        )
        fidelities = np.empty(num_trajectories, dtype=np.float64)
        signals = np.empty(num_trajectories, dtype=np.complex128)
        workspace = QuantumState(self.channel.num_qubits)

        for trajectory_index, trajectory in enumerate(branch_indices):
            workspace.load(initial_state)
            global_phase = 1.0 + 0.0j
            for branch_index in trajectory:
                index = int(branch_index)
                self.channel.circuits[index].update_quantum_state(workspace)
                global_phase *= self.global_phases[index]

            final_state = workspace.get_vector()
            ideal_overlap = global_phase * np.vdot(ideal_state, final_state)
            signals[trajectory_index] = (
                global_phase * np.vdot(initial_state, final_state)
            )
            fidelities[trajectory_index] = min(
                1.0,
                max(0.0, float(abs(ideal_overlap) ** 2)),
            )

        return fidelities, signals


@dataclass(frozen=True)
class _PreparedInputState:
    """One Haar state and its ideal evolution returned by an input worker."""

    state_index: int
    initial_state: NDArray[np.complex128]
    ideal_state: NDArray[np.complex128]
    ideal_signal: complex


@dataclass(frozen=True)
class _TrajectoryChunkSummary:
    """Sufficient statistics returned by one trajectory-chunk worker."""

    decomposition_index: int
    state_index: int
    count: int
    fidelity_sum: float
    fidelity_squared_sum: float
    signal_sum: complex
    signal_squared_norm_sum: float


@dataclass(frozen=True)
class QDriftTrajectoryMetricEstimate:
    """Nested Haar-state/qDRIFT-trajectory Monte Carlo result."""

    target_hamiltonian: LCP
    qdrift_decomposition: LCH
    total_time: float
    number_of_steps: int
    step_time: float
    num_initial_states: int
    num_trajectories: int
    ideal_signals: NDArray[np.complex128]
    approximate_signals: NDArray[np.complex128]
    infidelity: ScalarSampleStatistics
    absolute_qpe_signal_error: ScalarSampleStatistics
    infidelity_trajectory_standard_errors: NDArray[np.float64]
    qpe_signal_trajectory_standard_errors: NDArray[np.float64]

    @property
    def mean_infidelity(self) -> float:
        return self.infidelity.mean

    @property
    def mean_absolute_qpe_signal_error(self) -> float:
        return self.absolute_qpe_signal_error.mean

@dataclass(frozen=True)
class TauMEstimate:
    """Exact normalized-trace second moment computed in coefficient time."""

    lambda_sum: float
    tau_h_squared: float
    tau_sample_second_moment: float
    tau_m: float


def calculate_tau_m(decomposition: LCH) -> TauMEstimate:
    r"""Return ``tau(M_p)`` in O(L) without Pauli products or matrices."""
    if not isinstance(decomposition, LCH):
        raise TypeError("decomposition must be an LCH")
    lambda_sum = decomposition.coefficient_one_norm()
    tau_h_squared = math.fsum(
        coefficient * coefficient
        for _, coefficient in decomposition.lcp.term_items()
    )
    if isinstance(decomposition, SinglePauliLCH):
        tau_sample_second_moment = lambda_sum * lambda_sum
    else:
        weighted_inner_squared_sum = math.fsum(
            abs(outer_coefficient)
            * math.fsum(value * value for value in operator.terms.values())
            for outer_coefficient, operator in decomposition.terms
        )
        tau_sample_second_moment = lambda_sum * weighted_inner_squared_sum
    return TauMEstimate(
        lambda_sum=lambda_sum,
        tau_h_squared=tau_h_squared,
        tau_sample_second_moment=tau_sample_second_moment,
        tau_m=max(0.0, tau_sample_second_moment - tau_h_squared),
    )


def prepare_qdrift_trajectory_sampler(
    decomposition: LCH,
    step_time: float,
) -> PreparedQDriftTrajectorySampler:
    r"""Prepare branch circuits for ``exp(-i step_time B_j)``.

    ``prepare_qdrift_branches`` omits identity Pauli strings because global
    phases cancel from density operators.  QPE signals retain those phases,
    so they are explicitly restored here.
    """
    channel = prepare_qdrift_branches(decomposition, step_time)
    lambda_sum = decomposition.coefficient_one_norm()
    identity = "I" * decomposition.num_qubits
    phases = np.asarray(
        [
            np.exp(
                -1j
                * lambda_sum
                * step_time
                * float(np.sign(outer_coefficient))
                * operator.terms.get(identity, 0.0)
            )
            for outer_coefficient, operator in decomposition.terms
        ],
        dtype=np.complex128,
    )
    phases.setflags(write=False)
    return PreparedQDriftTrajectorySampler(
        channel=channel,
        global_phases=phases,
    )


def estimate_qdrift_haar_trajectory_metrics(
    target_hamiltonian: LCP,
    total_time: float,
    number_of_steps: int,
    qdrift_decompositions: Sequence[LCH],
    *,
    num_initial_states: int = 20,
    num_trajectories: int = 200,
    seed: int | None = None,
    num_workers: int = 1,
    trajectory_chunks_per_state: int | None = None,
    initial_state_indices: Sequence[int] | None = None,
) -> list[QDriftTrajectoryMetricEstimate]:
    r"""Estimate Haar-average infidelity and QPE signal error.

    The same Haar input states and ideal evolved states are reused for every
    decomposition.  qDRIFT trajectories are sampled independently for each
    decomposition.  With ``num_workers > 1``, Haar-state ideal evolutions and
    qDRIFT trajectory chunks are evaluated in separate process-pool phases.
    Qulacs OpenMP is fixed to one thread per worker.
    """
    if not isinstance(target_hamiltonian, LCP):
        raise TypeError("target_hamiltonian must be an LCP")
    validated_time, validated_steps = _validated_time_and_steps(
        total_time,
        number_of_steps,
    )
    _validate_positive_integer(num_initial_states, "num_initial_states")
    _validate_positive_integer(num_trajectories, "num_trajectories")
    _validate_positive_integer(num_workers, "num_workers")
    if num_trajectories < 2:
        raise ValueError("num_trajectories must be at least 2")
    if trajectory_chunks_per_state is not None:
        _validate_positive_integer(
            trajectory_chunks_per_state,
            "trajectory_chunks_per_state",
        )
    state_indices = _validated_state_indices(
        initial_state_indices,
        num_initial_states,
    )
    local_num_initial_states = len(state_indices)
    resolved_seed = _resolved_root_seed(seed)

    decompositions = tuple(qdrift_decompositions)
    if not decompositions:
        return []
    for decomposition in decompositions:
        if not isinstance(decomposition, LCH):
            raise TypeError("every qdrift_decomposition must be an LCH")
        if not _same_lcp(decomposition.lcp, target_hamiltonian):
            raise ValueError(
                "every qDRIFT decomposition must flatten to target_hamiltonian"
            )

    step_time = validated_time / validated_steps
    prepared = tuple(
        None
        if decomposition.coefficient_one_norm() == 0.0
        else prepare_qdrift_trajectory_sampler(decomposition, step_time)
        for decomposition in decompositions
    )
    if num_workers > 1:
        chunks_per_state = min(
            num_trajectories,
            num_workers
            if trajectory_chunks_per_state is None
            else trajectory_chunks_per_state,
        )
        return _estimate_parallel_trajectory_metrics(
            target_hamiltonian=target_hamiltonian,
            decompositions=decompositions,
            prepared=prepared,
            total_time=validated_time,
            number_of_steps=validated_steps,
            state_indices=state_indices,
            num_trajectories=num_trajectories,
            num_workers=num_workers,
            chunks_per_state=chunks_per_state,
            root_seed=resolved_seed,
        )

    num_decompositions = len(decompositions)
    ideal_signals = np.empty(local_num_initial_states, dtype=np.complex128)
    approximate_signals = np.empty(
        (num_decompositions, local_num_initial_states),
        dtype=np.complex128,
    )
    infidelities = np.empty(
        (num_decompositions, local_num_initial_states),
        dtype=np.float64,
    )
    absolute_signal_errors = np.empty_like(infidelities)
    infidelity_trajectory_errors = np.empty_like(infidelities)
    signal_trajectory_errors = np.empty_like(infidelities)

    for local_state_index, global_state_index in enumerate(state_indices):
        state_rng = np.random.default_rng(
            _indexed_seed(resolved_seed, 0, global_state_index)
        )
        initial_state = haar_random_state(target_hamiltonian.num_qubits, state_rng)
        ideal_state = ideal_time_evolved_state(
            target_hamiltonian,
            validated_time,
            initial_state,
        )
        ideal_signal = np.vdot(initial_state, ideal_state)
        ideal_signals[local_state_index] = ideal_signal

        for decomposition_index, sampler in enumerate(prepared):
            if sampler is None:
                trajectory_fidelities = np.ones(
                    num_trajectories,
                    dtype=np.float64,
                )
                trajectory_signals = np.full(
                    num_trajectories,
                    np.vdot(initial_state, initial_state),
                    dtype=np.complex128,
                )
            else:
                trajectory_fidelities, trajectory_signals = (
                    sampler.sample_overlaps(
                        initial_state,
                        ideal_state,
                        validated_steps,
                        num_trajectories,
                        np.random.default_rng(
                            _indexed_seed(
                                resolved_seed,
                                1,
                                decomposition_index,
                                global_state_index,
                                0,
                            )
                        ),
                    )
                )

            mean_fidelity = float(np.mean(trajectory_fidelities))
            infidelities[decomposition_index, local_state_index] = min(
                1.0,
                max(0.0, 1.0 - mean_fidelity),
            )
            approximate_signal = complex(np.mean(trajectory_signals))
            approximate_signals[decomposition_index, local_state_index] = (
                approximate_signal
            )
            absolute_signal_errors[decomposition_index, local_state_index] = abs(
                ideal_signal - approximate_signal
            )
            infidelity_trajectory_errors[
                decomposition_index,
                local_state_index,
            ] = _real_mean_standard_error(trajectory_fidelities)
            signal_trajectory_errors[
                decomposition_index,
                local_state_index,
            ] = _complex_mean_standard_error(trajectory_signals)

    return _build_metric_estimates(
        target_hamiltonian=target_hamiltonian,
        decompositions=decompositions,
        total_time=validated_time,
        number_of_steps=validated_steps,
        num_initial_states=local_num_initial_states,
        num_trajectories=num_trajectories,
        ideal_signals=ideal_signals,
        approximate_signals=approximate_signals,
        infidelities=infidelities,
        absolute_signal_errors=absolute_signal_errors,
        infidelity_trajectory_errors=infidelity_trajectory_errors,
        signal_trajectory_errors=signal_trajectory_errors,
    )


def _estimate_parallel_trajectory_metrics(
    *,
    target_hamiltonian: LCP,
    decompositions: tuple[LCH, ...],
    prepared: tuple[PreparedQDriftTrajectorySampler | None, ...],
    total_time: float,
    number_of_steps: int,
    state_indices: tuple[int, ...],
    num_trajectories: int,
    num_workers: int,
    chunks_per_state: int,
    root_seed: int,
) -> list[QDriftTrajectoryMetricEstimate]:
    """Run ideal-state and trajectory phases in forked process pools."""
    if "fork" not in get_all_start_methods():
        raise RuntimeError(
            "parallel trajectory sampling currently requires the fork "
            "multiprocessing start method"
        )
    context = get_context("fork")
    num_initial_states = len(state_indices)
    input_tasks = [
        (
            local_state_index,
            _indexed_seed(root_seed, 0, global_state_index),
        )
        for local_state_index, global_state_index in enumerate(state_indices)
    ]

    with ProcessPoolExecutor(
        max_workers=min(num_workers, num_initial_states),
        mp_context=context,
        initializer=_initialize_input_worker,
        initargs=(target_hamiltonian, total_time),
    ) as executor:
        prepared_inputs = list(
            executor.map(_prepare_input_state_worker, input_tasks, chunksize=1)
        )
    prepared_inputs.sort(key=lambda item: item.state_index)
    initial_states = tuple(item.initial_state for item in prepared_inputs)
    ideal_states = tuple(item.ideal_state for item in prepared_inputs)
    ideal_signals = np.asarray(
        [item.ideal_signal for item in prepared_inputs],
        dtype=np.complex128,
    )
    for state in (*initial_states, *ideal_states):
        state.setflags(write=False)

    global _TRAJECTORY_INITIAL_STATES
    global _TRAJECTORY_IDEAL_STATES
    global _TRAJECTORY_SAMPLERS
    _TRAJECTORY_INITIAL_STATES = initial_states
    _TRAJECTORY_IDEAL_STATES = ideal_states
    _TRAJECTORY_SAMPLERS = prepared

    chunk_sizes = _balanced_chunk_sizes(num_trajectories, chunks_per_state)
    trajectory_tasks = [
        (
            decomposition_index,
            local_state_index,
            chunk_size,
            _indexed_seed(
                root_seed,
                1,
                decomposition_index,
                global_state_index,
                chunk_index,
            ),
            number_of_steps,
        )
        for decomposition_index in range(len(decompositions))
        for local_state_index, global_state_index in enumerate(state_indices)
        for chunk_index, chunk_size in enumerate(chunk_sizes)
    ]
    try:
        with ProcessPoolExecutor(
            max_workers=min(num_workers, len(trajectory_tasks)),
            mp_context=context,
            initializer=_initialize_single_thread_worker,
        ) as executor:
            summaries = list(
                executor.map(
                    _sample_trajectory_chunk_worker,
                    trajectory_tasks,
                    chunksize=1,
                )
            )
    finally:
        _TRAJECTORY_INITIAL_STATES = ()
        _TRAJECTORY_IDEAL_STATES = ()
        _TRAJECTORY_SAMPLERS = ()

    shape = (len(decompositions), num_initial_states)
    counts = np.zeros(shape, dtype=np.int64)
    fidelity_sums = np.zeros(shape, dtype=np.float64)
    fidelity_squared_sums = np.zeros(shape, dtype=np.float64)
    signal_sums = np.zeros(shape, dtype=np.complex128)
    signal_squared_norm_sums = np.zeros(shape, dtype=np.float64)
    for summary in summaries:
        index = (summary.decomposition_index, summary.state_index)
        counts[index] += summary.count
        fidelity_sums[index] += summary.fidelity_sum
        fidelity_squared_sums[index] += summary.fidelity_squared_sum
        signal_sums[index] += summary.signal_sum
        signal_squared_norm_sums[index] += summary.signal_squared_norm_sum
    if not np.all(counts == num_trajectories):
        raise RuntimeError("parallel trajectory chunks produced wrong counts")

    approximate_signals = signal_sums / num_trajectories
    infidelities = np.clip(
        1.0 - fidelity_sums / num_trajectories,
        0.0,
        1.0,
    )
    absolute_signal_errors = np.abs(
        ideal_signals[np.newaxis, :] - approximate_signals
    )

    fidelity_centered_sums = np.maximum(
        0.0,
        fidelity_squared_sums
        - fidelity_sums * fidelity_sums / num_trajectories,
    )
    infidelity_trajectory_errors = np.sqrt(
        fidelity_centered_sums
        / (num_trajectories * (num_trajectories - 1))
    )
    signal_centered_sums = np.maximum(
        0.0,
        signal_squared_norm_sums
        - np.abs(signal_sums) ** 2 / num_trajectories,
    )
    signal_trajectory_errors = np.sqrt(
        signal_centered_sums
        / (num_trajectories * (num_trajectories - 1))
    )

    return _build_metric_estimates(
        target_hamiltonian=target_hamiltonian,
        decompositions=decompositions,
        total_time=total_time,
        number_of_steps=number_of_steps,
        num_initial_states=num_initial_states,
        num_trajectories=num_trajectories,
        ideal_signals=ideal_signals,
        approximate_signals=approximate_signals,
        infidelities=infidelities,
        absolute_signal_errors=absolute_signal_errors,
        infidelity_trajectory_errors=infidelity_trajectory_errors,
        signal_trajectory_errors=signal_trajectory_errors,
    )


def _initialize_single_thread_worker() -> None:
    """Keep Qulacs and numerical-library inner threading disabled."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"


def _initialize_input_worker(target_hamiltonian: LCP, total_time: float) -> None:
    _initialize_single_thread_worker()
    global _INPUT_WORKER_TARGET
    global _INPUT_WORKER_TIME
    _INPUT_WORKER_TARGET = target_hamiltonian
    _INPUT_WORKER_TIME = total_time


def _prepare_input_state_worker(task: tuple[int, int]) -> _PreparedInputState:
    state_index, seed = task
    if _INPUT_WORKER_TARGET is None:
        raise RuntimeError("input-state worker was not initialized")
    rng = np.random.default_rng(seed)
    initial_state = haar_random_state(_INPUT_WORKER_TARGET.num_qubits, rng)
    ideal_state = ideal_time_evolved_state(
        _INPUT_WORKER_TARGET,
        _INPUT_WORKER_TIME,
        initial_state,
    )
    return _PreparedInputState(
        state_index=state_index,
        initial_state=initial_state,
        ideal_state=ideal_state,
        ideal_signal=complex(np.vdot(initial_state, ideal_state)),
    )


def _sample_trajectory_chunk_worker(
    task: tuple[int, int, int, int, int],
) -> _TrajectoryChunkSummary:
    decomposition_index, state_index, count, seed, number_of_steps = task
    initial_state = _TRAJECTORY_INITIAL_STATES[state_index]
    ideal_state = _TRAJECTORY_IDEAL_STATES[state_index]
    sampler = _TRAJECTORY_SAMPLERS[decomposition_index]
    if sampler is None:
        fidelities = np.ones(count, dtype=np.float64)
        signals = np.ones(count, dtype=np.complex128)
    else:
        fidelities, signals = sampler.sample_overlaps(
            initial_state,
            ideal_state,
            number_of_steps,
            count,
            np.random.default_rng(seed),
        )
    return _TrajectoryChunkSummary(
        decomposition_index=decomposition_index,
        state_index=state_index,
        count=count,
        fidelity_sum=float(np.sum(fidelities, dtype=np.float64)),
        fidelity_squared_sum=float(
            np.sum(fidelities * fidelities, dtype=np.float64)
        ),
        signal_sum=complex(np.sum(signals, dtype=np.complex128)),
        signal_squared_norm_sum=float(
            np.sum(np.abs(signals) ** 2, dtype=np.float64)
        ),
    )


def _balanced_chunk_sizes(total: int, chunks: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, chunks)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(chunks)
    )


def _seed_value(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def _resolved_root_seed(seed: int | None) -> int:
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        return seed
    return _seed_value(np.random.SeedSequence())


def _indexed_seed(root_seed: int, *coordinates: int) -> int:
    """Derive a seed from stable global coordinates, independent of sharding."""
    sequence = np.random.SeedSequence(root_seed, spawn_key=coordinates)
    return _seed_value(sequence)


def _validated_state_indices(
    initial_state_indices: Sequence[int] | None,
    num_initial_states: int,
) -> tuple[int, ...]:
    if initial_state_indices is None:
        return tuple(range(num_initial_states))
    indices = tuple(initial_state_indices)
    if not indices:
        raise ValueError("initial_state_indices must not be empty")
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= num_initial_states
        for index in indices
    ):
        raise ValueError(
            "initial_state_indices must contain integers in "
            "range(num_initial_states)"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("initial_state_indices must not contain duplicates")
    return indices


def _build_metric_estimates(
    *,
    target_hamiltonian: LCP,
    decompositions: tuple[LCH, ...],
    total_time: float,
    number_of_steps: int,
    num_initial_states: int,
    num_trajectories: int,
    ideal_signals: NDArray[np.complex128],
    approximate_signals: NDArray[np.complex128],
    infidelities: NDArray[np.float64],
    absolute_signal_errors: NDArray[np.float64],
    infidelity_trajectory_errors: NDArray[np.float64],
    signal_trajectory_errors: NDArray[np.float64],
) -> list[QDriftTrajectoryMetricEstimate]:
    ideal_signals.setflags(write=False)
    approximate_signals.setflags(write=False)
    infidelity_trajectory_errors.setflags(write=False)
    signal_trajectory_errors.setflags(write=False)
    return [
        QDriftTrajectoryMetricEstimate(
            target_hamiltonian=target_hamiltonian,
            qdrift_decomposition=decomposition,
            total_time=total_time,
            number_of_steps=number_of_steps,
            step_time=total_time / number_of_steps,
            num_initial_states=num_initial_states,
            num_trajectories=num_trajectories,
            ideal_signals=ideal_signals,
            approximate_signals=approximate_signals[decomposition_index],
            infidelity=ScalarSampleStatistics.from_values(
                infidelities[decomposition_index]
            ),
            absolute_qpe_signal_error=ScalarSampleStatistics.from_values(
                absolute_signal_errors[decomposition_index]
            ),
            infidelity_trajectory_standard_errors=(
                infidelity_trajectory_errors[decomposition_index]
            ),
            qpe_signal_trajectory_standard_errors=(
                signal_trajectory_errors[decomposition_index]
            ),
        )
        for decomposition_index, decomposition in enumerate(decompositions)
    ]


def _validated_time_and_steps(
    total_time: float,
    number_of_steps: int,
) -> tuple[float, int]:
    if not isinstance(total_time, Real) or isinstance(total_time, bool):
        raise TypeError("total_time must be a real number")
    value = float(total_time)
    if not math.isfinite(value):
        raise ValueError("total_time must be finite")
    _validate_positive_integer(number_of_steps, "number_of_steps")
    return value, number_of_steps


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _same_lcp(left: LCP, right: LCP) -> bool:
    if left.num_qubits != right.num_qubits:
        return False
    left_terms = left.terms
    right_terms = right.terms
    return all(
        math.isclose(
            left_terms.get(pauli, 0.0),
            right_terms.get(pauli, 0.0),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
        for pauli in left_terms.keys() | right_terms.keys()
    )


def _real_mean_standard_error(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _complex_mean_standard_error(values: NDArray[np.complex128]) -> float:
    if values.size < 2:
        return 0.0
    centered_squared_norm = float(np.sum(np.abs(values - np.mean(values)) ** 2))
    return math.sqrt(centered_squared_norm / (values.size * (values.size - 1)))


__all__ = [
    "PreparedQDriftTrajectorySampler",
    "QDriftTrajectoryMetricEstimate",
    "ScalarSampleStatistics",
    "TauMEstimate",
    "calculate_tau_m",
    "estimate_qdrift_haar_trajectory_metrics",
    "prepare_qdrift_trajectory_sampler",
]
