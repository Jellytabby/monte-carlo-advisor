import io
import logging
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import psutil
from scipy import stats

from datastructures import *

logger = logging.getLogger(__name__)

LOOP_UNROLL_ERROR_CODE = -999
TIMEOUT_ERROR_CODE = -111


class MonteCarloError(Exception):
    pass


def basename(file: str) -> str:
    return Path(file).stem


def get_core_maps() -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    output = subprocess.check_output("lscpu -p=CPU,Core", shell=True).decode()
    lines = [line for line in output.splitlines() if not line.startswith("#")]
    physical_to_logical = {}
    logical_to_physical = {}
    for line in lines:
        cpu, core = map(int, line.split(","))
        if core not in physical_to_logical:
            physical_to_logical[core] = [cpu]
        else:
            physical_to_logical[core].append(cpu)
        logical_to_physical[cpu] = core
    return physical_to_logical, logical_to_physical


def get_next_free_physical_core(start=2):
    num_cores = psutil.cpu_count(logical=False)  # physical cores only
    _, l_to_p_map = get_core_maps()
    assert num_cores
    used = set()
    for p in psutil.process_iter(attrs=["username", "cpu_affinity", "name"]):
        if "sophia.herrmann" in p.info["username"] and "make run" in p.info["name"]:
            print(p.info)
            try:
                aff = p.info["cpu_affinity"]
                [used.add(l_to_p_map[c]) for c in aff]
                logger.info(f"Logical cpus already in use: {aff}")
            except Exception:
                continue
    for core in range(start, num_cores):
        if core not in used:
            logger.info(f"First free physical core: {core}")
            return core
    raise RuntimeError("No free core found")


def get_cmd_output(
    cmd, stdin=None, timeout=None, pre_exec_function=None, env_vars=None
):
    logger.debug(f"Running cmd: {' '.join(cmd)}")

    # sns = False if cmd[0] != "clang++" else True
    # Only clang++ needs it but just in case let's use a process group for everything

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=(subprocess.PIPE if stdin is not None else None),
        preexec_fn=pre_exec_function,
        env=env_vars,
    ) as proc:
        try:
            outs, errs = proc.communicate(input=stdin, timeout=timeout)
            status = proc.wait()
        # except subprocess.TimeoutExpired as e:
        #     if sns:
        #         kill_fn = kill_proc_sns
        #         terminate_fn = terminate_proc_sns
        #     else:
        #         kill_fn = kill_proc
        #         terminate_fn = terminate_proc
        #
        #     logger.debug("Process timed out! Terminating...")
        #     terminate_fn(proc)
        #     try:
        #         proc.communicate(timeout=1)
        #     except subprocess.TimeoutExpired as e:
        #         logger.debug("Termination timed out! Killing...")
        #         kill_fn(proc)
        #         proc.communicate()
        #
        #         logger.debug("Killed.")
        #         raise InputGenTimeout(f"Timed out: {cmd}")
        #
        #     logger.debug("Terminated.")
        #     raise InputGenTimeout(f"Timed out: {cmd}")
        except subprocess.TimeoutExpired as e:
            logger.warning("Process timed out! Terminating...")
            proc.terminate()
            try:
                proc.communicate(timeout=1)
            except subprocess.TimeoutExpired as e:
                logger.warning("Termination timed out! Killing...")
                proc.kill()
                proc.communicate()
                logger.debug("Killed.")
                raise e

            logger.debug("Terminated.")
            raise e

        if status != 0:
            logger.error(f"Exit with status {status}")
            logger.error(f"Command run: {' '.join(cmd)}")
            logger.error("Output:")
            logger.error(errs.decode())

            logger.error("Failed.")
            exit(status)

        logger.debug("Finished.")
        logger.debug(f"Output: {outs.decode()}")
        return outs


def terminate(process: subprocess.Popen[bytes]):
    process.terminate()
    clean_up_process(process)


def clean_up_process(
    process: subprocess.Popen[bytes], error_buffer: io.BufferedRandom | None = None
):

    try:
        outs, _ = process.communicate()
    except ValueError:
        return 0
    status = process.wait()
    if error_buffer:
        error_buffer.seek(0)
        if error_buffer.peek(1):
            if status != 0:
                logger.error(error_buffer.read().decode())
            else:
                logger.debug(
                    f"\n{selective_mlgo_output(error_buffer.read().decode('utf-8'))}"
                )
    logger.debug(f"Outs size {len(outs)}")
    logger.debug(f"Status {status}")
    return status


def selective_mlgo_output(log: str):
    lines = log.splitlines(True)
    lines = [l for l in lines if not l.startswith("unrolling_decision")]
    lines = [l for l in lines if not "ShouldInstrument" in l]
    lines = [("\n" + l) if l.startswith("Loop Unroll") else l for l in lines]
    return "".join(lines)


def readout_mc_inline_timer(input: str) -> int:
    re_match = re.search("MC_TIMER ([0-9]+)", input)
    if re_match is None:
        raise Exception(
            "No measurement found. Are you calling __mc_profiling_begin() in your code?"
        )
    else:
        f = int(re_match.group(1))
        return f


def get_benchmarking_median_ci(samples, confidence=0.95) -> tuple[float, float]:
    """
    Compute a nonparametric (distribution-free) confidence interval for the median.
    Returns a tuple: (median, lower_bound, upper_bound).

    Parameters
    ----------
    data : array-like of numeric
        The sample of measurements.
    p : float, optional
        Desired confidence level (default 0.95).
    """
    if len(samples) == 0:
        return np.nan, np.inf
    if len(samples) == 1:
        return samples[0], np.inf

    samples = np.sort(np.asarray(samples))
    n = samples.size
    z = stats.norm.ppf((1 + confidence) / 2)

    # compute 1-based ranks
    lower_rank = math.floor((n - z * math.sqrt(n)) / 2)
    upper_rank = math.ceil(1 + (n + z * math.sqrt(n)) / 2)
    # clamp to [1, n]
    lower_rank = max(lower_rank, 1)
    upper_rank = min(upper_rank, n)

    median = float(np.median(samples))

    # convert to zero-based indices
    lower = samples[lower_rank - 1]
    upper = samples[upper_rank - 1]
    interval = upper - lower
    relative_ci_width = interval / median
    return median, relative_ci_width


def adaptive_benchmark(
    iterator,
    warmup_runs,
    initial_samples,
    max_samples,
    max_initial_samples=20,
    confidence=0.95,
    relative_ci_threshold=0.05,
    fail_on_non_convergence=False,
) -> AdaptiveBenchmarkingResult:
    """
    Adaptive benchmarking loop to estimate mean runtime with confidence.

    Parameters:
        iterator: An iterator yielding runtime samples.
        initial_samples: Number of initial samples to take.
        max_initial_samples: Max number of tries to get initial samples.
        max_samples: Max number of samples to avoid infinite loop.
        confidence: Desired confidence level (e.g., 0.95 for 95% CI).
        relative_ci_threshold: Target relative CI width (e.g., 0.05 means CI width < 5% of mean).

    Returns:
        AdaptiveBenchmarkingResult
    """
    assert max_initial_samples < max_samples

    logger.info("Starting adaptive benchmarking")

    samples = np.array([], dtype=float)
    n = 0

    if warmup_runs > 0:
        logger.debug("Starting warmup runs")
        for _ in range(warmup_runs):
            next(iterator)
    while len(samples) < initial_samples and n < max_initial_samples:
        new_sample = next(iterator)
        if new_sample is not None:
            new_sample = float(new_sample)
            samples = np.append(samples, new_sample)
            if n == 0 and new_sample == 0:
                logger.debug("Got zero")
                return get_zero_rt_abr()
            logger.debug(
                f"Obtained sample {new_sample}, len {len(samples)}, iteration {n}"
            )
        n += 1

    if len(samples) < initial_samples:
        logger.error("Too many replay failures")
        median, relative_ci_width = get_benchmarking_median_ci(samples, confidence)
        return AdaptiveBenchmarkingResult(samples, median, relative_ci_width, False)

    assert n < max_samples

    median = 0.0
    relative_ci_width = 0.0
    while n < max_samples:
        median, relative_ci_width = get_benchmarking_median_ci(samples, confidence)

        if relative_ci_width < relative_ci_threshold:
            logger.debug(f"Converged: median {median}, ci {relative_ci_width}")
            return AdaptiveBenchmarkingResult(samples, median, relative_ci_width, True)

        new_sample = None
        while new_sample is None and n < max_samples:
            new_sample = next(iterator)
            logger.debug(
                f"Obtained sample {new_sample}, len {len(samples)}, iteration {n}"
            )
            n += 1
        if new_sample is not None:

            samples = np.append(samples, float(new_sample))

    logger.error(f"Did not converge: median {median}, ci {relative_ci_width}")

    if fail_on_non_convergence:
        return get_invalid_abr()
    else:
        return AdaptiveBenchmarkingResult(samples, median, relative_ci_width, False)


def get_speedup_factor(base: np.ndarray, opt: np.ndarray):
    # This will get element wise speedup factors for all inputs where both succeeded
    base = base[: len(opt)]
    opt = opt[: len(base)]

    arr = base / opt
    arr = arr[~np.isnan(arr)]  # remove NaNs
    if arr.size == 0:
        return None
    geomean = np.exp(np.mean(np.log(arr)))
    return geomean


if __name__ == "__main__":
    pass
