import io
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

from datastructures import *

logger = logging.getLogger(__name__)


def basename(file: str) -> str:
    return Path(file).stem


def get_cmd_output(cmd, stdin=None, timeout=None, pre_exec_function=None, env_vars=None):
    logger.debug(f"Running cmd: {' '.join(cmd)}")

    # sns = False if cmd[0] != "clang++" else True
    # Only clang++ needs it but just in case let's use a process group for everything

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=(subprocess.PIPE if stdin is not None else None),
        preexec_fn=pre_exec_function,
        env=env_vars
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
            print(f"Some error {e}")
            exit(-1)

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


def clean_up_process(
    process: subprocess.Popen[bytes], error_buffer: io.BufferedRandom | None = None
):
    if process.poll() != None:
        return 0
    outs, _ = process.communicate()
    status = process.wait()
    if error_buffer:
        error_buffer.seek(0)
        if status != 0:
            logger.error(error_buffer.read().decode())
        else:
            logger.info(
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


def readout_mc_inline_timer(input: str) -> int | None:
    re_match = re.search("MC_INLINE_TIMER ([0-9]+)", input)
    if re_match is None:
        return None
    else:
        f = int(re_match.group(1))
        return f


def get_benchmarking_mean_ci(samples, confidence):
    if len(samples) == 0:
        return np.nan, np.inf
    if len(samples) == 1:
        return samples[0], np.inf

    sample_mean = np.mean(samples)
    sample_std = np.std(samples, ddof=1)  # sample std (unbiased)
    if sample_mean == 0:
        assert sample_std == 0
        return 0.0, 0.0

    # t critical value
    n = len(samples)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin_error = t_crit * (sample_std / np.sqrt(n))
    relative_ci_width = (2 * margin_error) / sample_mean
    return sample_mean, relative_ci_width


def remove_outliers_tscore(samples, alpha: float = 0.05):
    """
    Remove any sample whose studentized residual |t_i| exceeds
    the two‐sided t_crit (df=n-1) at level alpha, mutating samples.
    """
    n = len(samples)
    if n < 2:
        return

    mean = samples.mean()
    std = samples.std(ddof=1)
    # studentized residuals
    t_vals = (samples - mean) / std
    # two-sided critical threshold
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    keep_mask = np.abs(t_vals) <= t_crit
    logger.debug(
        f"Removed {len([x for x in keep_mask if x == False])} outliers from {len(samples)} samples."
    )
    return samples[keep_mask]


def adaptive_benchmark(
    iterator,
    warmup_runs,
    initial_samples=5,
    max_initial_samples=20,
    max_samples=50,
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

    logger.debug("Starting adaptive benchmarking")

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
        sample_mean, relative_ci_width = get_benchmarking_mean_ci(samples, confidence)
        return AdaptiveBenchmarkingResult(
            samples, sample_mean, relative_ci_width, False
        )

    assert n < max_samples

    samples = remove_outliers_tscore(samples)
    sample_mean = 0.0
    relative_ci_width = 0.0
    while n < max_samples:
        sample_mean, relative_ci_width = get_benchmarking_mean_ci(samples, confidence)

        if relative_ci_width < relative_ci_threshold:
            logger.debug(f"Converged: mean {sample_mean}, ci {relative_ci_width}")
            return AdaptiveBenchmarkingResult(
                samples, sample_mean, relative_ci_width, True
            )

        new_sample = None
        while new_sample is None and n < max_samples:
            new_sample = next(iterator)
            logger.debug(
                f"Obtained sample {new_sample}, len {len(samples)}, iteration {n}"
            )
            n += 1
        if new_sample is not None:

            samples = np.append(samples, float(new_sample))

    logger.error(f"Did not converge: mean {sample_mean}, ci {relative_ci_width}")

    if fail_on_non_convergence:
        return get_invalid_abr()
    else:
        return AdaptiveBenchmarkingResult(
            samples, sample_mean, relative_ci_width, False
        )


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
