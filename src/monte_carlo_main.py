import argparse
import logging
import os
from datetime import datetime

import psutil
from matplotlib.pyplot import plot

import plot_main
import utils
from advisors.inline import inline_mc_advisor
from advisors.loop_unroll import loop_unroll_mc_advisor
from advisors.merged.merged_mc_advisor import MergedMonteCarloAdvisor

logger = logging.getLogger(__name__)
datefmt = "%Y-%m-%d %H:%M:%S"
fmt = "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(funcName)s(): %(message)s"


def list_of_args(args: str) -> list[str]:
    return args.split(",")


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        prog="Monte Carlo Autotuner",
        description="This python programs tunes compiler passes based on a Monte Carlo tree",
    )
    advisor_selection = parser.add_argument_group("Advisor Selection")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input. The script expects two files, <input>_main.[c|cpp] and <input>_kernel.[c|cpp] respectively.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Set the logging level to debug",
    )
    advisor_selection.add_argument(
        "-ia",
        "--inline-advisor",
        default=False,
        action="store_true",
        help="Enable the Inline Monte Carlo Advisor",
    )
    advisor_selection.add_argument(
        "-lua",
        "--loop-unroll-advisor",
        default=False,
        action="store_true",
        help="Enable the Loop Unroll Monte Carlo Advisor",
    )
    parser.add_argument(
        "-r",
        "--number-of-runs",
        type=int,
        default=50,
        help="Number of iterations to run the Monte Carlo Simulation",
    )
    parser.add_argument(
        "-w",
        "--warmup_runs",
        type=int,
        default=15,
        help="Number of warumup runs to discard before benchmarking.",
    )
    parser.add_argument(
        "-i",
        "--initial-samples",
        type=int,
        default=20,
        help="Number of initial samples the adaptive benchmark generates.",
    )
    parser.add_argument(
        "-m",
        "--max-samples",
        default=100,
        help="Maximum number of runtime samples to take until convergence.",
    )
    parser.add_argument(
        "-c",
        "--core",
        type=int,
        required=True,
        help="Core on which to execute the benchmark runs on.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        help="Timeout for llc and opt processes",
    )

    args = parser.parse_args()
    main(args)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)

    MANAGER_PHYSICAL_CORES = 8
    physical_to_logical, _ = utils.get_core_maps()

    logical_cores = sum(
        [physical_to_logical[i] for i in range(MANAGER_PHYSICAL_CORES)], []
    )
    logger.info(f"Python script running on logical cores: {logical_cores}")
    os.sched_setaffinity(0, set(logical_cores))

    # next_free_core = physical_to_logical[
    #     utils.get_next_free_physical_core(MANAGER_PHYSICAL_CORES)
    # ][0]

    if args.core < MANAGER_PHYSICAL_CORES:
        raise RuntimeError(f"Core {args.core} is reserved for the manager process")
    try:
        next_free_core = physical_to_logical[args.core][0]
    except:
        raise RuntimeError(f"Core {args.core} is out of range")

    logger.info(f"Benchmark core is {next_free_core}")
    input_file = args.input_file
    input_dir = os.path.dirname(input_file)
    input_name = os.path.basename(input_file)
    os.environ["INPUT"] = input_file

    match (args.inline_advisor, args.loop_unroll_advisor):
        case (True, True):
            advisor = MergedMonteCarloAdvisor(input_name)
        case (True, False):
            advisor = inline_mc_advisor.InlineMonteCarloAdvisor(input_name)
        case (False, True):
            advisor = loop_unroll_mc_advisor.LoopUnrollMonteCarloAdvisor(input_name)
        case _:
            raise Exception(
                "You need to specify at least one advisor. See '--help' for more information."
            )

    start = datetime.now().strftime("%Y%m%d_%H%M%S")
    make_clean()
    get_input_module()

    logger.info("Starting baseline benchmarking")
    baseline = get_baseline_runtime(
        args.warmup_runs, args.initial_samples, args.max_samples, next_free_core
    )
    logger.info("Completed baseline benchmarking")

    logger.info("Starting Monte Carlo Tree runs")
    advisor.run_monte_carlo(
        args.number_of_runs,
        input_dir + "/",
        args.timeout,
        lambda: get_score(
            baseline,
            args.warmup_runs,
            args.initial_samples,
            args.max_samples,
            args.timeout,
            next_free_core,
        ),
    )
    plot_main.log_results(advisor, args, start, input_name)
    plot_main.plot_speedup(advisor, input_name)
    del os.environ["INPUT"]  # NOTE: makes no difference apparently?


def make_clean():
    cmd = ["make", "clean"]
    utils.get_cmd_output(cmd)


def get_input_module():
    cmd = ["make", "mod-pre-mc.bc"]
    utils.get_cmd_output(cmd)


def get_baseline_runtime(
    warmup_runs: int, initial_samples: int, max_samples: int, core: int
):
    cmd = ["make", "run_baseline"]
    return utils.adaptive_benchmark(
        runtime_generator(cmd, core),
        warmup_runs=warmup_runs,
        initial_samples=initial_samples,
        max_samples=max_samples,
    )


def runtime_generator(cmd: list[str], core: int):
    logger.debug(cmd)
    while True:
        outs = utils.get_cmd_output(
            cmd, pre_exec_function=lambda: os.sched_setaffinity(0, {core})
        )
        yield utils.readout_mc_inline_timer(outs.decode())


def get_score(
    baseline: utils.AdaptiveBenchmarkingResult,
    warmup_runs: int,
    initial_samples: int,
    max_samples: int,
    timeout: float,
    core: int,
):
    cmd = ["make", "module_obj"]
    utils.get_cmd_output(cmd, timeout=timeout)
    cmd = ["make", "run"]
    runtimes = utils.adaptive_benchmark(
        runtime_generator(cmd, core),
        warmup_runs=warmup_runs,
        initial_samples=initial_samples,
        max_samples=max_samples,
    )
    return baseline.mean / runtimes.mean
    # return utils.get_speedup_factor(baseline, runtimes)


if __name__ == "__main__":
    parse_args_and_run()
