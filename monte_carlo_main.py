import argparse
import logging
import os
import sys

import utils
from advisors.inline import inline_mc_advisor
from advisors.loop_unroll import loop_unroll_mc_advisor
from advisors.merged.merged_mc_advisor import MergedMonteCarloAdvisor
from plots import plot_main

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

    args = parser.parse_args()
    main(args)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)

    input_file = args.input_file
    input_dir = os.path.dirname(input_file)
    input_name = os.path.basename(input_file)
    os.environ["INPUT"] = input_file
    make_clean()
    get_input_module()
    baseline = get_baseline_runtime(
        args.warmup_runs, args.initial_samples, args.max_samples, args.core
    )

    match (args.inline_advisor, args.loop_unroll_advisor):
        case (True, True):
            advisor = MergedMonteCarloAdvisor(input_name)
        case (True, False):
            advisor = inline_mc_advisor.InlineMonteCarloAdvisor(input_name)
        case (False, True):
            advisor = loop_unroll_mc_advisor.LoopUnrollMonteCarloAdvisor(input_name)
        case _:
            exit(-1)
    advisor.run_monte_carlo(
        args.number_of_runs,
        input_dir + "/",
        lambda: get_score(
            baseline,
            args.warmup_runs,
            args.initial_samples,
            args.max_samples,
            args.core,
        ),
    )
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
    core: int,
):
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
