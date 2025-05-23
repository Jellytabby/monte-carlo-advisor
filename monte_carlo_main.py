import argparse

from numpy import ndarray
import inline_mc_advisor
import utils
import logging

logger = logging.getLogger(__name__)
datefmt = "%Y-%m-%d %H:%M:%S"
fmt = "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(funcName)s(): %(message)s"


def list_of_args(args: str) -> list[str]:
    return args.split(',')


def parse_args_and_run():
    parser = argparse.ArgumentParser(
        prog="Monte Carlo Autotuner",
        description="This python programs tunes compiler passes based on a Monte Carlo tree"
    )
    # parser.add_argument('input_file', type=str, help="Path to input file")
    parser.add_argument("--debug", default=False,
                        action="store_true", help="Set the logging level to debug")
    parser.add_argument('-r', '--number-of-runs', type=int, default=50,
                        help="Number of iterations to run the Monte Carlo Simulation")
    parser.add_argument('-w', '--warmup_runs', type=int, default=15, help="Number of warumup runs to discard before benchmarking.")
    parser.add_argument('-i', '--initial-samples', type=int, default=20, help="Number of initial samples the adaptive benchmark generates.")
    parser.add_argument('-m', '--max-samples', default=100, help="Maximum number of runtime samples to take until convergence.")

    args = parser.parse_args()
    main(args)


def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)

    m = get_input_module()
    baseline = get_baseline_runtime(args.warmup_runs, args.initial_samples, args.max_samples)
    advisor = inline_mc_advisor.InlineMonteCarloAdvisor()
    advisor.run_monte_carlo(args.number_of_runs, m,
                            lambda x: get_score(x, baseline, args.warmup_runs, args.initial_samples, args.max_samples))

def get_input_module():
    cmd = ['make', 'mod-pre-mc.bc']
    utils.get_cmd_output(cmd)
    with open("mod-pre-mc.bc", 'rb') as f:
        mod = f.read()
    return mod


def get_baseline_runtime(warmup_runs:int, initial_samples:int, max_samples:int):
    cmd = ['make', 'run_baseline']
    return utils.adaptive_benchmark(runtime_generator(cmd),warmup_runs=warmup_runs, initial_samples=initial_samples, max_samples=max_samples)


def runtime_generator(cmd: list[str]):
    logger.debug(cmd)
    while True:
        outs = utils.get_cmd_output(cmd)
        yield utils.readout_mc_inline_timer(outs.decode())


def get_score(mod: bytes, baseline: utils.AdaptiveBenchmarkingResult, warmup_runs:int, initial_samples:int, max_samples:int):
    with open("mod-post-mc.bc", 'wb') as f:
        f.write(mod)
    cmd = ['make', 'run']
    runtimes = utils.adaptive_benchmark(
        runtime_generator(cmd), warmup_runs=warmup_runs, initial_samples=initial_samples, max_samples=max_samples)
    return baseline.mean / runtimes.mean
    # return utils.get_speedup_factor(baseline, runtimes)


if __name__ == "__main__":
    parse_args_and_run()
