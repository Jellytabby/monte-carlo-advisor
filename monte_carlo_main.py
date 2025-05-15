import argparse
import inline_mc_advisor
import utils
import logging

logger = logging.getLogger(__name__)
datefmt="%Y-%m-%d %H:%M:%S"
fmt = "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(funcName)s(): %(message)s"


def list_of_args(args:str) -> list[str]:
    return args.split(',')

def parse_args_and_run():
    parser = argparse.ArgumentParser(
        prog="Monte Carlo Autotuner", 
        description="This python programs tunes compiler passes based on a Monte Carlo tree"
    )
    # parser.add_argument('input_file', type=str, help="Path to input file") 
    parser.add_argument("--debug", default=False, action="store_true", help="Set the logging level to debug")
    parser.add_argument('-r', '--number-of-runs', type=int, default=50, help="Number of iterations to run the Monte Carlo Simulation")
    # parser.add_argument('-i', '--initial-samples', type=int, default=5, help="Number of initial samples the adaptive benchmark generates.")


    args = parser.parse_args()
    main(args)

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)

    m = get_input_module()
    advisor = inline_mc_advisor.InlineMonteCarloAdvisor()
    advisor.run_monte_carlo(args.number_of_runs, m, get_score)



def get_input_module():
    cmd = ['make', 'mod-pre-mc.bc']
    utils.get_cmd_output(cmd)
    with open("mod-pre-mc.bc", 'rb') as f:
        mod  = f.read()
    return mod

def runtime_generator():
    while True:
        cmd = ['make', 'run']
        outs = utils.get_cmd_output(cmd)
        yield utils.readout_mc_inline_timer(outs.decode())


def get_score(mod: bytes):
    with open("mod-post-mc.bc", 'wb') as f:
        f.write(mod)
    return utils.adaptive_benchmark(runtime_generator()).mean




if __name__ == "__main__":
    parse_args_and_run()
