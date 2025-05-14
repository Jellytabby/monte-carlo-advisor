import argparse
import inline_mc_advisor
import utils

def list_of_args(args:str) -> list[str]:
    return args.split(',')

parser = argparse.ArgumentParser(
    prog="Monte Carlo Autotuner", 
    description="This python programs tunes compiler passes based on a Monte Carlo tree"
)
# parser.add_argument('input_file', type=str, help="Path to input file") 

args = parser.parse_args()

# def get_baseline(path_to_file: str) -> float:
#     filename = utils.basename(path_to_file)
#     cmd = ['clang++','-O3' , path_to_file, '-o',f"{filename}_baseline.out" ]
#     utils.get_cmd_output(cmd)
#     return utils.measure_execution_time(f"{filename}_baseline.out", output=None)

def get_input_module():
    cmd = ['make', 'mod-pre-mc.bc']
    utils.get_cmd_output(cmd)
    with open("mod-pre-mc.bc", 'rb') as f:
        mod  = f.read()
    return mod



def get_score(mod: bytes):
    with open("mod-post-mc.bc", 'wb') as f:
        f.write(mod)
    cmd = ['make', 'run']
    outs = utils.get_cmd_output(cmd)
    return utils.readout_mc_inline_timer(outs.decode())


if __name__ == "__main__":
    m = get_input_module()
    advisor = inline_mc_advisor.InlineMonteCarloAdvisor()
    advisor.run_monte_carlo(10, m, get_score)
    # optimized_m = inline_mc_advisor.optimize_module(m, get_score)
