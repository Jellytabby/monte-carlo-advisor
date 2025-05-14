import argparse
import os
import stat
import tempfile
import interactive_host
import log_reader
import sys
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
    optimized_m = inline_mc_advisor.optimize_module(m, get_score)
    print(get_score(optimized_m))




# if __name__ == "__main__":
#     baseline = get_baseline(args.input_file)
#
#     filename = utils.basename(args.input_file)
#     print(f"Filename: {filename}")
#     mc_advisor = inline_mc_advisor.InlineMonteCarloAdvisor()
#
#     clang_out = utils.get_ir_from_clang(args.input_file)
#
#     with tempfile.NamedTemporaryFile(suffix=".ll") as f1, \
#         tempfile.NamedTemporaryFile(suffix=".bc") as f2:
#         f1.write(clang_out)
#         f1.flush()
#
#         interactive_host.run_interactive(
#             f"{filename}.channel-basename", 
#             mc_advisor.advice,
#             ['opt',
#             '-passes=scc-oz-module-inliner',
#             '-interactive-model-runner-echo-reply',
#             '-enable-ml-inliner=release',
#             f"-inliner-interactive-channel-base={filename}.channel-basename",
#             '-S',
#             '-o', f2.name,
#             f1.name
#             ])
#
#         # with open("inline-decision-one", 'w') as f:
#         #    f.write(f2.read().decode()) 
#         # f2.seek(0)
#
#         utils.compile_llvm_ir(f"{filename}_mc", f2.read())
#     execution_time = utils.measure_execution_time(f"{filename}_mc.out")
#     print(f"Baseline:    {baseline}")
#     print(f"Monte Carlo: {execution_time}s")
#
#     print("Baseline wins") if baseline <= execution_time else print("Monte Carlo wins")
