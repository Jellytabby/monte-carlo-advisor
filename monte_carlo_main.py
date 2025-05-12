import argparse
import os
import stat
import tempfile
import interactive_host
import log_reader
import sys
import inline_mc_advisor
import compile

def list_of_args(args:str) -> list[str]:
    return args.split(',')

parser = argparse.ArgumentParser(
    prog="Monte Carlo Autotuner", 
    description="This python programs tunes compiler passes based on a Monte Carlo tree"
)
parser.add_argument('input_file', type=str, help="Path to input file") 
# parser.add_argument('--opt-flags', type=list_of_args, help="Comma separated list of flags to pass to the llvm IR optimizer")

args = parser.parse_args()

if __name__ == "__main__":
    filename = os.path.basename(args.input_file)[:-4]
    mc_advisor = inline_mc_advisor.InlineMonteCarloAdvisor()
    
    clang_out = compile.get_ir_from_clang(args.input_file)

    with tempfile.NamedTemporaryFile(suffix=".ll") as f1, \
        tempfile.NamedTemporaryFile(suffix=".bc") as f2:
        f1.write(clang_out)
        f1.flush()

        interactive_host.run_interactive(
            f"{filename}.channel-basename", 
            mc_advisor.advice,
            ['opt',
            '-passes=scc-oz-module-inliner',
            '-interactive-model-runner-echo-reply',
            '-enable-ml-inliner=release',
            f"-inliner-interactive-channel-base={filename}.channel-basename",
            '-o', f2.name,
            f1.name
            ])

        compile.compile_llvm_ir(filename, f2.read())
    execution_time = compile.measure_execution_time(f"{filename}.out")
    print(f"Binary took {execution_time}s to run.")



    # compile.compile_llvm_ir(f"{filename}.out", sys.stdin.read())

    # os.chmod(f"bin_{filename}.out",stat.S_IRWXU)
    # execution_time = compile.measure_execution_time(f"bin_{filename}.out")
    # print(f"Binary took {execution_time}s to run.")

    
