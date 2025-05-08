import argparse
import subprocess
from subprocess import CompletedProcess

parser = argparse.ArgumentParser(
    description='Take a C++ file and return LLVM IR')
parser.add_argument('cpp_file', type=str, help='Path to the C++ file')
parser.add_argument('--flags', nargs=argparse.REMAINDER,
                    default=[], help='Flags to pass to the clang compiler')

args = parser.parse_args()


def get_ir_from_clang(file_path: str) -> CompletedProcess[str]:
    clang_cmd = ['clang++',  '-cc1', '-O1', '-disable-llvm-passes',
                 '-emit-llvm'] + args.flags + [file_path, '-o', '-']
    tee_cmd = ['tee', file_path+'.ll']
    try:
        clang_result = subprocess.run(
            clang_cmd, capture_output=True, text=True, check=True)
        # tee_result = subprocess.run(tee_cmd, input=clang_result.stdout, capture_output=True, text=True, check=True)
        print(f"Compilation successful.\n"
              f"Warnings:\n\n{clang_result.stderr}\n"
              f"Output:\n\n{clang_result.stdout}\n")
        return clang_result
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error code {e.returncode}:")
        print(e.stderr)
        exit(e.returncode)


def optimize_llvm_ir(clang_output: str):
    opt_cmd = ['opt', '-O3']
    try:
        opt_result = subprocess.run(
            opt_cmd, capture_output=True, text=True, check=True, input=clang_output)
        print(f"Optimization successful.\n"
              f"Warnings:\n\n{opt_result.stderr}\n"
              f"Output:\n\n{opt_result.stdout}\n")
    except subprocess.CalledProcessError as e:
        print(f"Optimization failed with error code {e.returncode}:")


def compile_llvm_ir():
    cmd = ['llc', ]


if __name__ == "__main__":
    clang_result = get_ir_from_clang(args.cpp_file)
    optimize_llvm_ir(clang_result.stdout)
