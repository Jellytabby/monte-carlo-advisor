import argparse
import time
import os
import subprocess
from subprocess import CompletedProcess

parser = argparse.ArgumentParser(
    description='Take a C++ file and return LLVM IR')
parser.add_argument('-S', action='store_true', help="Only generate llvm IR" )
parser.add_argument('cpp_file', type=str, help='Path to the C++ file')
parser.add_argument('--flags', nargs=argparse.REMAINDER,
                    default=[], help='Flags to pass to the clang compiler')

args = parser.parse_args()


def get_ir_from_clang(file_path: str) -> CompletedProcess[bytes]:
    clang_cmd = ['clang++',  '-cc1', '-O1', '-disable-llvm-passes',
                 '-emit-llvm'] + args.flags + [file_path, '-o', '-']
    # tee_cmd = ['tee', file_path+'.ll']
    try:
        clang_result = subprocess.run(
            clang_cmd, capture_output=True, check=True)
        # tee_result = subprocess.run(tee_cmd, input=clang_result.stdout, capture_output=True, text=True, check=True)
        print(f"Compilation successful.\n"
              f"Warnings:\n\n{clang_result.stderr.decode()}\n"
              f"Output:\n\n{clang_result.stdout.decode()}\n")
        return clang_result
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error code {e.returncode}:")
        print(e.stderr)
        exit(e.returncode)


def optimize_llvm_ir(llvm_input: CompletedProcess[bytes]) -> CompletedProcess[bytes]:
    opt_cmd = ['opt', '-O3', '-S', '-', '-o', '-']
    try:
        opt_result = subprocess.run(
            opt_cmd, capture_output=True, check=True, input=llvm_input.stdout)
        print(f"Optimization successful.\n"
              f"Warnings:\n\n{opt_result.stderr.decode()}\n"
              f"Output:\n\n{opt_result.stdout.decode()}\n")
        return opt_result
    except subprocess.CalledProcessError as e:
        print(f"Optimization failed with error code {e.returncode}:")


def compile_llvm_ir(filename: str, input_pipe: CompletedProcess[bytes]):
    clang_cmd = ['clang', '-x', 'ir', '-', '-o', f"{filename}.out"]
    try:
        p2 = subprocess.run(clang_cmd, input=input_pipe.stdout,
                            stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Compilation from object files failed with error code: {e.returncode}\n"
              f"Output: {e.stderr.decode()}")
        exit(e.returncode)


def measure_execution_time(path_to_binary: str) -> float:
    start_time = time.perf_counter()
    subprocess.run(f"./{path_to_binary}")
    end_time = time.perf_counter()
    return end_time - start_time


if __name__ == "__main__":
    filename = os.path.basename(args.cpp_file)[:-4]
    clang_stdout = get_ir_from_clang(args.cpp_file)
    if args.S:
        with open(f"{filename}.ll", 'w') as open_file:
            open_file.write(clang_stdout.stdout.decode())
        exit(0)
    opt_stdout = optimize_llvm_ir(clang_stdout)
    compile_llvm_ir(filename, opt_stdout)
    execution_time = measure_execution_time(f"{filename}.out")
    print(f"The binary took {execution_time}s to run.")
