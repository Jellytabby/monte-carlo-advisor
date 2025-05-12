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

def get_output(
        cmd,
        stdin=None,
        timeout=None
    ):
        # logger.debug(f"Running cmd: {' '.join(cmd)}")

        # sns = False if cmd[0] != "clang++" else True
        # Only clang++ needs it but just in case let's use a process group for everything

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=(subprocess.PIPE if stdin is not None else None),
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
            #
            # if status != 0 and not allow_fail:
            #     logger.debug(f"Exit with status {status}")
            #     logger.debug(f"cmd: {' '.join(cmd)}")
            #     logger.debug(f"output:")
            #     errs_decoded = errs.decode("utf-8")
            #     logger.debug(errs_decoded)
            #
            #     logger.debug("Failed.")
            #     raise ExecFailTy(cmd, outs.decode("utf-8"), errs_decoded)
            #
            # logger.debug("Finished.")
            except subprocess.CalledProcessError as e:
                print(f"Compilation failed with error code {e.returncode}:")
                print(e.stderr)
                exit(e.returncode)
                
            return outs, errs


def get_ir_from_clang(file_path: str) -> bytes:
    clang_cmd = ['clang++',  '-cc1', '-O1', '-disable-llvm-passes',
                 '-emit-llvm'] + args.flags + [file_path, '-o', '-']
    clang_result, errs = get_output(clang_cmd)
    return clang_result


def optimize_llvm_ir(llvm_input: bytes, opt_flags: list[str]) -> bytes:
    opt_cmd = ['opt', '-S'] + opt_flags + ['-', '-o', '-']
    opt_result, errs = get_output(opt_cmd, llvm_input)
    return opt_result


def compile_llvm_ir(filename: str, input_pipe: bytes):
    compile_cmd = ['clang++', '-x', 'ir', '-', '-o', f"{filename}.out"]
    get_output(compile_cmd, input_pipe)
    


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
            open_file.write(clang_stdout.decode())
        exit(0)
    opt_stdout = optimize_llvm_ir(clang_stdout, [])
    compile_llvm_ir(filename, opt_stdout)
    execution_time = measure_execution_time(f"{filename}.out")
    print(f"The binary took {execution_time}s to run.")
