import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def basename(file:str) -> str:
    return Path(file).stem

def get_cmd_output(
        cmd,
        stdin=None,
        timeout=None
    ):
        logger.debug(f"Running cmd: {' '.join(cmd)}")

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
            except subprocess.TimeoutExpired as e:
                print(f"Some error {e}")
                exit(-1)
               
            if status != 0:
                logger.error(f"Exit with status {status}")
                logger.error(f"Command run: {' '.join(cmd)}")
                logger.error(f"Output:")
                logger.error(errs.decode())

                logger.error("Failed.")
                exit(status)

            logger.debug("Finished.")
            logger.debug(f"Output: {outs.decode()}")
            return outs

def readout_mc_inline_timer(input:str) -> int|None:
    re_match = re.search("MC_INLINE_TIMER ([0-9]+)", input)
    if re_match is None:
        return None
    else:
        f = int(re_match.group(1))
        return f


if __name__ == "__main__":
    pass
