"""Utility for testing InteractiveModelRunner.

Use it from pass-specific tests by providing a main .py which calls this library's
`run_interactive` with an appropriate callback to provide advice.

From .ll tests, just call the above-mentioned main as a prefix to the opt/llc
invocation (with the appropriate flags enabling the interactive mode)

Examples:
test/Transforms/Inline/ML/interactive-mode.ll
test/CodeGen/MLRegAlloc/interactive-mode.ll
"""

import ctypes
import json
import logging
import log_reader
import io
import math
import os
import subprocess
from typing import Callable, List, Union

logger = logging.getLogger(__name__)


def send(f: io.BufferedWriter, value: Union[int, float], spec: log_reader.TensorSpec):
    """Send the `value` - currently just a scalar - formatted as per `spec`."""

    if spec.element_type == ctypes.c_int64:
        convert_el_func = int
        ctype_func = ctypes.c_int64
    elif spec.element_type == ctypes.c_float:
        convert_el_func = float
        ctype_func = ctypes.c_float
    else:
        print(spec.element_type, "not supported")
        assert False

    if isinstance(value, list):
        to_send = (ctype_func * len(value))(*
                                            [convert_el_func(el) for el in value])
    else:
        to_send = ctype_func(convert_el_func(value))

    assert f.write(bytes(to_send)) == ctypes.sizeof(spec.element_type) * math.prod(
        spec.shape
    )
    f.flush()


def clean_up_process(process: subprocess.Popen[bytes]):
    outs, errs = process.communicate()
    logger.info(f"\n{errs.decode('utf-8')}")
    logger.debug(f"Outs size {len(outs)}")
    status = process.wait()
    logger.debug(f"Status {status}")
    return status


def run_interactive(
    temp_rootname: str,
    make_response: Callable[[List[log_reader.TensorValue]], Union[int, float, list]],
    process_and_args: list[str], before_advice=None, after_advice=None
):

    to_compiler = temp_rootname + ".in"
    from_compiler = temp_rootname + ".out"

    cur_decision = 0

    logger.debug(f"Opening pipes {to_compiler} and {from_compiler}")

    try:
        os.unlink(to_compiler)
    except FileNotFoundError:
        pass
    try:
        os.unlink(from_compiler)
    except FileNotFoundError:
        pass

    os.mkfifo(to_compiler, 0o666)
    os.mkfifo(from_compiler, 0o666)

    compiler_proc = None
    try:
        logger.debug(f"Launching compiler {' '.join(process_and_args)}")
        compiler_proc = subprocess.Popen(
            process_and_args,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stdin=subprocess.PIPE,
        )
        logger.debug("Sending module")
        # compiler_proc.stdin.write(mod)

        # FIXME is this the proper way to close the pipe? if we don't set it to
        # None then the communicate call will try to close it again and raise an
        # error
        # compiler_proc.stdin.close()
        # compiler_proc.stdin = None

        def set_nonblocking(pipe):
            os.set_blocking(pipe.fileno(), False)

        def set_blocking(pipe):
            os.set_blocking(pipe.fileno(), True)

        output_module = b""
        set_nonblocking(compiler_proc.stdout)

        logger.debug("Starting communication")

        with io.BufferedWriter(io.FileIO(to_compiler, "w+b")) as tc:
            with io.BufferedReader(io.FileIO(from_compiler, "r+b")) as fc:
                # We need to set the reading pipe to nonblocking for the purpose
                # of peek'ing and checking if it is readable without blocking
                # and watch for the process diyng as well. We rever to blocking
                # mode for the actual communication.

                # while True:
                # print(compiler_proc.poll()) if compiler_proc is not None else ()
                # print(fc.readline())

                def input_available():
                    nonlocal output_module
                    assert compiler_proc.stdout
                    output = compiler_proc.stdout.read()
                    if output is not None:
                        output_module += output
                    if len(fc.peek(1)) > 0:
                        return "yes"
                    if compiler_proc.poll() is not None:
                        return "dead"
                    return "no"

                set_nonblocking(fc)
                while True:
                    ia = input_available()
                    if ia == "dead":
                        assert compiler_proc.stderr
                        logger.error(compiler_proc.stderr.read().decode())
                        exit(-1)
                    elif ia == "yes":
                        break
                    elif ia == "no":
                        continue
                    else:
                        assert False

                set_blocking(fc)

                header, tensor_specs, _, advice_spec = log_reader.read_header(
                    fc)
                context = None

                set_nonblocking(fc)
                while True:
                    ia = input_available()
                    if ia == "dead":
                        break
                    elif ia == "yes":
                        ...
                    elif ia == "no":
                        continue
                    else:
                        assert False

                    # set_blocking(fc)

                    next_event = fc.readline()
                    if not next_event:
                        break
                    event = json.loads(next_event)
                    # when runnning with O3 enabled, we have multiple passes, with multiple headers
                    if "observation" not in event and "context" not in event:
                        assert event == header
                        continue

                    while (len(fc.peek(1)) <= 0):
                        if compiler_proc.poll() is not None:
                            logger.warning(
                                "opt gave context but not observations")
                            clean_up_process(compiler_proc)
                            return

                    logger.debug(f"Len of readable content {len(fc.peek(1))}")
                    (
                        last_context,
                        observation_id,
                        features,
                        _,
                    ) = log_reader.read_one_observation(
                        context, next_event, fc, tensor_specs, None
                    )
                    if last_context != context:
                        logger.debug(f"context: {last_context}")
                    context = last_context
                    logger.debug(f"observation: {observation_id}")
                    tensor_values: list[log_reader.TensorValue] = []
                    for fv in features:
                        # logger.debug(fv.to_numpy())
                        logger.debug(log_reader.string_tensor_value(fv))
                        tensor_values.append(fv)
                    if before_advice is not None:
                        before_advice(tc, fc)
                    send(tc, make_response(tensor_values), advice_spec)
                    if after_advice is not None:
                        after_advice(tc, fc)

                    cur_decision += 1
                    # set_nonblocking(fc)

                set_blocking(fc)

        set_blocking(compiler_proc.stdout)

        status = clean_up_process(compiler_proc)
        if status != 0:
            exit(-1)

        # if self.emit_assembly:
        #     outs = outs.decode("utf-8")
        #     logger.debug("Output module:")
        #     logger.debug(outs)

        # return CompilationResult(
        #     module=outs,
        #     features_spec=tensor_specs,
        #     advice_spec=advice_spec,
        #     num_decisions=cur_decision,
        # )
    finally:
        if compiler_proc is not None:
            compiler_proc.kill()
