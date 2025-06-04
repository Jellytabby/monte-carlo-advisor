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
import io
import json
import logging
import math
import os
import subprocess
import tempfile
from threading import Event
from time import sleep
from typing import IO, Callable, List, Union

from .. import log_reader

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
        to_send = (ctype_func * len(value))(*[convert_el_func(el) for el in value])
    else:
        to_send = ctype_func(convert_el_func(value))

    assert f.write(bytes(to_send)) == ctypes.sizeof(spec.element_type) * math.prod(
        spec.shape
    )
    f.flush()


def clean_up_process(process: subprocess.Popen[bytes], error_buffer: IO[bytes]):
    outs, _ = process.communicate()
    status = process.wait()
    error_buffer.seek(0)
    if status != 0:
        logger.error(error_buffer.read().decode())
    else:
        logger.info(f"\n{error_buffer.read().decode('utf-8')}")
    logger.debug(f"Outs size {len(outs)}")
    logger.debug(f"Status {status}")
    return status


class InlineCompilerCommunicator:
    def __init__(self, event = None):
        self.channel_base = type(self).__name__
        self.to_compiler = self.channel_base + ".channel-basename.in"
        self.from_compiler = self.channel_base + ".channel-basename.out"
        self.event:Event|None = event

    def compile_once(
        self,
        process_and_args: list[str],
        advice: Callable[[List[log_reader.TensorValue]], Union[int, float, list]],
        before_advice=None,
        after_advice=None,
    ):

        with tempfile.TemporaryFile("b+x") as error_buffer:
            compiler_proc = None
            try:
                logger.debug(f"Launching compiler {' '.join(process_and_args)}")
                compiler_proc = subprocess.Popen(
                    process_and_args,
                    stderr=error_buffer,
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
                self.communicate_with_proc(
                    compiler_proc, advice, before_advice, after_advice
                )

                status = clean_up_process(compiler_proc, error_buffer)
                if status != 0:
                    exit(status)

            finally:
                if compiler_proc is not None:
                    compiler_proc.kill()

    def communicate_with_proc(
        self,
        compiler_proc: subprocess.Popen[bytes],
        advice: Callable[[list[log_reader.TensorValue], int], Union[int, float, list]],
        before_advice=None,
        after_advice=None,
    ):

        def set_nonblocking(pipe):
            os.set_blocking(pipe.fileno(), False)

        def set_blocking(pipe):
            os.set_blocking(pipe.fileno(), True)

        logger.debug(f"Opening pipes {self.to_compiler} and {self.from_compiler}")

        try:
            os.unlink(self.to_compiler)
        except FileNotFoundError:
            pass
        try:
            os.unlink(self.from_compiler)
        except FileNotFoundError:
            pass

        os.mkfifo(self.to_compiler, 0o666)
        os.mkfifo(self.from_compiler, 0o666)

        set_nonblocking(compiler_proc.stdout)

        logger.debug("Starting communication")

        with io.BufferedWriter(io.FileIO(self.to_compiler, "w+b")) as tc:
            with io.BufferedReader(io.FileIO(self.from_compiler, "r+b")) as fc:
                # We need to set the reading pipe to nonblocking for the purpose
                # of peek'ing and checking if it is readable without blocking
                # and watch for the process diyng as well. We rever to blocking
                # mode for the actual communication.

                # while True:
                # print(compiler_proc.poll()) if compiler_proc is not None else ()
                # print(fc.readline())

                def input_available():
                    if len(fc.peek(1)) > 0:
                        return "yes"
                    if compiler_proc.poll() is not None:
                        return "dead"
                    return "no"

                def no_stop_event():
                    return not (self.event and self.event.is_set())

                set_nonblocking(fc)
                while no_stop_event():
                    ia = input_available()
                    if ia == "dead":
                        return None
                    elif ia == "yes":
                        break
                    elif ia == "no":
                        sleep(0)
                        continue
                    else:
                        assert False

                set_blocking(fc)

                header, tensor_specs, _, advice_spec = log_reader.read_header(fc)
                context = None

                set_nonblocking(fc)
                while no_stop_event():
                    ia = input_available()
                    if ia == "dead":
                        break
                    elif ia == "yes":
                        ...
                    elif ia == "no":
                        sleep(0)
                        continue
                    else:
                        assert False

                    set_blocking(fc)

                    next_event = fc.readline()
                    if not next_event:
                        break
                    event = json.loads(next_event)
                    # when runnning with O3 enabled, we have multiple passes, with multiple headers
                    if "observation" not in event and "context" not in event:
                        assert event == header
                        sleep(0)
                        continue

                    # while len(fc.peek(1)) <= 0:
                    #     if compiler_proc.poll() is not None:
                    #         logger.warning("opt gave context but not observations")
                    #         clean_up_process(compiler_proc)
                    #         return

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
                        # logger.debug(log_reader.string_tensor_value(fv))
                        tensor_values.append(fv)
                    if before_advice is not None:
                        before_advice(tc, fc)
                    send(tc, advice(tensor_values, None), advice_spec)
                    if after_advice is not None:
                        after_advice(tc, fc)

                    set_nonblocking(fc)

                set_blocking(fc)

        set_blocking(compiler_proc.stdout)

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
