"""Utility for testing InteractiveModelRunner.

Use it from pass-specific tests by providing a main .py which calls this library's
`run_interactive` with an appropriate callback to provide advice.

From .ll tests, just call the above-mentioned main as a prefix to the opt/llc
invocation (with the appropriate flags enabling the interactive mode)

Examples:
test/Transforms/Inline/ML/interactive-mode.ll
test/CodeGen/MLRegAlloc/interactive-mode.ll
"""

import io
import json
import logging
import os
import subprocess
import threading
import time
from threading import Event
from time import sleep
from typing import Any, Callable, Optional, final

import utils
from advisors import log_reader
from advisors.mc_runner import CompilerCommunicator

logger = logging.getLogger(__name__)


@final
class InlineCompilerCommunicator(CompilerCommunicator):
    def __init__(self, input_name: str, debug, event=None):
        super().__init__(input_name, debug)
        self.event: Event | None = event

    def communicate_with_proc(
        self,
        compiler_proc: subprocess.Popen[bytes],
        advice: Callable[[str, list[log_reader.TensorValue], Optional[int]], int],
        on_features: Optional[Callable[[list[log_reader.TensorValue]], Any]] = None,
        on_heuristic: Optional[Callable[[int], Any]] = None,
        on_action: Optional[Callable[[bool], Any]] = None,
        timeout: Optional[float] = None,
    ):

        def set_nonblocking(pipe):
            os.set_blocking(pipe.fileno(), False)

        def set_blocking(pipe):
            os.set_blocking(pipe.fileno(), True)

        logger.debug(f"Opening pipes {self.to_compiler} and {self.from_compiler}")

        os.mkfifo(self.to_compiler, 0o666)
        os.mkfifo(self.from_compiler, 0o666)

        set_nonblocking(compiler_proc.stdout)

        logger.debug("Starting communication")
        # threading.current_thread().name = "Inline-communicator"
        start = time.time()

        with io.BufferedWriter(io.FileIO(self.to_compiler, "w+b")) as tc:
            with io.BufferedReader(io.FileIO(self.from_compiler, "r+b")) as fc:
                # We need to set the reading pipe to nonblocking for the purpose
                # of peek'ing and checking if it is readable without blocking
                # and watch for the process diyng as well. We rever to blocking
                # mode for the actual communication.

                def input_available():
                    if len(fc.peek(1)) > 0:
                        return "yes"
                    if compiler_proc.poll() is not None:
                        return "dead"
                    return "no"

                def no_stop_event():
                    if timeout and time.time() - start > timeout:
                        logger.debug(
                            f"Timeout for opt in inline runner: opt took longer than {timeout} seconds to complete."
                        )
                        # utils.terminate(compiler_proc) # we don't terminate the process here because it leads to issues when two threads call terminate() in rapid succession
                        raise TimeoutError()
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

                    next_event = fc.readline()
                    if not next_event:
                        break
                    event = json.loads(next_event)
                    if "observation" not in event and "context" not in event:
                        assert event == header
                        sleep(0)
                        continue

                    while len(fc.peek(1)) <= 0:
                        if not no_stop_event():
                            return
                        if compiler_proc.poll() is not None:
                            logger.warning("opt gave context but not observations")
                            utils.clean_up_process(
                                compiler_proc
                            )  # cleanup not terminate, because we want loop unroll to finish
                            return

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
                    if on_features is not None:
                        on_features(tensor_values)
                    log_reader.send(
                        tc, advice(utils.INLINE, tensor_values, None), advice_spec
                    )
                    if on_heuristic is not None:
                        on_heuristic(None)
                set_blocking(fc)
