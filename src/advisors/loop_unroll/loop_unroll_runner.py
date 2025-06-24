# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for communicate with compiler for unroll decisions"""

import ctypes
import dataclasses
import io
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from time import sleep
from typing import Any, BinaryIO, Callable, List, Optional, Tuple, Union, final

from typing_extensions import override

import utils
from advisors import log_reader
from advisors.mc_runner import CompilerCommunicator

logger = logging.getLogger(__name__)


def send_instrument_response(f: BinaryIO, response: Optional[Tuple[str, str]]):
    if response is None:
        f.write(bytes([0]))
        f.flush()
    else:
        f.write(bytes([1]))
        begin = response[0].encode("ascii") + bytes([0])
        end = response[1].encode("ascii") + bytes([0])
        f.write(begin)
        f.write(end)
        f.flush()


@dataclasses.dataclass(frozen=True)
class UnrollFactorResult:
    factor: int
    action: bool
    module: bytes


@dataclasses.dataclass(frozen=True)
class UnrollDecision:
    features: List
    results: List[UnrollFactorResult]


@dataclasses.dataclass(frozen=True)
class CompilationResult:
    module: bytes
    features_spec: List
    advice_spec: List
    num_decisions: int


@final
class LoopUnrollCompilerCommunicator(CompilerCommunicator):
    def __init__(
        self,
        input_name: str,
        debug,
    ):
        """
        on_features: operation on tensor with feature values
        on_heuristic: operation on default decision of compiler
        on_action: operation on whether given unroll decision succeeded
        """
        super().__init__(input_name, debug)
        self.features = []

        self.tensor_mode = "numpy"
        # self.tensor_mode = 'TensorValue'

        self.features_spec = None
        self.advice_spec = None

    def on_features_collect(self, tensor_values):
        if self.tensor_mode == "numpy":
            tensor_values = [tv.to_numpy() for tv in tensor_values]
        self.features.append(tensor_values)

    def on_heuristic_print(self, heuristic):
        logger.debug(heuristic)

    def on_action_print(self, action):
        logger.debug(action)

    def on_action_save(self, action):
        logger.debug(f"Saving action {action}")
        self.cur_action = action

    def read_heuristic(self, fc):
        event = json.loads(fc.readline())
        logger.debug("Read" + str(event))
        assert "heuristic" in event
        heuristic = int.from_bytes(fc.read(8), byteorder=sys.byteorder, signed=True)
        logger.debug(heuristic)
        fc.readline()
        return heuristic

    def read_action(self, fc):
        event = json.loads(fc.readline())
        logger.debug("Read" + str(event))
        assert "action" in event
        action = bool(int.from_bytes(fc.read(1)))
        logger.debug(action)
        fc.readline()
        return action

    def get_advice_spec(self):
        return self.advice_spec

    def get_features_spec(self):
        return self.features_spec

    def communicate_with_proc(
        self,
        compiler_proc: subprocess.Popen[bytes],
        advice: Callable[[str, list[log_reader.TensorValue], int], int],
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

        logger.debug(f"Starting communication")
        # threading.current_thread().name = "loop-unroll-communicator"
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
                            f"Timeout for opt in loop unroll runner: opt took longer than {timeout} seconds to complete."
                        )
                        # utils.terminate(compiler_proc) # we don't terminate the process here because it leads to issues when two threads call terminate() in rapid succession
                        raise TimeoutError()
                    return True

                set_nonblocking(fc)
                while no_stop_event():
                    ia = input_available()
                    if ia == "dead":
                        return None
                    elif ia == "yes":
                        break
                    elif ia == "no":
                        sleep(0)
                    else:
                        assert False
                #
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
                    tensor_values = []
                    for fv in features:
                        # logger.debug(fv.to_numpy())
                        # logger.debug(log_reader.string_tensor_value(fv))
                        tensor_values.append(fv)

                    if on_features:
                        on_features(tensor_values)

                    heuristic = self.read_heuristic(fc)
                    if on_heuristic:
                        on_heuristic(heuristic)

                    log_reader.send(
                        tc,
                        advice(utils.LOOP_UNROLL, tensor_values, heuristic),
                        advice_spec,
                    )

                    action = self.read_action(fc)
                    if on_action:
                        try:
                            on_action(action)
                        except utils.MonteCarloError as e:
                            # utils.terminate(
                            #     compiler_proc
                            # )  # do want to terminate, since action only raises exception when we have path we dont want to continue anymore -> no need to let inline continue, but termination is handled in corresponding compile_once function
                            raise e

                    send_instrument_response(tc, None)
                    set_nonblocking(fc)

                set_blocking(fc)
