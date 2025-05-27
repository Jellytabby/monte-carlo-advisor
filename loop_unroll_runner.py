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

import os
import io
import argparse
import dataclasses
import json
import tempfile
import subprocess
import ctypes
import math
import logging
from typing import Any, Callable, Tuple, BinaryIO, Union, List, Optional
import log_reader

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


def clean_up_process(process: subprocess.Popen[bytes]):
    outs, errs = process.communicate()
    logger.info(f"\n{selective_mlgo_output(errs.decode('utf-8'))}")
    logger.debug(f"Outs size {len(outs)}")
    status = process.wait()
    logger.debug(f"Status {status}")
    return status


def selective_mlgo_output(log: str):
    lines = log.splitlines(True)
    lines = [l for l in lines if not l.startswith("unrolling_decision")]
    lines = [l for l in lines if not "ShouldInstrument" in l]
    lines = [("\n" + l) if l.startswith("Loop Unroll") else l for l in lines]
    return "".join(lines)


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


class LoopUnrollCompilerCommunicator:
    def __init__(self, emit_assembly, debug):
        self.emit_assembly = emit_assembly
        self.debug = debug

        self.num_decisions = None
        self.decisions = None

        self.features = []

        self.tensor_mode = "numpy"
        # self.tensor_mode = 'TensorValue'

        self.channel_base = None
        self.to_compiler = None
        self.from_compiler = None

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
        heuristic = int.from_bytes(fc.read(8))
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

    def compile_once(
        self,
        temp_rootname: str,
        make_response: Callable[
            [List[log_reader.TensorValue], int], Union[int, float, list]
        ],
        process_and_args: list[str],
        on_features: Optional[Callable[[list[log_reader.TensorValue]], Any]] = None,
        on_heuristic: Optional[Callable[[int], Any]] = None,
        on_action: Optional[Callable[[bool], Any]] = None,
        get_instrument_response=lambda _: None,
    ):
        """
        on_features: operation on tensor with feature values
        on_heuristic: operation on default decision of compiler
        on_action: operation on whether given unroll decision succeeded
        """
        to_compiler = temp_rootname + ".in"
        from_compiler = temp_rootname + ".out"
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
                stderr=subprocess.DEVNULL if not self.debug else subprocess.PIPE,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )
            logger.debug(f"Sending module")
            # compiler_proc.stdin.write(mod)

            # FIXME is this the proper way to close the pipe? if we don't set it to
            # None then the communicate call will try to close it again and raise an
            # error
            compiler_proc.stdin.close()
            compiler_proc.stdin = None

            def set_nonblocking(pipe):
                os.set_blocking(pipe.fileno(), False)

            def set_blocking(pipe):
                os.set_blocking(pipe.fileno(), True)

            output_module = b""
            tensor_specs = None
            advice_spec = None

            set_nonblocking(compiler_proc.stdout)

            logger.debug(f"Starting communication")

            with io.BufferedWriter(io.FileIO(to_compiler, "w+b")) as tc:
                with io.BufferedReader(io.FileIO(from_compiler, "r+b")) as fc:
                    # We need to set the reading pipe to nonblocking for the purpose
                    # of peek'ing and checking if it is readable without blocking
                    # and watch for the process diyng as well. We rever to blocking
                    # mode for the actual communication.

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
                            return None
                        elif ia == "yes":
                            break
                        elif ia == "no":
                            continue
                        else:
                            assert False

                    set_blocking(fc)

                    header, tensor_specs, _, advice_spec = log_reader.read_header(fc)
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

                        send(
                            tc,
                            make_response(tensor_values, heuristic),
                            advice_spec,
                        )

                        action = self.read_action(fc)
                        if on_action:
                            on_action(action)

                        send_instrument_response(tc, None)

                        set_nonblocking(fc)

                    set_blocking(fc)

            set_blocking(compiler_proc.stdout)

            status = clean_up_process(compiler_proc)
            if status != 0:
                exit(status)

        finally:
            if compiler_proc is not None:
                compiler_proc.kill()
