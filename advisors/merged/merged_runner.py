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

import asyncio
import ctypes
import dataclasses
import io
import logging
import math
import subprocess
import sys
from typing import Any, BinaryIO, Callable, List, Optional, Tuple, Union

from advisors.inline.inline_runner import InlineCompilerCommunicator
from advisors.loop_unroll.loop_unroll_runner import \
    LoopUnrollCompilerCommunicator

from .. import log_reader

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
    # logger.info(f"\n{selective_mlgo_output(errs.decode('utf-8'))}")
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


class MergedCompilerCommunicator:
    def __init__(
        self,
        emit_assembly,
        debug,
    ):
        self.debug = debug

    def compile_once(
        self,
        process_and_args: list[str],
        advice: Callable[[List[log_reader.TensorValue], int], Union[int, float, list]],
        on_features: Optional[Callable[[list[log_reader.TensorValue]], Any]] = None,
        on_heuristic: Optional[Callable[[int], Any]] = None,
        on_action: Optional[Callable[[bool], Any]] = None,
        get_instrument_response=lambda _: None,
    ):
        self.process_and_args = process_and_args
        self.advice = advice
        self.on_features = on_features
        self.on_heuristic = on_heuristic
        self.on_action = on_action

        compiler_proc = None
        try:
            print(process_and_args)
            logger.debug(f"Launching compiler {' '.join(process_and_args)}")
            compiler_proc = subprocess.Popen(
                process_and_args,
                stderr=subprocess.DEVNULL if not self.debug else sys.stderr,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )
            logger.debug(f"Sending module")
            # compiler_proc.stdin.write(mod)

            # FIXME: is this the proper way to close the pipe? if we don't set it to
            # None then the communicate call will try to close it again and raise an
            # error

            output_module = b""
            tensor_specs = None
            advice_spec = None

            asyncio.run(self.spawn_comminicators(compiler_proc, advice))

            status = clean_up_process(compiler_proc)
            if status != 0:
                exit(status)

        finally:
            if compiler_proc is not None:
                compiler_proc.kill()

    async def loop_wrapper(self, compiler_proc, advice):
        LoopUnrollCompilerCommunicator(False, True).communicate_with_proc(
            compiler_proc, advice
        )

    async def spawn_comminicators(self, compiler_proc, advice):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.loop_wrapper(compiler_proc, advice))
        print("RAAAAA")
