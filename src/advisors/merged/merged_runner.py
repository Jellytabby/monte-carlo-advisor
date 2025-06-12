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
import io
import logging
import math
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, BinaryIO, Callable, List, Optional, Tuple, Union

import utils
from advisors import log_reader
from advisors.inline.inline_runner import InlineCompilerCommunicator
from advisors.loop_unroll.loop_unroll_runner import LoopUnrollCompilerCommunicator

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


class MergedCompilerCommunicator:
    def __init__(
        self,
        input_name: str,
        emit_assembly,
        debug,
    ):
        self.input_name = input_name
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
        self.stop_event: threading.Event = threading.Event()

        compiler_proc = None
        try:
            with tempfile.TemporaryFile("b+x") as error_buffer:
                logger.debug(f"Launching compiler {' '.join(process_and_args)}")
                compiler_proc = subprocess.Popen(
                    process_and_args,
                    stderr=subprocess.DEVNULL if not self.debug else error_buffer,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                )
                logger.debug(f"Sending module")

                # FIXME: is this the proper way to close the pipe? if we don't set it to
                # None then the communicate call will try to close it again and raise an
                # error

                inline_comm = InlineCompilerCommunicator(
                    self.input_name, True, self.stop_event
                )
                loop_comm = LoopUnrollCompilerCommunicator(
                    self.input_name, False, False
                )

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_inline = executor.submit(
                        inline_comm.communicate_with_proc, compiler_proc, advice
                    )
                    fut_loop = executor.submit(
                        loop_comm.communicate_with_proc,
                        compiler_proc,
                        advice,
                        None,
                        None,
                        self.on_action,
                    )

                    for fut in as_completed([fut_inline, fut_loop]):
                        try:
                            fut.result()
                        except Exception as e:
                            raise e

                status = utils.clean_up_process(compiler_proc, error_buffer)
                if status != 0:
                    exit(status)

        finally:
            if compiler_proc is not None:
                compiler_proc.kill()
