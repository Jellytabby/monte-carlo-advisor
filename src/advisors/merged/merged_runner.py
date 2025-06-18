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

import logging
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, BinaryIO, Callable, List, Optional, Tuple, Union

import utils
from advisors import log_reader
from advisors.inline.inline_runner import InlineCompilerCommunicator
from advisors.loop_unroll.loop_unroll_runner import LoopUnrollCompilerCommunicator
from advisors.mc_runner import CompilerCommunicator

logger = logging.getLogger(__name__)


class MergedCompilerCommunicator(CompilerCommunicator):
    def __init__(
        self,
        input_name: str,
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
        timeout: Optional[float] = 10,
    ):
        self.process_and_args = process_and_args
        self.advice = advice
        self.on_features = on_features
        self.on_heuristic = on_heuristic
        self.on_action = on_action
        self.stop_event: threading.Event = threading.Event()

        # typechecker shenanigans
        compiler_proc = None
        inline_comm = None
        loop_comm = None
        with tempfile.TemporaryFile("b+x") as error_buffer:
            try:
                logger.debug(f"Launching compiler {' '.join(process_and_args)}")
                compiler_proc = subprocess.Popen(
                    process_and_args,
                    stderr=subprocess.DEVNULL if not self.debug else error_buffer,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                )
                logger.debug(f"Sending module")

                inline_comm = InlineCompilerCommunicator(
                    self.input_name, True, self.stop_event
                )
                loop_comm = LoopUnrollCompilerCommunicator(self.input_name, True)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_inline = executor.submit(
                        inline_comm.communicate_with_proc,
                        compiler_proc,
                        advice,
                        None,
                        None,
                        timeout,
                    )
                    fut_loop = executor.submit(
                        loop_comm.communicate_with_proc,
                        compiler_proc,
                        advice,
                        None,
                        None,
                        self.on_action,
                        timeout,
                    )

                    for fut in as_completed([fut_inline, fut_loop]):
                        try:
                            fut.result()
                        except TimeoutError as t:
                            if fut_inline.done() and fut_loop.done():
                                logger.warning(
                                    f"Timeout: opt timed out after {timeout} seconds"
                                )
                                raise t
                        except utils.MonteCarloError as e:
                            self.stop_event.set()
                            raise e

            finally:
                assert compiler_proc and inline_comm and loop_comm
                inline_comm.clean_up_pipes()
                loop_comm.clean_up_pipes()
                if compiler_proc.returncode is None:
                    utils.terminate(compiler_proc)
                else:
                    status = utils.clean_up_process(compiler_proc, error_buffer)
                    if (
                        status != 0 and status != -15
                    ):  # -15 is us terminating the process
                        logger.error(f"Process failed with error code: {status}")
                        exit(status)
