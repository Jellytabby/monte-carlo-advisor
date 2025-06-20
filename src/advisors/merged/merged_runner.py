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
from typing import Any, BinaryIO, Callable, List, Optional, Tuple, Union, final

from typing_extensions import override

import utils
from advisors import log_reader
from advisors.inline.inline_runner import InlineCompilerCommunicator
from advisors.loop_unroll.loop_unroll_runner import LoopUnrollCompilerCommunicator
from advisors.mc_runner import CompilerCommunicator

logger = logging.getLogger(__name__)


@final
class MergedCompilerCommunicator(CompilerCommunicator):
    def __init__(self, input_name: str, debug):
        super().__init__(input_name, debug)
        # self.input_name = input_name
        # self.debug = debug
        self.stop_event: threading.Event = threading.Event()
        self.inline_comm = InlineCompilerCommunicator(
            input_name, False, self.stop_event
        )
        self.loop_comm = LoopUnrollCompilerCommunicator(input_name, False)

    @override
    def clean_up_pipes(self):
        self.inline_comm.clean_up_pipes()
        self.loop_comm.clean_up_pipes()

    def communicate_with_proc(
        self,
        compiler_proc: subprocess.Popen[bytes],
        advice: Callable[[str, list[log_reader.TensorValue], Optional[int]], int],
        on_features: Optional[Callable[[list[log_reader.TensorValue]], Any]] = None,
        on_heuristic: Optional[Callable[[int], Any]] = None,
        on_action: Optional[Callable[[bool], Any]] = None,
        timeout: Optional[float] = None,
    ):
        self.stop_event.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_inline = executor.submit(
                self.inline_comm.communicate_with_proc,
                compiler_proc,
                advice,
                None,
                None,
                None,
                timeout,
            )
            fut_loop = executor.submit(
                self.loop_comm.communicate_with_proc,
                compiler_proc,
                advice,
                None,
                None,
                on_action,
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
