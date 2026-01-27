from collections.abc import Callable, Mapping
from typing import Any

from _typeshed import Incomplete

from triton.runtime.autotuner import Autotuner, Config, Heuristics, autotune, heuristics
from triton.runtime.cache import RedisRemoteCacheBackend, RemoteCacheBackend
from triton.runtime.errors import InterpreterError, OutOfResources
from triton.runtime.jit import JITFunction as _JITFunction
from triton.runtime.jit import KernelInterface, MockTensor, TensorWrapper, reinterpret

from .driver import driver

__all__ = [
    "autotune",
    "Autotuner",
    "Config",
    "driver",
    "Heuristics",
    "heuristics",
    "InterpreterError",
    "JITFunction",
    "KernelInterface",
    "MockTensor",
    "OutOfResources",
    "RedisRemoteCacheBackend",
    "reinterpret",
    "RemoteCacheBackend",
    "TensorWrapper",
]

type _GridType = tuple[int, ...] | Callable[[Mapping[str, Incomplete]], tuple[int, ...]]

class JITFunction[T: Callable[..., Any]](_JITFunction[T]):
    def __getitem__(self, grid: _GridType) -> T: ...
    def run(
        self, *args: Incomplete, grid: _GridType, warmup: bool, **kwargs: Incomplete
    ) -> None: ...
    def warmup(
        self, *args: Incomplete, grid: _GridType, **kwargs: Incomplete
    ) -> _CompiledKernel[T]: ...
    __call__: T  # type: ignore

class _CompiledKernel[T: Callable[..., Any]]:
    metadata: Incomplete
    packed_metadata: Incomplete
    src: Incomplete
    hash: Incomplete
    name: Incomplete
    asm: Incomplete
    metadata_group: Incomplete
    kernel: Incomplete
    module: Incomplete
    function: Incomplete
    _run: Incomplete

    n_regs: int
    n_spills: int
    n_max_threads: int

    def _init_handles(self) -> None: ...
    def __getitem__(self, grid: _GridType) -> T: ...
    def run(
        self, *args: Incomplete, grid: _GridType, warmup: bool, **kwargs: Incomplete
    ) -> None: ...

    # Fallback
    def __getattr__(self, name: str) -> Incomplete: ...
