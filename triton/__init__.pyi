__all__ = [
    # "AsyncCompileMode",
    # "autotune",
    "cdiv",
    # "CompilationError",
    # "compile",
    # "Config",
    # "constexpr_function",
    # "FutureKernel",
    # "heuristics",
    # "InterpreterError",
    "jit",
    "JITFunction",
    # "KernelInterface",
    # "language",
    # "MockTensor",
    # "must_use_result",
    "next_power_of_2",
    # "OutOfResources",
    # "reinterpret",
    # "runtime",
    # "set_allocator",
    # "TensorWrapper",
    # "TritonError",
    # "testing",
    # "tools",
]

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

def jit[**P, R](fn: Callable[P, R]) -> JITFunction[P, R]: ...

type _GridType = tuple[int, ...] | Callable[[Mapping[str, Any]], tuple[int, ...]]

class JITFunction[**P, R]:
    def __getitem__(self, grid: _GridType) -> Callable[P, R]: ...
    def run(self, *args: Any, grid: _GridType, warmup: bool, **kwargs: Any) -> R: ...
    def warmup(
        self, *args: Any, grid: _GridType, **kwargs: Any
    ) -> _CompiledKernel[P, R]: ...

class _CompiledKernel[**P, R]:
    metadata: Any
    packed_metadata: Any
    src: Any
    hash: Any
    name: Any
    asm: Any
    metadata_group: Any
    kernel: Any
    module: Any
    function: Any
    _run: Any

    def _init_handles(self) -> None:
        self.n_regs: int
        self.n_spills: int
        self.n_max_threads: int
        ...

    def __getitem__(self, grid: _GridType) -> Callable[P, R]: ...
    def run(self, *args: Any, grid: _GridType, warmup: bool, **kwargs: Any) -> R: ...

def cdiv(x: int, y: int) -> int: ...
def next_power_of_2(x: int) -> int: ...
