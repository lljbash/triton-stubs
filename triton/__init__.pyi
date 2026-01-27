from collections.abc import Callable, Iterable
from typing import Any, overload

from _typeshed import Incomplete

"""isort:skip_file"""
__version__: str

from triton.runtime import (
    autotune,
    Config,
    heuristics,
    JITFunction,
    KernelInterface,
    reinterpret,
    TensorWrapper,
    OutOfResources,
    InterpreterError,
    MockTensor,
)
from triton.runtime.jit import constexpr_function
from triton.runtime._async_compile import AsyncCompileMode, FutureKernel
from triton.compiler import compile, CompilationError
from triton.errors import TritonError
from triton.runtime._allocation import set_allocator

from . import language
from . import testing
from . import tools
from . import runtime

__all__ = [
    "AsyncCompileMode",
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "constexpr_function",
    "FutureKernel",
    "heuristics",
    "InterpreterError",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "MockTensor",
    "must_use_result",
    "next_power_of_2",
    "OutOfResources",
    "reinterpret",
    "runtime",
    "set_allocator",
    "TensorWrapper",
    "TritonError",
    "testing",
    "tools",
]

def must_use_result[T: Callable[..., Any]](x: T, s: bool = True) -> T: ...
@overload
def jit[T: Callable[..., Any]](fn: T) -> JITFunction[T]: ...
@overload
def jit[T: Callable[..., Any]](
    *,
    version: Incomplete = None,
    repr: Callable[..., Incomplete] | None = None,
    launch_metadata: Callable[..., Incomplete] | None = None,
    do_not_specialize: Iterable[int | str] | None = None,
    do_not_specialize_on_alignment: Iterable[int | str] | None = None,
    debug: bool | None = None,
    noinline: bool | None = None,
) -> Callable[[T], JITFunction[T]]: ...
def cdiv(x: int, y: int) -> int: ...
def next_power_of_2(x: int) -> int: ...
