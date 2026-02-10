from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import Any, Literal, overload

import torch
from _typeshed import Incomplete
from pandas import DataFrame

def nvsmi(attrs: Iterable[str]) -> list[int]: ...
def do_bench_cudagraph(
    fn: Callable[..., Any],
    rep: int = 20,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
) -> Incomplete: ...
def do_bench(
    fn: Callable[..., Any],
    warmup: int = 25,
    rep: int = 20,
    grad_to_none: torch.Tensor | None = None,
    quantiles: list[float] | None = None,
    return_mode: Literal["min", "max", "mean", "median", "all"] = "mean",
) -> Incomplete: ...
def assert_close(
    x: Incomplete,
    y: Incomplete,
    atol: float | None = None,
    rtol: float | None = None,
    err_msg: str = "",
) -> None: ...

class Benchmark:
    def __init__(
        self,
        x_names: list[str],
        x_vals: list[Any],
        line_arg: str,
        line_vals: list[Any],
        line_names: list[str],
        plot_name: str,
        args: dict[str, Any],
        xlabel: str = "",
        ylabel: str = "",
        x_log: bool = False,
        y_log: bool = False,
        styles: list[tuple[str, str]] | None = None,
    ) -> None: ...

class Mark[T]:
    @overload
    def __new__(
        cls, fn: Callable[..., Any], benchmark: Benchmark
    ) -> Mark[DataFrame]: ...
    @overload
    def __new__(
        cls, fn: Callable[..., Any], benchmark: Iterable[Benchmark]
    ) -> Mark[list[DataFrame]]: ...
    @overload
    def run(
        self,
        show_plots: bool = False,
        print_data: bool = False,
        save_path: str = "",
        return_df: Literal[False] = False,
        **kwargs: Incomplete,
    ) -> None: ...
    @overload
    def run(
        self,
        show_plots: bool,
        print_data: bool,
        save_path: str,
        return_df: Literal[True],
        **kwargs: Incomplete,
    ) -> T: ...
    @overload
    def run(
        self,
        *,
        return_df: Literal[True],
        **kwargs: Incomplete,
    ) -> T: ...
    @overload
    def run(
        self,
        show_plots: bool = False,
        print_data: bool = False,
        save_path: str = "",
        return_df: bool = False,
        **kwargs: Incomplete,
    ) -> T | None: ...

@overload
def perf_report(
    benchmarks: Benchmark,
) -> Callable[[Callable[..., Any]], Mark[DataFrame]]: ...
@overload
def perf_report(
    benchmarks: Iterable[Benchmark],
) -> Callable[[Callable[..., Any]], Mark[list[DataFrame]]]: ...
def get_dram_gbps(device: torch.types.Device = None) -> float: ...
def get_max_tensorcore_tflops(
    dtype: torch.dtype, clock_rate: float, device: torch.types.Device = None
) -> float: ...
def cuda_memcheck(**target_kwargs: Incomplete) -> Incomplete: ...
@contextmanager
def set_gpu_clock(
    ref_sm_clock: float = 1350, ref_mem_clock: float = 1215
) -> Generator[tuple[float, float], Any, None]: ...
def get_max_simd_tflops(
    dtype: torch.dtype, clock_rate: float, device: torch.types.Device = None
) -> float: ...
