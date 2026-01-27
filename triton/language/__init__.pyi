from collections.abc import Callable, Sequence
from functools import total_ordering
from typing import Literal, Self, overload

from _typeshed import Incomplete

from triton.language.core import dtype, pointer_type

__all__ = [
    "PropagateNan",
    "TRITON_MAX_TENSOR_NUMEL",
    "load_tensor_descriptor",
    "store_tensor_descriptor",
    "make_tensor_descriptor",
    "tensor_descriptor",
    "abs",
    "add",
    "advance",
    "arange",
    "argmax",
    "argmin",
    "associative_scan",
    "assume",
    "async_task",
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
    "bfloat16",
    "bitonic_merge",
    "block_type",
    "broadcast",
    "broadcast_to",
    "cat",
    "cast",
    "cdiv",
    "ceil",
    "clamp",
    "condition",
    "const",
    "constexpr",
    "constexpr_type",
    "cos",
    "cumprod",
    "cumsum",
    "debug_barrier",
    "device_assert",
    "device_print",
    "div_rn",
    "dot",
    "dot_scaled",
    "dtype",
    "erf",
    "exp",
    "exp2",
    "expand_dims",
    "extra",
    "fdiv",
    "flip",
    "float16",
    "float32",
    "float64",
    "float8e4b15",
    "float8e4nv",
    "float8e4b8",
    "float8e5",
    "float8e5b16",
    "floor",
    "fma",
    "full",
    "gather",
    "histogram",
    "inline_asm_elementwise",
    "interleave",
    "int1",
    "int16",
    "int32",
    "int64",
    "int8",
    "join",
    "load",
    "log",
    "log2",
    "make_block_ptr",
    "map_elementwise",
    "math",
    "max",
    "max_constancy",
    "max_contiguous",
    "maximum",
    "min",
    "minimum",
    "multiple_of",
    "num_programs",
    "pair_uniform_to_normal",
    "permute",
    "philox",
    "philox_impl",
    "pi32_t",
    "pointer_type",
    "program_id",
    "rand",
    "rand4x",
    "randint",
    "randint4x",
    "randn",
    "randn4x",
    "range",
    "ravel",
    "reduce",
    "reduce_or",
    "reshape",
    "rsqrt",
    "slice",
    "sigmoid",
    "sin",
    "softmax",
    "sort",
    "split",
    "sqrt",
    "sqrt_rn",
    "static_assert",
    "static_print",
    "static_range",
    "store",
    "sum",
    "swizzle2d",
    "target_info",
    "tensor",
    "topk",
    "trans",
    "tuple",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "uint_to_uniform_float",
    "umulhi",
    "view",
    "void",
    "where",
    "xor_sum",
    "zeros",
    "zeros_like",
]

PropagateNan: Incomplete
TRITON_MAX_TENSOR_NUMEL: int

type constexpr = Incomplete
type constexpr_type = Incomplete

void: dtype
int1: dtype
int8: dtype
int16: dtype
int32: dtype
int64: dtype
uint8: dtype
uint16: dtype
uint32: dtype
uint64: dtype
float8e5: dtype
float8e5b16: dtype
float8e4nv: dtype
float8e4b8: dtype
float8e4b15: dtype
float16: dtype
bfloat16: dtype
float32: dtype
float64: dtype
pi32_t: pointer_type

# ===  Programming Model  ===

type _Scalar = int | float | bool
type _TensorLike = tensor | _Scalar
type _SliceKey = slice | int | None

@total_ordering
class tensor:
    __add__ = _binary_op
    __radd__ = _binary_op
    __sub__ = _binary_op
    __rsub__ = _binary_op
    __mul__ = _binary_op
    __rmul__ = _binary_op
    __truediv__ = _binary_op
    __rtruediv__ = _binary_op
    __floordiv__ = _binary_op
    __rfloordiv__ = _binary_op
    __mod__ = _binary_op
    __rmod__ = _binary_op

    __neg__ = _unary_op
    __invert__ = _unary_op

    __and__ = _binary_op
    __rand__ = _binary_op
    __or__ = _binary_op
    __ror__ = _binary_op
    __xor__ = _binary_op
    __rxor__ = _binary_op
    __lshift__ = _binary_op
    __rlshift__ = _binary_op
    __rshift__ = _binary_op
    __rrshift__ = _binary_op

    __eq__ = _binary_op  # type: ignore
    __lt__ = _binary_op

    def __getitem__(self, slices: _SliceKey | tuple[_SliceKey, ...]) -> tensor: ...
    @property
    def T(self) -> tensor: ...
    @property
    def type(self) -> dtype: ...

    abs = _unary_op
    argmax = _argminmax
    argmin = _argminmax
    broadcast_to = _broadcast_to
    cast = _cast
    cdiv = _binary_op
    ceil = _unary_op
    cos = _unary_op
    div_rn = _binary_op
    erf = _unary_op
    exp = _unary_op
    exp2 = _unary_op
    expand_dims = _expand_dims
    floor = _unary_op
    log = _unary_op
    log2 = _unary_op
    logical_and = _binary_op
    logical_or = _binary_op
    max = _minmax
    min = _minmax
    permute = _permute
    ravel = _ravel
    reduce = _reduce
    reshape = _reshape
    rsqrt = _unary_op
    sigmoid = _unary_op
    sin = _unary_op
    softmax = _softmax
    split = _split
    sqrt = _unary_op
    sqrt_rn = _unary_op
    store = _store
    sum = _sum
    trans = _trans
    to = _cast
    view = _view
    xor_sum = _xor_sum

def program_id(axis: int) -> tensor: ...
def num_programs(axis: int) -> tensor: ...

# ===  Creation Ops  ===

def arange(start: int, end: int) -> tensor: ...
def cat(input: tensor, other: tensor, can_reorder: bool = False) -> tensor: ...
def full(shape: tuple[int, ...], value: _Scalar, dtype: dtype) -> tensor: ...
def zeros(shape: tuple[int, ...], dtype: dtype) -> tensor: ...
def zeros_like(input: tensor) -> tensor: ...
def _cast(
    dtype: dtype,
    fp_downcast_rounding: Literal["rtne", "rtz", None] = None,
    bitcast: bool = False,
) -> tensor: ...

cast = _cast

# ===  Shape Manipulation Ops ===

def broadcast(input: _TensorLike, other: _TensorLike) -> tuple[tensor, tensor]: ...
@overload
def _broadcast_to(input: _TensorLike, *shape: int) -> tensor: ...
@overload
def _broadcast_to(input: _TensorLike, shape: tuple[int, ...]) -> tensor: ...
def _expand_dims(input: tensor, axis: int | Sequence[int]) -> tensor: ...
def interleave(a: tensor, b: tensor) -> tensor: ...
@overload
def _permute(input: tensor, first_dim: int, *rest_dims: int) -> tensor: ...
@overload
def _permute(input: tensor, dims: tuple[int, *tuple[int, ...]]) -> tensor: ...
def _ravel(x: tensor, can_reorder: bool = False) -> tensor: ...
@overload
def _reshape(input: _TensorLike, *shape: int, can_reorder: bool = False) -> tensor: ...
@overload
def _reshape(
    input: _TensorLike, shape: tuple[int, ...], can_reorder: bool = False
) -> tensor: ...
def _split(a: tensor) -> tuple[tensor, tensor]: ...
@overload
def _trans(input: tensor, *dims: int) -> tensor: ...
@overload
def _trans(input: tensor, dims: tuple[int, ...]) -> tensor: ...
@overload
def _view(input: _TensorLike, *shape: int) -> tensor: ...
@overload
def _view(input: _TensorLike, shape: tuple[int, ...]) -> tensor: ...

broadcast_to = _broadcast_to
expand_dims = _expand_dims
join = _binary_op
permute = _permute
ravel = _ravel
reshape = _reshape
split = _split
trans = _trans
view = _view

# ===  Memory/Pointer Ops  ===

type _PointerType = Incomplete

def load(
    pointer: _PointerType,
    mask: _TensorLike | None = None,
    other: _TensorLike | None = None,
    boundary_check: tuple[int, ...] = (),
    padding_option: Literal["", "zero", "nan"] = "",
    cache_modifier: Literal["", ".ca", ".cg", ".cv"] = "",
    eviction_policy: Literal["", "evict_first", "evict_last"] = "",
    volatile: bool = False,
) -> tensor: ...
def _store(
    pointer: _PointerType,
    value: _TensorLike,
    mask: _TensorLike | None = None,
    boundary_check: tuple[int, ...] = (),
    cache_modifier: Literal["", ".wb", ".cg", ".cs", ".wt"] = "",
    eviction_policy: Literal["", "evict_first", "evict_last"] = "",
) -> None: ...

store = _store

# ===  Math Ops  ===

def _unary_op(a: _TensorLike) -> tensor: ...
def _binary_op(a: _TensorLike, b: _TensorLike) -> tensor: ...
def _binary_op_of(
    a: _TensorLike,
    b: _TensorLike,
    sanitize_overflow: constexpr = True,
) -> tensor: ...
def _ternary_op(a: _TensorLike, b: _TensorLike, c: _TensorLike) -> tensor: ...
def _minimaxi(
    x: _TensorLike,
    y: _TensorLike,
    propagate_nan: PropagateNan = ...,
) -> tensor: ...
def _softmax(
    x: _TensorLike,
    dim: int | None = None,
    keep_dims: bool = False,
    ieee_rounding: bool = False,
) -> tensor: ...

# Undocumented
add = _binary_op_of
sub = _binary_op_of
mul = _binary_op_of

abs = _unary_op
cdiv = _binary_op
ceil = _unary_op

def clamp(
    x: _TensorLike,
    min: _TensorLike,
    max: _TensorLike,
    propagate_nan: PropagateNan = ...,
) -> tensor: ...

cos = _unary_op
div_rn = _binary_op
erf = _unary_op
exp = _unary_op
exp2 = _unary_op

def fdiv(x: _TensorLike, y: _TensorLike, ieee_rounding: bool = False) -> tensor: ...

floor = _unary_op
fma = _ternary_op
log = _unary_op
log2 = _unary_op
maximum = _minimaxi
minimum = _minimaxi
rsqrt = _unary_op
sigmoid = _unary_op
sin = _unary_op
softmax = _softmax
sqrt = _unary_op
sqrt_rn = _unary_op
umulhi = _binary_op

# ===  Reduction Ops  ===

def _argminmax(
    input: tensor,
    axis: int | None,
    tie_break_left: bool = True,
    keep_dims: bool = False,
) -> tensor: ...
@overload
def _minmax(
    input: tensor,
    axis: int | None = None,
    return_indices: Literal[False] = False,
    return_indices_tie_break_left: bool = True,
    keep_dims: bool = False,
) -> tensor: ...
@overload
def _minmax(
    input: tensor,
    axis: int | None,
    return_indices: Literal[True],
    return_indices_tie_break_left: bool = True,
    keep_dims: bool = False,
) -> tuple[tensor, tensor]: ...
@overload
def _minmax(
    input: tensor,
    *,
    return_indices: Literal[True],
    return_indices_tie_break_left: bool = True,
    keep_dims: bool = False,
) -> tuple[tensor, tensor]: ...
@overload
def _minmax(
    input: tensor,
    axis: int | None = None,
    return_indices: bool = False,
    return_indices_tie_break_left: bool = True,
    keep_dims: bool = False,
) -> tuple[tensor, tensor] | tensor: ...
@overload
def _reduce(
    input: tensor,
    axis: int | None,
    combine_fn: Callable[[tensor, tensor], tensor],
    keep_dims: bool = False,
) -> tensor: ...
@overload
def _reduce(
    input: tuple[tensor, tensor],
    axis: int | None,
    combine_fn: Callable[[tensor, tensor, tensor, tensor], tuple[tensor, tensor]],
    keep_dims: bool = False,
) -> tuple[tensor, tensor]: ...
@overload
def _reduce[T: tuple[tensor, tensor, tensor, *tuple[tensor, ...]]](
    input: T,
    axis: int | None,
    combine_fn: Callable[
        [tensor, tensor, tensor, tensor, tensor, tensor, *tuple[tensor, ...]], T
    ],
    keep_dims: bool = False,
) -> T: ...
def _sum(
    input: tensor,
    axis: int | None = None,
    keep_dims: bool = False,
    dtype: dtype | None = None,
) -> tensor: ...
def _xor_sum(
    input: tensor, axis: int | None = None, keep_dims: bool = False
) -> tensor: ...

argmax = _argminmax
argmin = _argminmax
max = _minmax
min = _minmax
reduce = _reduce
sum = _sum
xor_sum = _xor_sum

# ===  Iterators ===

class range:
    def __init__(
        self,
        arg1: tensor | int,
        arg2: tensor | int | None = None,
        step: tensor | int | None = None,
        num_stages: int | None = None,
        loop_unroll_factor: int | None = None,
        disallow_acc_multi_buffer: bool = False,
        flatten: bool = False,
        warp_specialize: bool = False,
        disable_licm: bool = False,
    ) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> tensor: ...

class static_range:
    def __init__(
        self,
        arg1: int,
        arg2: int | None = None,
        step: int | None = None,
    ) -> None: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> int: ...
