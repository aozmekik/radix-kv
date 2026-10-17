"""KV-cache block manager and prefix caches for LLM inference servers.

The memory-management core of a serving engine's KV cache, implemented in C++20:

- :class:`BlockAllocator`: fixed pool of reference-counted blocks (PagedAttention).
- :class:`Sequence`: tokens plus block table, with copy-on-write :meth:`Sequence.fork`.
- :class:`HashPrefixCache`: vLLM-style chained block hashes with LRU eviction.
- :class:`RadixPrefixCache`: SGLang-style radix tree with LRU eviction of leaves.
- :func:`generate_trace` and :func:`simulate`: a scheduler simulation over synthetic traces.

No GPU and no tensors: the library manages block ids and metadata, as a scheduler does.
"""

from ._core import (
    AppendResult,
    BlockAllocator,
    CacheStats,
    ChatWorkload,
    CopyOp,
    FewShotWorkload,
    HashPrefixCache,
    InsertResult,
    InvalidHandleError,
    LengthDistribution,
    NodeHandle,
    OutOfBlocksError,
    PrefixMatch,
    RadixKvError,
    RadixPrefixCache,
    RagWorkload,
    RandomWorkload,
    Sequence,
    SimulationReport,
    SimulatorConfig,
    Strategy,
    Trace,
    TraceRequest,
    __version__,
    generate_trace,
    simulate,
)

__all__ = [
    "AppendResult",
    "BlockAllocator",
    "CacheStats",
    "ChatWorkload",
    "CopyOp",
    "FewShotWorkload",
    "HashPrefixCache",
    "InsertResult",
    "InvalidHandleError",
    "LengthDistribution",
    "NodeHandle",
    "OutOfBlocksError",
    "PrefixMatch",
    "RadixKvError",
    "RadixPrefixCache",
    "RagWorkload",
    "RandomWorkload",
    "Sequence",
    "SimulationReport",
    "SimulatorConfig",
    "Strategy",
    "Trace",
    "TraceRequest",
    "__version__",
    "generate_trace",
    "simulate",
]
