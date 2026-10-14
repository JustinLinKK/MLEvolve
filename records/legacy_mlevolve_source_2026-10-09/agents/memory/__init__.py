"""
Memory module: persistent context and global memory for the search process.
"""

from .record import MemRecord

try:
    from .retriever import HybridRetriever
except ModuleNotFoundError:
    HybridRetriever = None  # type: ignore

try:
    from .global_memory import GlobalMemoryLayer
except ModuleNotFoundError:
    GlobalMemoryLayer = None  # type: ignore

__all__ = [
    'HybridRetriever',
    'MemRecord',
    'GlobalMemoryLayer',
]
