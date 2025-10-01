"""Core modules for TNSIM (Theory of Null-Sum Infinite Multitudes)."""

from .sets.zero_sum_infinite_set import ZeroSumInfiniteSet
from .cache.tnsim_cache import TNSIMCache, cached_operation, get_global_cache
from .operations.parallel_tnsim import ParallelTNSIM, get_global_parallel_processor

__all__ = [
    'ZeroSumInfiniteSet',
    'TNSIMCache',
    'cached_operation',
    'get_global_cache',
    'ParallelTNSIM',
    'get_global_parallel_processor'
]

__version__ = '1.0.0'
__author__ = 'TNSIM Development Team'
__description__ = 'Theory of Null-Sum Infinite Multitudes - Core Modules'