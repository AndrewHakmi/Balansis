"""Performance tests for TNSIM."""

import pytest
import time
import asyncio
import threading
import multiprocessing
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from decimal import Decimal
import numpy as np
import torch
from typing import List, Dict, Any

from tnsim.core.sets import ZeroSumInfiniteSet
from tnsim.core.cache import TNSIMCache
from tnsim.core.parallel import ParallelTNSIM
from tnsim.integrations.balansis_integration import (
    BalansisCompensator,
    ZeroSumAttention
)


class PerformanceTimer:
    """Utility for measuring execution time."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        """Returns execution time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class MemoryProfiler:
    """Utility for profiling memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def update_peak(self):
        """Updates peak memory usage."""
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
    
    @property
    def memory_increase_mb(self) -> float:
        """Returns memory increase in MB."""
        if self.initial_memory is None or self.peak_memory is None:
            return 0.0
        return (self.peak_memory - self.initial_memory) / 1024 / 1024


@pytest.mark.performance
class TestZeroSumSetPerformance:
    """Performance tests for ZeroSumInfiniteSet."""
    
    def test_large_set_creation_performance(self):
        """Test performance of creating large sets."""
        sizes = [1000, 5000, 10000, 50000]
        
        for size in sizes:
            elements = [1/n for n in range(1, size + 1)]
            
            with PerformanceTimer(f"Creation of {size} elements") as timer:
                zero_sum_set = ZeroSumInfiniteSet(elements)
            
            print(f"Created set with {size} elements in {timer.elapsed:.4f}s")
            
            # Creation time should grow linearly
            assert timer.elapsed < size * 0.001  # Maximum 1ms per 1000 elements
    
    def test_zero_sum_operation_performance(self, sample_harmonic_set):
        """Test performance of zero sum operations."""
        methods = ['direct', 'compensated', 'stabilized']
        
        for method in methods:
            with PerformanceTimer(f"Zero sum operation ({method})") as timer:
                result = sample_harmonic_set.zero_sum_operation(method=method)
            
            print(f"Zero sum operation ({method}) took {timer.elapsed:.4f}s")
            assert timer.elapsed < 1.0  # Should execute in less than a second
    
    def test_compensating_set_performance(self, sample_harmonic_set):
        """Test performance of finding compensating set."""
        methods = ['direct', 'iterative', 'adaptive']
        
        for method in methods:
            with PerformanceTimer(f"Compensating set ({method})") as timer:
                compensating_set = sample_harmonic_set.find_compensating_set(
                    method=method, max_iterations=100
                )
            
            print(f"Compensating set ({method}) took {timer.elapsed:.4f}s")
            assert timer.elapsed < 5.0  # Should execute in less than 5 seconds
    
    def test_convergence_analysis_performance(self, sample_harmonic_set):
        """Test performance of convergence analysis."""
        tests = ['ratio', 'root', 'integral']
        
        for test_type in tests:
            with PerformanceTimer(f"Convergence analysis ({test_type})") as timer:
                result = sample_harmonic_set.convergence_analysis(test=test_type)
            
            print(f"Convergence analysis ({test_type}) took {timer.elapsed:.4f}s")
            assert timer.elapsed < 2.0  # Should execute in less than 2 seconds
    
    def test_serialization_performance(self, sample_harmonic_set):
        """Test serialization performance."""
        # Test serialization to dict
        with PerformanceTimer("Serialization to dict") as timer:
            data = sample_harmonic_set.to_dict()
        
        print(f"Serialization took {timer.elapsed:.4f}s")
        assert timer.elapsed < 0.1  # Should be very fast
        
        # Test deserialization from dict
        with PerformanceTimer("Deserialization from dict") as timer:
            restored_set = ZeroSumInfiniteSet.from_dict(data)
        
        print(f"Deserialization took {timer.elapsed:.4f}s")
        assert timer.elapsed < 0.1  # Should be very fast
    
    def test_memory_usage_large_sets(self):
        """Test memory usage for large sets."""
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            with MemoryProfiler() as profiler:
                elements = [1/n for n in range(1, size + 1)]
                zero_sum_set = ZeroSumInfiniteSet(elements)
                
                # Perform several operations
                zero_sum_set.zero_sum_operation()
                zero_sum_set.get_partial_sum(0, min(100, size-1))
                
                profiler.update_peak()
            
            print(f"Memory increase for {size} elements: {profiler.memory_increase_mb:.2f} MB")
            
            # Memory usage should be reasonable
            assert profiler.memory_increase_mb < size * 0.001  # Maximum 1KB per element


@pytest.mark.performance
class TestCachePerformance:
    """Performance tests for TNSIMCache."""
    
    def test_cache_write_performance(self):
        """Test cache write performance."""
        cache = TNSIMCache(max_size=10000)
        
        # Test writing large number of items
        num_items = 5000
        
        with PerformanceTimer(f"Writing {num_items} items to cache") as timer:
            for i in range(num_items):
                key = f"key_{i}"
                value = {"data": i, "result": i * 2}
                cache.put(key, value)
        
        print(f"Cache write performance: {timer.elapsed:.4f}s for {num_items} items")
        print(f"Average write time: {timer.elapsed/num_items*1000:.4f}ms per item")
        
        # Average write speed should be high
        assert timer.elapsed / num_items < 0.001  # Less than 1ms per item
    
    def test_cache_read_performance(self):
        """Test cache read performance."""
        cache = TNSIMCache(max_size=10000)
        
        # Fill cache
        num_items = 5000
        for i in range(num_items):
            cache.put(f"key_{i}", {"data": i})
        
        # Test reading
        with PerformanceTimer(f"Reading {num_items} items from cache") as timer:
            for i in range(num_items):
                value = cache.get(f"key_{i}")
                assert value is not None
        
        print(f"Cache read performance: {timer.elapsed:.4f}s for {num_items} items")
        print(f"Average read time: {timer.elapsed/num_items*1000:.4f}ms per item")
        
        # Average read speed should be very high
        assert timer.elapsed / num_items < 0.0005  # Less than 0.5ms per item
    
    def test_cache_eviction_performance(self):
        """Test cache eviction performance."""
        cache_size = 1000
        cache = TNSIMCache(max_size=cache_size, eviction_strategy='lru')
        
        # Fill cache to limit
        for i in range(cache_size):
            cache.put(f"key_{i}", {"data": i})
        
        # Test eviction (add items beyond limit)
        num_evictions = 500
        
        with PerformanceTimer(f"Evicting {num_evictions} items") as timer:
            for i in range(cache_size, cache_size + num_evictions):
                cache.put(f"key_{i}", {"data": i})
        
        print(f"Cache eviction performance: {timer.elapsed:.4f}s for {num_evictions} evictions")
        
        # Eviction should be fast
        assert timer.elapsed / num_evictions < 0.001  # Less than 1ms per eviction
    
    def test_cache_memory_efficiency(self):
        """Test cache memory efficiency."""
        with MemoryProfiler() as profiler:
            cache = TNSIMCache(max_size=5000)
            
            # Fill cache with various data types
            for i in range(1000):
                # Simple data
                cache.put(f"simple_{i}", i)
                
                # Complex data
                cache.put(f"complex_{i}", {
                    "id": i,
                    "data": list(range(10)),
                    "metadata": {"created": time.time()}
                })
                
                profiler.update_peak()
        
        print(f"Cache memory usage: {profiler.memory_increase_mb:.2f} MB for 2000 items")
        
        # Memory usage should be reasonable
        assert profiler.memory_increase_mb < 50  # Less than 50 MB for 2000 items
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access performance."""
        cache = TNSIMCache(max_size=10000)
        num_threads = 10
        operations_per_thread = 500
        
        def worker(thread_id: int):
            """Worker function for thread."""
            for i in range(operations_per_thread):
                key = f"thread_{thread_id}_key_{i}"
                value = {"thread": thread_id, "operation": i}
                
                # Write
                cache.put(key, value)
                
                # Read
                retrieved = cache.get(key)
                assert retrieved is not None
        
        with PerformanceTimer("Concurrent cache access") as timer:
            threads = []
            for thread_id in range(num_threads):
                thread = threading.Thread(target=worker, args=(thread_id,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        total_operations = num_threads * operations_per_thread * 2  # read + write
        print(f"Concurrent access: {total_operations} operations in {timer.elapsed:.4f}s")
        print(f"Operations per second: {total_operations/timer.elapsed:.0f}")
        
        # Should handle many operations per second
        assert total_operations / timer.elapsed > 1000  # More than 1000 ops/sec


@pytest.mark.performance
class TestParallelPerformance:
    """Performance tests for ParallelTNSIM."""
    
    def test_parallel_vs_sequential_performance(self):
        """Compare performance of parallel vs sequential operations."""
        # Create sets for testing
        sets = []
        for i in range(20):
            elements = [1/(n+i) for n in range(1, 101)]
            sets.append(ZeroSumInfiniteSet(elements))
        
        # Sequential execution
        with PerformanceTimer("Sequential execution") as seq_timer:
            seq_results = []
            for zs_set in sets:
                result = zs_set.zero_sum_operation()
                seq_results.append(result)
        
        # Parallel execution
        parallel_tnsim = ParallelTNSIM(max_workers=4)
        
        with PerformanceTimer("Parallel execution") as par_timer:
            par_results = parallel_tnsim.parallel_zero_sum(sets)
        
        print(f"Sequential time: {seq_timer.elapsed:.4f}s")
        print(f"Parallel time: {par_timer.elapsed:.4f}s")
        print(f"Speedup: {seq_timer.elapsed/par_timer.elapsed:.2f}x")
        
        # Results should be the same
        assert len(seq_results) == len(par_results)
        for seq_res, par_res in zip(seq_results, par_results):
            assert abs(seq_res - par_res) < 1e-10
        
        # Parallel execution should be faster (with sufficient load)
        if len(sets) >= 4:  # Enough work for parallelism
            assert par_timer.elapsed < seq_timer.elapsed
    
    def test_parallel_scalability(self):
        """Test parallel operations scalability."""
        # Create large number of sets
        num_sets = 100
        sets = []
        for i in range(num_sets):
            elements = [1/(n+i) for n in range(1, 51)]
            sets.append(ZeroSumInfiniteSet(elements))
        
        worker_counts = [1, 2, 4, 8]
        times = []
        
        for workers in worker_counts:
            parallel_tnsim = ParallelTNSIM(max_workers=workers)
            
            with PerformanceTimer(f"Parallel with {workers} workers") as timer:
                results = parallel_tnsim.parallel_zero_sum(sets)
            
            times.append(timer.elapsed)
            print(f"{workers} workers: {timer.elapsed:.4f}s")
            
            assert len(results) == num_sets
        
        # Time should decrease with more workers (up to a limit)
        assert times[1] <= times[0]  # 2 workers better than 1
        assert times[2] <= times[1]  # 4 workers better than 2
    
    def test_parallel_memory_usage(self):
        """Test memory usage during parallel operations."""
        with MemoryProfiler() as profiler:
            # Create sets
            sets = []
            for i in range(50):
                elements = [1/(n+i) for n in range(1, 201)]
                sets.append(ZeroSumInfiniteSet(elements))
            
            profiler.update_peak()
            
            # Execute parallel operations
            parallel_tnsim = ParallelTNSIM(max_workers=4)
            results = parallel_tnsim.parallel_zero_sum(sets)
            
            profiler.update_peak()
        
        print(f"Parallel operations memory usage: {profiler.memory_increase_mb:.2f} MB")
        
        # Memory usage should be reasonable
        assert profiler.memory_increase_mb < 100  # Less than 100 MB
        assert len(results) == 50
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        # Create large number of operations
        operations = []
        for i in range(200):
            elements = [1/(n+i) for n in range(1, 26)]
            zs_set = ZeroSumInfiniteSet(elements)
            operations.append(('zero_sum', zs_set, {}))
        
        parallel_tnsim = ParallelTNSIM(max_workers=4, chunk_size=10)
        
        with PerformanceTimer("Batch processing") as timer:
            results = parallel_tnsim.batch_process(operations)
        
        print(f"Batch processing: {len(operations)} operations in {timer.elapsed:.4f}s")
        print(f"Operations per second: {len(operations)/timer.elapsed:.0f}")
        
        assert len(results) == len(operations)
        # Should handle many operations per second
        assert len(operations) / timer.elapsed > 50  # More than 50 ops/sec


@pytest.mark.performance
class TestBalansisIntegrationPerformance:
    """Performance tests for Balansis integration."""
    
    def test_compensator_performance(self):
        """Test compensator performance."""
        compensator = BalansisCompensator()
        
        # Test with various data sizes
        sizes = [100, 1000, 5000, 10000]
        
        for size in sizes:
            values = [1/n - 1/(n+1) for n in range(1, size + 1)]
            
            with PerformanceTimer(f"Compensation of {size} values") as timer:
                result = compensator.compensated_sum(values)
            
            print(f"Compensated {size} values in {timer.elapsed:.4f}s")
            
            # Time should grow linearly with size
            assert timer.elapsed < size * 0.0001  # Maximum 0.1ms per element
    
    def test_attention_performance(self):
        """Test attention mechanism performance."""
        configs = [
            {"d_model": 128, "n_heads": 4, "seq_len": 32},
            {"d_model": 256, "n_heads": 8, "seq_len": 64},
            {"d_model": 512, "n_heads": 16, "seq_len": 128},
        ]
        
        for config in configs:
            attention = ZeroSumAttention(
                d_model=config["d_model"],
                n_heads=config["n_heads"]
            )
            
            batch_size = 4
            x = torch.randn(batch_size, config["seq_len"], config["d_model"])
            
            # Warmup
            _ = attention(x)
            
            with PerformanceTimer(f"Attention {config}") as timer:
                for _ in range(10):  # Multiple iterations for stability
                    output, weights = attention(x)
            
            avg_time = timer.elapsed / 10
            print(f"Attention {config}: {avg_time:.4f}s per forward pass")
            
            # Time should be reasonable
            assert avg_time < 1.0  # Less than a second per pass
    
    def test_attention_memory_efficiency(self):
        """Test attention mechanism memory efficiency."""
        with MemoryProfiler() as profiler:
            attention = ZeroSumAttention(d_model=512, n_heads=8)
            
            batch_size, seq_len = 8, 64
            x = torch.randn(batch_size, seq_len, 512)
            
            # Execute multiple passes
            for _ in range(20):
                output, weights = attention(x)
                profiler.update_peak()
                
                # Clear intermediate results
                del output, weights
        
        print(f"Attention memory usage: {profiler.memory_increase_mb:.2f} MB")
        
        # Memory usage should be reasonable
        assert profiler.memory_increase_mb < 200  # Less than 200 MB
    
    def test_large_scale_compensation(self):
        """Test large scale data compensation."""
        compensator = BalansisCompensator()
        
        # Create very large series
        large_size = 100000
        large_series = [1/n for n in range(1, large_size + 1)]
        
        with MemoryProfiler() as mem_profiler:
            with PerformanceTimer("Large scale compensation") as timer:
                result, metrics = compensator.compensate_series(large_series)
        
        print(f"Large scale compensation: {timer.elapsed:.4f}s for {large_size} elements")
        print(f"Memory usage: {mem_profiler.memory_increase_mb:.2f} MB")
        print(f"Compensation error: {metrics.compensation_error:.2e}")
        
        # Should handle large volumes of data
        assert timer.elapsed < 10.0  # Less than 10 seconds
        assert mem_profiler.memory_increase_mb < 100  # Less than 100 MB
        assert isinstance(result, float)


@pytest.mark.performance
class TestIntegratedSystemPerformance:
    """Performance tests for integrated system."""
    
    def test_end_to_end_performance(self):
        """Test end-to-end system performance."""
        # Initialize components
        cache = TNSIMCache(max_size=1000)
        parallel_tnsim = ParallelTNSIM(max_workers=4)
        compensator = BalansisCompensator()
        
        # Create test data
        test_sets = []
        for i in range(50):
            elements = [1/(n+i) for n in range(1, 101)]
            test_sets.append(ZeroSumInfiniteSet(elements))
        
        with PerformanceTimer("End-to-end processing") as timer:
            results = []
            
            for zs_set in test_sets:
                # Check cache
                cache_key = cache.generate_key('zero_sum', zs_set.elements[:10])
                cached_result = cache.get(cache_key)
                
                if cached_result is not None:
                    results.append(cached_result)
                else:
                    # Compute with compensation
                    compensated_sum, _ = compensator.compensate_zero_sum_set(zs_set)
                    
                    # Save to cache
                    cache.put(cache_key, compensated_sum)
                    results.append(compensated_sum)
        
        print(f"End-to-end processing: {timer.elapsed:.4f}s for {len(test_sets)} sets")
        print(f"Cache hit rate: {cache.get_stats()['hit_rate']:.2%}")
        
        assert len(results) == len(test_sets)
        assert timer.elapsed < 5.0  # Should be fast thanks to caching
    
    def test_system_scalability(self):
        """Test system scalability."""
        data_sizes = [10, 50, 100, 200]
        times = []
        
        for size in data_sizes:
            # Create data
            test_sets = []
            for i in range(size):
                elements = [1/(n+i) for n in range(1, 51)]
                test_sets.append(ZeroSumInfiniteSet(elements))
            
            # Initialize system
            cache = TNSIMCache(max_size=size * 2)
            parallel_tnsim = ParallelTNSIM(max_workers=4)
            
            with PerformanceTimer(f"Processing {size} sets") as timer:
                # Parallel processing
                results = parallel_tnsim.parallel_zero_sum(test_sets)
                
                # Cache results
                for i, result in enumerate(results):
                    cache.put(f"result_{i}", result)
            
            times.append(timer.elapsed)
            print(f"Processed {size} sets in {timer.elapsed:.4f}s")
        
        # Time should grow sublinearly thanks to parallelism
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            size_ratio = data_sizes[i] / data_sizes[i-1]
            print(f"Time ratio: {ratio:.2f}, Size ratio: {size_ratio:.2f}")
            
            # Time should grow slower than data size
            assert ratio < size_ratio * 1.5
    
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Execute many operations
        for iteration in range(10):
            # Create temporary objects
            cache = TNSIMCache(max_size=100)
            parallel_tnsim = ParallelTNSIM(max_workers=2)
            
            test_sets = []
            for i in range(20):
                elements = [1/(n+i) for n in range(1, 51)]
                test_sets.append(ZeroSumInfiniteSet(elements))
            
            # Execute operations
            results = parallel_tnsim.parallel_zero_sum(test_sets)
            
            # Cache results
            for i, result in enumerate(results):
                cache.put(f"iter_{iteration}_result_{i}", result)
            
            # Force cleanup
            del cache, parallel_tnsim, test_sets, results
            
            # Check memory every few iterations
            if iteration % 3 == 0:
                current_memory = psutil.Process().memory_info().rss
                memory_increase = (current_memory - initial_memory) / 1024 / 1024
                print(f"Iteration {iteration}: Memory increase: {memory_increase:.2f} MB")
                
                # Memory increase should be limited
                assert memory_increase < 50  # Less than 50 MB increase
    
    def test_concurrent_system_usage(self):
        """Test concurrent system usage."""
        # Shared components
        shared_cache = TNSIMCache(max_size=1000)
        
        def worker(worker_id: int, num_operations: int):
            """Worker function for thread."""
            parallel_tnsim = ParallelTNSIM(max_workers=2)
            compensator = BalansisCompensator()
            
            for i in range(num_operations):
                # Create unique data for each worker
                elements = [1/(n + worker_id * 1000 + i) for n in range(1, 21)]
                zs_set = ZeroSumInfiniteSet(elements)
                
                # Check cache
                cache_key = f"worker_{worker_id}_op_{i}"
                cached_result = shared_cache.get(cache_key)
                
                if cached_result is None:
                    # Compute result
                    result = zs_set.zero_sum_operation()
                    
                    # Compensate
                    compensated_result, _ = compensator.compensate_zero_sum_set(zs_set)
                    
                    # Save to cache
                    shared_cache.put(cache_key, compensated_result)
        
        num_workers = 5
        operations_per_worker = 20
        
        with PerformanceTimer("Concurrent system usage") as timer:
            threads = []
            for worker_id in range(num_workers):
                thread = threading.Thread(
                    target=worker,
                    args=(worker_id, operations_per_worker)
                )
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        total_operations = num_workers * operations_per_worker
        print(f"Concurrent processing: {total_operations} operations in {timer.elapsed:.4f}s")
        print(f"Operations per second: {total_operations/timer.elapsed:.0f}")
        
        # System should efficiently handle concurrent requests
        assert total_operations / timer.elapsed > 20  # More than 20 ops/sec
        
        # Check cache state
        cache_stats = shared_cache.get_stats()
        print(f"Final cache stats: {cache_stats}")
        assert cache_stats['size'] > 0  # Cache should contain data


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__,
        "-v",
        "-m", "performance",
        "--tb=short"
    ])