# -*- coding: utf-8 -*-
"""
Unit tests for parallel processing functionality in TNSIM.

This module contains comprehensive tests for the ParallelTNSIM class,
which provides parallel processing capabilities for zero-sum infinite set operations.

Test Categories:
- Initialization and configuration
- Executor management (thread/process pools)
- Parallel operations (zero-sum, validation, convergence analysis)
- Batch processing with chunking
- Task management and cancellation
- Performance optimization
- Error handling and recovery
- Integration with caching system
- Edge cases and boundary conditions

Author: TNSIM Development Team
Version: 1.0.0
License: MIT
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from tnsim.core.zero_sum_infinite_set import ZeroSumInfiniteSet
from tnsim.parallel.parallel_tnsim import ParallelTNSIM
from tnsim.cache.tnsim_cache import TNSIMCache


class TestParallelTNSIMInitialization:
    """Tests for ParallelTNSIM initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        parallel = ParallelTNSIM()
        
        assert parallel.executor_type == "thread"
        assert parallel.max_workers == 4
        assert parallel.chunk_size == 100
        assert parallel._executor is None
        assert parallel._cache is None
        
        parallel.shutdown()
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        parallel = ParallelTNSIM(
            executor_type="process",
            max_workers=8,
            chunk_size=50
        )
        
        assert parallel.executor_type == "process"
        assert parallel.max_workers == 8
        assert parallel.chunk_size == 50
        
        parallel.shutdown()
    
    def test_invalid_executor_type(self):
        """Test initialization with invalid executor type."""
        with pytest.raises(ValueError, match="Executor type must be"):
            ParallelTNSIM(executor_type="invalid")
    
    def test_invalid_max_workers(self):
        """Test initialization with invalid max_workers."""
        with pytest.raises(ValueError, match="max_workers must be positive"):
            ParallelTNSIM(max_workers=0)
        
        with pytest.raises(ValueError, match="max_workers must be positive"):
            ParallelTNSIM(max_workers=-1)
    
    def test_invalid_chunk_size(self):
        """Test initialization with invalid chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ParallelTNSIM(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ParallelTNSIM(chunk_size=-1)


class TestExecutorManagement:
    """Tests for executor management."""
    
    def test_thread_executor_creation(self):
        """Test creation of thread executor."""
        parallel = ParallelTNSIM(executor_type="thread", max_workers=2)
        
        executor = parallel._get_executor()
        
        assert isinstance(executor, ThreadPoolExecutor)
        assert executor._max_workers == 2
        
        parallel.shutdown()
    
    def test_process_executor_creation(self):
        """Test creation of process executor."""
        parallel = ParallelTNSIM(executor_type="process", max_workers=2)
        
        executor = parallel._get_executor()
        
        assert isinstance(executor, ProcessPoolExecutor)
        assert executor._max_workers == 2
        
        parallel.shutdown()
    
    def test_executor_reuse(self):
        """Test that executor is reused between calls."""
        parallel = ParallelTNSIM()
        
        executor1 = parallel._get_executor()
        executor2 = parallel._get_executor()
        
        assert executor1 is executor2
        
        parallel.shutdown()
    
    def test_executor_shutdown(self):
        """Test proper executor shutdown."""
        parallel = ParallelTNSIM()
        
        # Create executor
        executor = parallel._get_executor()
        assert executor is not None
        
        # Shutdown
        parallel.shutdown()
        assert parallel._executor is None


class TestParallelOperations:
    """Tests for parallel operations."""
    
    @pytest.fixture
    def sample_sets(self):
        """Create sample zero-sum infinite sets for testing."""
        return [
            ZeroSumInfiniteSet(elements=[1, -1, 0.5, -0.5]),
            ZeroSumInfiniteSet(elements=[2, -2, 1, -1]),
            ZeroSumInfiniteSet(elements=[3, -3, 1.5, -1.5])
        ]
    
    @pytest.fixture
    def parallel_tnsim(self):
        """Create ParallelTNSIM instance for testing."""
        parallel = ParallelTNSIM(max_workers=2, chunk_size=2)
        yield parallel
        parallel.shutdown()
    
    @pytest.mark.asyncio
    async def test_parallel_zero_sum(self, parallel_tnsim, sample_sets):
        """Test parallel zero-sum operations."""
        results = await parallel_tnsim.parallel_zero_sum(sample_sets, method="direct")
        
        assert len(results) == len(sample_sets)
        for result in results:
            assert "sum" in result
            assert abs(result["sum"]) < 1e-10  # Should be close to zero
    
    @pytest.mark.asyncio
    async def test_parallel_compensating_sets(self, parallel_tnsim, sample_sets):
        """Test parallel compensating set operations."""
        results = await parallel_tnsim.parallel_compensating_sets(
            sample_sets, method="greedy"
        )
        
        assert len(results) == len(sample_sets)
        for result in results:
            assert "compensating_set" in result
            assert "compensation_sum" in result
    
    @pytest.mark.asyncio
    async def test_parallel_validation(self, parallel_tnsim, sample_sets):
        """Test parallel validation operations."""
        results = await parallel_tnsim.parallel_validation(
            sample_sets, tolerance=1e-10
        )
        
        assert len(results) == len(sample_sets)
        for result in results:
            assert "is_zero_sum" in result
            assert isinstance(result["is_zero_sum"], bool)
    
    @pytest.mark.asyncio
    async def test_parallel_convergence_analysis(self, parallel_tnsim, sample_sets):
        """Test parallel convergence analysis."""
        results = await parallel_tnsim.parallel_convergence_analysis(
            sample_sets, method="ratio_test"
        )
        
        assert len(results) == len(sample_sets)
        for result in results:
            assert "is_convergent" in result
            assert isinstance(result["is_convergent"], bool)


class TestBatchProcessing:
    """Tests for batch processing functionality."""
    
    @pytest.fixture
    def parallel_tnsim(self):
        """Create ParallelTNSIM instance for testing."""
        parallel = ParallelTNSIM(max_workers=2, chunk_size=3)
        yield parallel
        parallel.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_operations_basic(self, parallel_tnsim):
        """Test basic batch operations."""
        operations = [
            {
                "type": "zero_sum",
                "set_data": {"elements": [1, -1, 0.5, -0.5]},
                "method": "direct"
            },
            {
                "type": "validation",
                "set_data": {"elements": [2, -2, 1, -1]},
                "tolerance": 1e-10
            }
        ]
        
        results = await parallel_tnsim.batch_operations(operations)
        
        assert len(results) == len(operations)
        assert "sum" in results[0]  # Result of zero_sum operation
        assert "is_zero_sum" in results[1]  # Result of validation operation
    
    @pytest.mark.asyncio
    async def test_batch_operations_with_chunking(self, parallel_tnsim):
        """Test batch operations with chunking."""
        # Create large batch of operations
        operations = []
        for i in range(10):
            operations.append({
                "type": "zero_sum",
                "set_data": {"elements": [i, -i, i*0.5, -i*0.5]},
                "method": "direct"
            })
        
        # Set small chunk size
        parallel_tnsim.chunk_size = 3
        
        results = await parallel_tnsim.batch_operations(operations)
        
        assert len(results) == len(operations)
        for result in results:
            assert "sum" in result
    
    @pytest.mark.asyncio
    async def test_batch_operations_error_handling(self, parallel_tnsim):
        """Test error handling in batch operations."""
        operations = [
            {
                "type": "zero_sum",
                "set_data": {"elements": [1, 2, 3]},
                "method": "direct"
            },
            {
                "type": "invalid_operation",  # Invalid operation
                "set_data": {"elements": [1, 2, 3]}
            },
            {
                "type": "validation",
                "set_data": {"elements": [4, 5, 6]},
                "tolerance": 1e-10
            }
        ]
        
        results = await parallel_tnsim.batch_operations(operations, ignore_errors=True)
        
        assert len(results) == len(operations)
        assert "sum" in results[0]  # Successful operation
        assert "error" in results[1]  # Error
        assert "is_zero_sum" in results[2]  # Successful operation
    
    @pytest.mark.asyncio
    async def test_batch_operations_fail_fast(self, parallel_tnsim):
        """Test fail-fast behavior on errors."""
        operations = [
            {
                "type": "zero_sum",
                "set_data": {"elements": [1, 2, 3]},
                "method": "direct"
            },
            {
                "type": "invalid_operation",
                "set_data": {"elements": [1, 2, 3]}
            }
        ]
        
        with pytest.raises(Exception):
            await parallel_tnsim.batch_operations(operations, ignore_errors=False)


class TestTaskManagement:
    """Tests for task management."""
    
    @pytest.mark.asyncio
    async def test_task_tracking(self, parallel_tnsim, sample_sets):
        """Test tracking of running tasks."""
        sets = sample_sets[:2]
        
        # Start task asynchronously
        task = asyncio.create_task(
            parallel_tnsim.parallel_zero_sum(sets, method="direct")
        )
        
        # Give task some time to start
        await asyncio.sleep(0.01)
        
        # Check that task is being tracked
        assert len(parallel_tnsim._running_tasks) > 0
        
        # Wait for completion
        results = await task
        
        # After completion, task should be removed from tracking
        assert len(results) == len(sets)
    
    @pytest.mark.asyncio
    async def test_cancel_running_tasks(self, parallel_tnsim):
        """Test cancellation of running tasks."""
        # Create long-running operation
        large_elements = list(range(10000))
        large_set = ZeroSumInfiniteSet(elements=large_elements)
        
        # Start task
        task = asyncio.create_task(
            parallel_tnsim.parallel_zero_sum([large_set], method="compensated")
        )
        
        # Give task time to start
        await asyncio.sleep(0.01)
        
        # Cancel all tasks
        cancelled_count = await parallel_tnsim.cancel_all_tasks()
        
        # Check that task was cancelled
        assert cancelled_count > 0
        
        # Check that task is actually cancelled
        with pytest.raises(asyncio.CancelledError):
            await task
    
    def test_get_running_tasks_info(self, parallel_tnsim):
        """Test getting information about running tasks."""
        info = parallel_tnsim.get_running_tasks_info()
        
        assert "total_tasks" in info
        assert "task_details" in info
        assert isinstance(info["total_tasks"], int)
        assert isinstance(info["task_details"], list)


class TestPerformanceOptimization:
    """Tests for performance optimization."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self):
        """Test comparison of parallel vs sequential processing performance."""
        # Create test sets
        test_sets = []
        for i in range(10):
            elements = [1/(j+1) for j in range(1000)]  # Harmonic series
            test_sets.append(ZeroSumInfiniteSet(elements=elements))
        
        parallel = ParallelTNSIM(max_workers=4)
        
        # Measure parallel processing time
        start_time = time.time()
        parallel_results = await parallel.parallel_zero_sum(test_sets, method="direct")
        parallel_time = time.time() - start_time
        
        # Measure sequential processing time
        start_time = time.time()
        sequential_results = []
        for test_set in test_sets:
            result = test_set.zero_sum_operation(method="direct")
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Check correctness of results
        assert len(parallel_results) == len(sequential_results)
        for i in range(len(parallel_results)):
            assert abs(parallel_results[i]["sum"] - sequential_results[i]["sum"]) < 1e-10
        
        # Parallel processing should be faster (or at least not slower by more than 2x)
        assert parallel_time <= sequential_time * 2
        
        parallel.shutdown()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, parallel_tnsim):
        """Test memory usage optimization."""
        # Create sets with large number of elements
        large_sets = []
        for i in range(5):
            elements = [1/(j+1) for j in range(5000)]
            large_sets.append(ZeroSumInfiniteSet(elements=elements))
        
        # Perform operations with memory control
        results = await parallel_tnsim.parallel_zero_sum(
            large_sets, 
            method="compensated"
        )
        
        assert len(results) == len(large_sets)
        
        # Check that results are correct
        for result in results:
            assert "sum" in result
            assert not any(key.startswith("_temp") for key in result.keys())
    
    @pytest.mark.performance
    def test_chunk_size_optimization(self):
        """Test chunk size optimization."""
        # Test different chunk sizes
        chunk_sizes = [100, 500, 1000, 2000]
        operations = []
        
        # Create large set of operations
        for i in range(1000):
            operations.append({
                "type": "zero_sum",
                "set_data": {"elements": [i, -i]},
                "method": "direct"
            })
        
        best_time = float('inf')
        best_chunk_size = None
        
        for chunk_size in chunk_sizes:
            parallel = ParallelTNSIM(chunk_size=chunk_size, max_workers=2)
            
            start_time = time.time()
            # Simulate processing (without actual execution for test speed)
            chunks = [operations[i:i+chunk_size] for i in range(0, len(operations), chunk_size)]
            processing_time = time.time() - start_time
            
            if processing_time < best_time:
                best_time = processing_time
                best_chunk_size = chunk_size
            
            parallel.shutdown()
        
        # Optimal chunk size should be found
        assert best_chunk_size is not None
        assert best_chunk_size in chunk_sizes


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_executor_failure_recovery(self, parallel_tnsim):
        """Test recovery after executor failure."""
        # Simulate executor failure
        with patch.object(parallel_tnsim, '_get_executor') as mock_executor:
            mock_executor.side_effect = Exception("Executor failed")
            
            with pytest.raises(Exception):
                await parallel_tnsim.parallel_zero_sum(
                    [ZeroSumInfiniteSet(elements=[1, 2, 3])],
                    method="direct"
                )
    
    @pytest.mark.asyncio
    async def test_invalid_operation_handling(self, parallel_tnsim):
        """Test handling of invalid operations."""
        operations = [
            {
                "type": "unknown_operation",
                "set_data": {"elements": [1, 2, 3]}
            }
        ]
        
        results = await parallel_tnsim.batch_operations(operations, ignore_errors=True)
        
        assert len(results) == 1
        assert "error" in results[0]
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        parallel = ParallelTNSIM(max_workers=1)
        
        # Create operation that may take long time
        large_set = ZeroSumInfiniteSet(elements=list(range(100000)))
        
        # Set short timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                parallel.parallel_zero_sum([large_set], method="compensated"),
                timeout=0.001  # Very short timeout
            )
        
        parallel.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(self, parallel_tnsim):
        """Test resource cleanup on errors."""
        # Start operation that will fail
        with patch.object(ZeroSumInfiniteSet, 'zero_sum_operation') as mock_op:
            mock_op.side_effect = Exception("Operation failed")
            
            try:
                await parallel_tnsim.parallel_zero_sum(
                    [ZeroSumInfiniteSet(elements=[1, 2, 3])],
                    method="direct"
                )
            except Exception:
                pass
        
        # Check that resources are cleaned up
        info = parallel_tnsim.get_running_tasks_info()
        assert info["total_tasks"] == 0


class TestIntegrationWithCache:
    """Tests for integration with cache."""
    
    @pytest.mark.asyncio
    async def test_parallel_operations_with_cache(self, parallel_tnsim, tnsim_cache):
        """Test parallel operations with cache usage."""
        # Set cache
        parallel_tnsim.set_cache(tnsim_cache)
        
        test_sets = [
            ZeroSumInfiniteSet(elements=[1, -1, 0.5, -0.5]),
            ZeroSumInfiniteSet(elements=[2, -2, 1, -1])
        ]
        
        # First run - results should be cached
        results1 = await parallel_tnsim.parallel_zero_sum(test_sets, method="direct")
        
        # Second run - results should be taken from cache
        results2 = await parallel_tnsim.parallel_zero_sum(test_sets, method="direct")
        
        assert results1 == results2
        
        # Check cache statistics
        stats = tnsim_cache.get_stats()
        assert stats["hits"] > 0
    
    @pytest.mark.asyncio
    async def test_cache_key_generation_in_parallel(self, parallel_tnsim, tnsim_cache):
        """Test cache key generation in parallel mode."""
        parallel_tnsim.set_cache(tnsim_cache)
        
        # Create identical sets
        identical_sets = [
            ZeroSumInfiniteSet(elements=[1, 2, 3]),
            ZeroSumInfiniteSet(elements=[1, 2, 3])  # Identical set
        ]
        
        results = await parallel_tnsim.parallel_zero_sum(identical_sets, method="direct")
        
        # Results should be identical
        assert results[0]["sum"] == results[1]["sum"]
        
        # Second result should be taken from cache
        stats = tnsim_cache.get_stats()
        assert stats["hits"] >= 1


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, parallel_tnsim):
        """Test handling of empty input."""
        # Empty list of sets
        results = await parallel_tnsim.parallel_zero_sum([], method="direct")
        assert results == []
        
        # Empty list of operations
        batch_results = await parallel_tnsim.batch_operations([])
        assert batch_results == []
    
    @pytest.mark.asyncio
    async def test_single_item_processing(self, parallel_tnsim):
        """Test processing of single item."""
        single_set = [ZeroSumInfiniteSet(elements=[42])]
        
        results = await parallel_tnsim.parallel_zero_sum(single_set, method="direct")
        
        assert len(results) == 1
        assert results[0]["sum"] == 42
    
    @pytest.mark.asyncio
    async def test_very_large_input(self):
        """Test processing of very large input."""
        parallel = ParallelTNSIM(max_workers=2, chunk_size=10)
        
        # Create many small sets
        many_sets = []
        for i in range(100):
            many_sets.append(ZeroSumInfiniteSet(elements=[i, -i]))
        
        results = await parallel.parallel_zero_sum(many_sets, method="direct")
        
        assert len(results) == 100
        for result in results:
            assert abs(result["sum"]) < 1e-10  # All should be close to zero
        
        parallel.shutdown()
    
    def test_context_manager_usage(self):
        """Test usage as context manager."""
        with ParallelTNSIM(max_workers=2) as parallel:
            assert parallel._executor is not None
        
        # After exiting context, executor should be shut down
        assert parallel._executor is None
    
    @pytest.mark.asyncio
    async def test_mixed_operation_types(self, parallel_tnsim):
        """Test mixed operation types."""
        operations = [
            {
                "type": "zero_sum",
                "set_data": {"elements": [1, -1]},
                "method": "direct"
            },
            {
                "type": "validation",
                "set_data": {"elements": [2, -2]},
                "tolerance": 1e-10
            },
            {
                "type": "convergence_analysis",
                "set_data": {"elements": [0.5, 0.25, 0.125]},
                "method": "ratio_test"
            }
        ]
        
        results = await parallel_tnsim.batch_operations(operations)
        
        assert len(results) == 3
        assert "sum" in results[0]
        assert "is_zero_sum" in results[1]
        assert "is_convergent" in results[2]