# -*- coding: utf-8 -*-
"""
Unit tests for TNSIM caching system.

This module contains comprehensive tests for the TNSIMCache class,
which provides caching functionality for zero-sum infinite set operations.

Test Categories:
- Cache initialization and configuration
- Basic cache operations (get, set, delete)
- Cache key generation and validation
- Cache statistics and monitoring
- Cache eviction policies (LRU, TTL)
- Memory management and optimization
- Persistence and serialization
- Thread safety and concurrent access
- Integration with TNSIM operations
- Performance benchmarks
- Error handling and recovery
- Edge cases and boundary conditions

Author: TNSIM Development Team
Version: 1.0.0
License: MIT
"""

import pytest
import time
import threading
import pickle
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from tnsim.cache.tnsim_cache import TNSIMCache
from tnsim.core.zero_sum_infinite_set import ZeroSumInfiniteSet


class TestTNSIMCacheInitialization:
    """Tests for TNSIMCache initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        cache = TNSIMCache()
        
        assert cache.max_size == 1000
        assert cache.ttl == 3600  # 1 hour
        assert cache.eviction_policy == "lru"
        assert cache.enable_persistence is False
        assert cache.persistence_file is None
        assert len(cache._cache) == 0
        assert cache._stats["hits"] == 0
        assert cache._stats["misses"] == 0
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        cache = TNSIMCache(
            max_size=500,
            ttl=1800,
            eviction_policy="ttl",
            enable_persistence=True,
            persistence_file="test_cache.pkl"
        )
        
        assert cache.max_size == 500
        assert cache.ttl == 1800
        assert cache.eviction_policy == "ttl"
        assert cache.enable_persistence is True
        assert cache.persistence_file == "test_cache.pkl"
    
    def test_invalid_max_size(self):
        """Test initialization with invalid max_size."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            TNSIMCache(max_size=0)
        
        with pytest.raises(ValueError, match="max_size must be positive"):
            TNSIMCache(max_size=-1)
    
    def test_invalid_ttl(self):
        """Test initialization with invalid TTL."""
        with pytest.raises(ValueError, match="ttl must be positive"):
            TNSIMCache(ttl=0)
        
        with pytest.raises(ValueError, match="ttl must be positive"):
            TNSIMCache(ttl=-1)
    
    def test_invalid_eviction_policy(self):
        """Test initialization with invalid eviction policy."""
        with pytest.raises(ValueError, match="eviction_policy must be"):
            TNSIMCache(eviction_policy="invalid")


class TestBasicCacheOperations:
    """Tests for basic cache operations."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        return TNSIMCache(max_size=10, ttl=60)
    
    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        key = "test_key"
        value = {"result": "test_value", "sum": 42}
        
        # Set value
        cache.set(key, value)
        
        # Get value
        retrieved_value = cache.get(key)
        
        assert retrieved_value == value
        assert cache._stats["hits"] == 1
        assert cache._stats["misses"] == 0
    
    def test_get_nonexistent_key(self, cache):
        """Test getting non-existent key."""
        result = cache.get("nonexistent_key")
        
        assert result is None
        assert cache._stats["hits"] == 0
        assert cache._stats["misses"] == 1
    
    def test_get_with_default(self, cache):
        """Test getting with default value."""
        default_value = {"default": True}
        result = cache.get("nonexistent_key", default_value)
        
        assert result == default_value
        assert cache._stats["misses"] == 1
    
    def test_delete(self, cache):
        """Test delete operation."""
        key = "test_key"
        value = {"test": "value"}
        
        # Set and verify
        cache.set(key, value)
        assert cache.get(key) == value
        
        # Delete and verify
        deleted = cache.delete(key)
        assert deleted is True
        assert cache.get(key) is None
        
        # Try to delete again
        deleted_again = cache.delete(key)
        assert deleted_again is False
    
    def test_contains(self, cache):
        """Test __contains__ method."""
        key = "test_key"
        value = {"test": "value"}
        
        assert key not in cache
        
        cache.set(key, value)
        assert key in cache
        
        cache.delete(key)
        assert key not in cache
    
    def test_len(self, cache):
        """Test __len__ method."""
        assert len(cache) == 0
        
        cache.set("key1", "value1")
        assert len(cache) == 1
        
        cache.set("key2", "value2")
        assert len(cache) == 2
        
        cache.delete("key1")
        assert len(cache) == 1
    
    def test_clear(self, cache):
        """Test clear operation."""
        # Add some items
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")
        
        assert len(cache) == 5
        
        # Clear cache
        cache.clear()
        
        assert len(cache) == 0
        assert cache._stats["hits"] == 0
        assert cache._stats["misses"] == 0


class TestCacheKeyGeneration:
    """Tests for cache key generation."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        return TNSIMCache()
    
    def test_generate_key_from_set(self, cache):
        """Test key generation from ZeroSumInfiniteSet."""
        test_set = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        
        key = cache.generate_key(test_set, method="direct")
        
        assert isinstance(key, str)
        assert len(key) > 0
        assert "direct" in key
    
    def test_generate_key_consistency(self, cache):
        """Test that identical inputs generate identical keys."""
        test_set1 = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        test_set2 = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        
        key1 = cache.generate_key(test_set1, method="direct")
        key2 = cache.generate_key(test_set2, method="direct")
        
        assert key1 == key2
    
    def test_generate_key_different_methods(self, cache):
        """Test that different methods generate different keys."""
        test_set = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        
        key1 = cache.generate_key(test_set, method="direct")
        key2 = cache.generate_key(test_set, method="compensated")
        
        assert key1 != key2
    
    def test_generate_key_different_sets(self, cache):
        """Test that different sets generate different keys."""
        test_set1 = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        test_set2 = ZeroSumInfiniteSet(elements=[4, 5, 6, -15])
        
        key1 = cache.generate_key(test_set1, method="direct")
        key2 = cache.generate_key(test_set2, method="direct")
        
        assert key1 != key2
    
    def test_generate_key_with_additional_params(self, cache):
        """Test key generation with additional parameters."""
        test_set = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        
        key1 = cache.generate_key(test_set, method="direct", tolerance=1e-10)
        key2 = cache.generate_key(test_set, method="direct", tolerance=1e-12)
        
        assert key1 != key2
    
    def test_generate_key_from_dict(self, cache):
        """Test key generation from dictionary data."""
        data = {
            "elements": [1, 2, 3, -6],
            "method": "direct",
            "tolerance": 1e-10
        }
        
        key = cache.generate_key_from_dict(data)
        
        assert isinstance(key, str)
        assert len(key) > 0


class TestCacheStatistics:
    """Tests for cache statistics."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        return TNSIMCache(max_size=5)
    
    def test_hit_statistics(self, cache):
        """Test hit statistics tracking."""
        key = "test_key"
        value = "test_value"
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        
        # Set value and get it (hit)
        cache.set(key, value)
        cache.get(key)
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 1.0
    
    def test_miss_statistics(self, cache):
        """Test miss statistics tracking."""
        # Try to get non-existent key (miss)
        cache.get("nonexistent_key")
        
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.0
    
    def test_mixed_statistics(self, cache):
        """Test mixed hit/miss statistics."""
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Mix of hits and misses
        cache.get("key1")  # hit
        cache.get("nonexistent1")  # miss
        cache.get("key2")  # hit
        cache.get("nonexistent2")  # miss
        cache.get("key1")  # hit
        
        stats = cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.6
    
    def test_reset_statistics(self, cache):
        """Test statistics reset."""
        # Generate some statistics
        cache.set("key", "value")
        cache.get("key")
        cache.get("nonexistent")
        
        # Verify stats exist
        stats = cache.get_stats()
        assert stats["hits"] > 0 or stats["misses"] > 0
        
        # Reset and verify
        cache.reset_stats()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
    
    def test_detailed_statistics(self, cache):
        """Test detailed statistics information."""
        # Add some data
        for i in range(3):
            cache.set(f"key_{i}", f"value_{i}")
        
        stats = cache.get_stats()
        
        assert "total_requests" in stats
        assert "cache_size" in stats
        assert "max_size" in stats
        assert "evictions" in stats
        assert stats["cache_size"] == 3
        assert stats["max_size"] == 5


class TestEvictionPolicies:
    """Tests for cache eviction policies."""
    
    def test_lru_eviction(self):
        """Test LRU (Least Recently Used) eviction policy."""
        cache = TNSIMCache(max_size=3, eviction_policy="lru")
        
        # Fill cache to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item - should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert "key1" in cache  # Recently accessed
        assert "key2" not in cache  # Should be evicted
        assert "key3" in cache
        assert "key4" in cache
    
    def test_ttl_eviction(self):
        """Test TTL (Time To Live) eviction policy."""
        cache = TNSIMCache(max_size=10, ttl=0.1, eviction_policy="ttl")
        
        # Set value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL to expire
        time.sleep(0.15)
        
        # Value should be expired
        assert cache.get("key1") is None
        assert "key1" not in cache
    
    def test_size_based_eviction(self):
        """Test size-based eviction when cache is full."""
        cache = TNSIMCache(max_size=2)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2
        
        # Add third item - should trigger eviction
        cache.set("key3", "value3")
        assert len(cache) == 2
        
        # One of the original keys should be evicted
        remaining_keys = [key for key in ["key1", "key2", "key3"] if key in cache]
        assert len(remaining_keys) == 2
        assert "key3" in remaining_keys  # Newly added should remain
    
    def test_eviction_statistics(self):
        """Test eviction statistics tracking."""
        cache = TNSIMCache(max_size=2)
        
        # Fill cache beyond capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should trigger eviction
        
        stats = cache.get_stats()
        assert stats["evictions"] >= 1


class TestMemoryManagement:
    """Tests for memory management."""
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        cache = TNSIMCache(max_size=100)
        
        # Add some large objects
        large_data = {"data": list(range(1000))}
        for i in range(10):
            cache.set(f"key_{i}", large_data.copy())
        
        stats = cache.get_stats()
        assert "memory_usage" in stats
        assert stats["memory_usage"] > 0
    
    def test_memory_cleanup_on_eviction(self):
        """Test memory cleanup when items are evicted."""
        cache = TNSIMCache(max_size=2)
        
        # Add items that will be evicted
        cache.set("key1", {"large_data": list(range(1000))})
        cache.set("key2", {"large_data": list(range(1000))})
        
        initial_size = len(cache)
        
        # Add more items to trigger eviction
        cache.set("key3", {"large_data": list(range(1000))})
        
        # Cache size should remain at max_size
        assert len(cache) == cache.max_size
        assert len(cache) <= initial_size + 1
    
    def test_garbage_collection_integration(self):
        """Test integration with garbage collection."""
        import gc
        
        cache = TNSIMCache(max_size=5)
        
        # Create objects that reference each other
        for i in range(10):
            obj = {"id": i, "data": list(range(100))}
            cache.set(f"key_{i}", obj)
        
        # Force garbage collection
        gc.collect()
        
        # Cache should still function correctly
        assert len(cache) <= cache.max_size
        stats = cache.get_stats()
        assert "memory_usage" in stats


class TestPersistence:
    """Tests for cache persistence."""
    
    def test_save_to_file(self):
        """Test saving cache to file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            cache = TNSIMCache(
                max_size=10,
                enable_persistence=True,
                persistence_file=tmp_file.name
            )
            
            # Add some data
            cache.set("key1", "value1")
            cache.set("key2", {"complex": "data", "number": 42})
            
            # Save to file
            cache.save_to_file()
            
            # Verify file exists and has content
            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    def test_load_from_file(self):
        """Test loading cache from file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            # Create and populate cache
            cache1 = TNSIMCache(
                max_size=10,
                enable_persistence=True,
                persistence_file=tmp_file.name
            )
            
            test_data = {
                "key1": "value1",
                "key2": {"complex": "data", "number": 42}
            }
            
            for key, value in test_data.items():
                cache1.set(key, value)
            
            cache1.save_to_file()
            
            # Create new cache and load data
            cache2 = TNSIMCache(
                max_size=10,
                enable_persistence=True,
                persistence_file=tmp_file.name
            )
            
            cache2.load_from_file()
            
            # Verify data was loaded correctly
            for key, expected_value in test_data.items():
                assert cache2.get(key) == expected_value
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    def test_auto_save_on_shutdown(self):
        """Test automatic saving on cache shutdown."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            cache = TNSIMCache(
                max_size=10,
                enable_persistence=True,
                persistence_file=tmp_file.name
            )
            
            # Add data
            cache.set("key1", "value1")
            
            # Shutdown cache (should auto-save)
            cache.shutdown()
            
            # Verify file was created
            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    def test_persistence_error_handling(self):
        """Test error handling in persistence operations."""
        # Test with invalid file path
        cache = TNSIMCache(
            enable_persistence=True,
            persistence_file="/invalid/path/cache.pkl"
        )
        
        cache.set("key1", "value1")
        
        # Should not raise exception, but should handle error gracefully
        try:
            cache.save_to_file()
        except Exception as e:
            # Should log error but not crash
            assert "permission" in str(e).lower() or "no such file" in str(e).lower()


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_access(self):
        """Test concurrent access from multiple threads."""
        cache = TNSIMCache(max_size=100)
        results = {}
        errors = []
        
        def worker(thread_id):
            try:
                # Each thread performs multiple operations
                for i in range(50):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    
                    cache.set(key, value)
                    retrieved = cache.get(key)
                    
                    if retrieved != value:
                        errors.append(f"Thread {thread_id}: Expected {value}, got {retrieved}")
                
                results[thread_id] = "completed"
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
    
    def test_concurrent_eviction(self):
        """Test concurrent access during eviction."""
        cache = TNSIMCache(max_size=10)
        errors = []
        
        def writer(thread_id):
            try:
                for i in range(100):
                    key = f"writer_{thread_id}_key_{i}"
                    cache.set(key, f"value_{i}")
            except Exception as e:
                errors.append(f"Writer {thread_id}: {str(e)}")
        
        def reader(thread_id):
            try:
                for i in range(100):
                    # Try to read various keys
                    for writer_id in range(2):
                        key = f"writer_{writer_id}_key_{i % 50}"
                        cache.get(key)  # May hit or miss
            except Exception as e:
                errors.append(f"Reader {thread_id}: {str(e)}")
        
        # Start writer and reader threads
        threads = []
        
        # Start writers
        for i in range(2):
            thread = threading.Thread(target=writer, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Start readers
        for i in range(3):
            thread = threading.Thread(target=reader, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check for errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Cache should be in valid state
        assert len(cache) <= cache.max_size
    
    def test_statistics_thread_safety(self):
        """Test thread safety of statistics tracking."""
        cache = TNSIMCache(max_size=50)
        
        def worker(thread_id):
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                cache.set(key, f"value_{i}")
                cache.get(key)  # Hit
                cache.get(f"nonexistent_{i}")  # Miss
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check statistics consistency
        stats = cache.get_stats()
        expected_total = 3 * 100 * 2  # 3 threads * 100 operations * 2 gets each
        actual_total = stats["hits"] + stats["misses"]
        
        assert actual_total == expected_total
        assert stats["hits"] == 3 * 100  # All set keys should be hits
        assert stats["misses"] == 3 * 100  # All nonexistent keys should be misses


class TestIntegrationWithTNSIM:
    """Tests for integration with TNSIM operations."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        return TNSIMCache(max_size=50)
    
    def test_cache_zero_sum_operations(self, cache):
        """Test caching of zero-sum operations."""
        test_set = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        
        # Generate cache key
        key = cache.generate_key(test_set, method="direct")
        
        # Simulate operation result
        result = {"sum": 0.0, "method": "direct", "iterations": 1}
        
        # Cache the result
        cache.set(key, result)
        
        # Retrieve from cache
        cached_result = cache.get(key)
        
        assert cached_result == result
        assert cache._stats["hits"] == 1
    
    def test_cache_validation_operations(self, cache):
        """Test caching of validation operations."""
        test_set = ZeroSumInfiniteSet(elements=[1, -1, 2, -2])
        
        # Generate cache key for validation
        key = cache.generate_key(test_set, method="validation", tolerance=1e-10)
        
        # Simulate validation result
        result = {
            "is_zero_sum": True,
            "total_sum": 0.0,
            "tolerance": 1e-10,
            "method": "validation"
        }
        
        # Cache and retrieve
        cache.set(key, result)
        cached_result = cache.get(key)
        
        assert cached_result == result
    
    def test_cache_convergence_analysis(self, cache):
        """Test caching of convergence analysis."""
        test_set = ZeroSumInfiniteSet(elements=[1, 0.5, 0.25, 0.125])
        
        # Generate cache key
        key = cache.generate_key(test_set, method="convergence", analysis_type="ratio_test")
        
        # Simulate convergence analysis result
        result = {
            "is_convergent": True,
            "convergence_type": "geometric",
            "ratio": 0.5,
            "method": "ratio_test"
        }
        
        # Cache and retrieve
        cache.set(key, result)
        cached_result = cache.get(key)
        
        assert cached_result == result
    
    def test_cache_performance_with_real_operations(self, cache):
        """Test cache performance with real TNSIM operations."""
        # Create multiple sets
        test_sets = [
            ZeroSumInfiniteSet(elements=[1, -1, 0.5, -0.5]),
            ZeroSumInfiniteSet(elements=[2, -2, 1, -1]),
            ZeroSumInfiniteSet(elements=[3, -3, 1.5, -1.5])
        ]
        
        # Simulate operations and caching
        for i, test_set in enumerate(test_sets):
            key = cache.generate_key(test_set, method="direct")
            
            # First access - should be a miss
            result = cache.get(key)
            assert result is None
            
            # Simulate operation and cache result
            operation_result = {
                "sum": 0.0,
                "method": "direct",
                "set_id": i
            }
            cache.set(key, operation_result)
            
            # Second access - should be a hit
            cached_result = cache.get(key)
            assert cached_result == operation_result
        
        # Check statistics
        stats = cache.get_stats()
        assert stats["hits"] == 3  # Second access to each set
        assert stats["misses"] == 3  # First access to each set
        assert stats["hit_rate"] == 0.5


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.performance
    def test_large_cache_performance(self):
        """Test performance with large cache."""
        cache = TNSIMCache(max_size=10000)
        
        # Measure insertion time
        start_time = time.time()
        for i in range(5000):
            cache.set(f"key_{i}", {"data": f"value_{i}", "number": i})
        insertion_time = time.time() - start_time
        
        # Measure retrieval time
        start_time = time.time()
        for i in range(5000):
            cache.get(f"key_{i}")
        retrieval_time = time.time() - start_time
        
        # Performance should be reasonable
        assert insertion_time < 5.0  # Should insert 5000 items in less than 5 seconds
        assert retrieval_time < 2.0  # Should retrieve 5000 items in less than 2 seconds
        
        # Verify cache state
        assert len(cache) == 5000
        stats = cache.get_stats()
        assert stats["hits"] == 5000
    
    @pytest.mark.performance
    def test_memory_efficiency(self):
        """Test memory efficiency of cache."""
        import sys
        
        cache = TNSIMCache(max_size=1000)
        
        # Add items and measure memory usage
        initial_size = sys.getsizeof(cache._cache)
        
        for i in range(500):
            cache.set(f"key_{i}", {"data": list(range(100))})
        
        final_size = sys.getsizeof(cache._cache)
        
        # Memory usage should be reasonable
        memory_per_item = (final_size - initial_size) / 500
        assert memory_per_item < 10000  # Less than 10KB per item overhead
    
    @pytest.mark.performance
    def test_eviction_performance(self):
        """Test performance of eviction operations."""
        cache = TNSIMCache(max_size=100, eviction_policy="lru")
        
        # Fill cache beyond capacity multiple times
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        eviction_time = time.time() - start_time
        
        # Eviction should be efficient
        assert eviction_time < 2.0  # Should handle 1000 insertions with evictions in less than 2 seconds
        assert len(cache) == 100  # Should maintain max size
        
        # Check eviction statistics
        stats = cache.get_stats()
        assert stats["evictions"] == 900  # Should have evicted 900 items


class TestErrorHandling:
    """Tests for error handling and recovery."""
    
    def test_invalid_key_types(self):
        """Test handling of invalid key types."""
        cache = TNSIMCache()
        
        # Test with various invalid key types
        invalid_keys = [None, [], {}, set(), object()]
        
        for invalid_key in invalid_keys:
            with pytest.raises((TypeError, ValueError)):
                cache.set(invalid_key, "value")
            
            with pytest.raises((TypeError, ValueError)):
                cache.get(invalid_key)
    
    def test_serialization_errors(self):
        """Test handling of serialization errors."""
        cache = TNSIMCache(enable_persistence=True)
        
        # Try to cache non-serializable object
        non_serializable = lambda x: x  # Function objects can't be pickled
        
        # Should handle gracefully
        try:
            cache.set("key", non_serializable)
            cache.save_to_file()
        except Exception as e:
            # Should be a serialization-related error
            assert "pickle" in str(e).lower() or "serialize" in str(e).lower()
    
    def test_corruption_recovery(self):
        """Test recovery from corrupted cache data."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            # Write corrupted data to file
            tmp_file.write(b"corrupted data")
            tmp_file.flush()
            
            cache = TNSIMCache(
                enable_persistence=True,
                persistence_file=tmp_file.name
            )
            
            # Should handle corrupted file gracefully
            try:
                cache.load_from_file()
            except Exception:
                # Should recover by starting with empty cache
                pass
            
            # Cache should still be functional
            cache.set("key", "value")
            assert cache.get("key") == "value"
            
            # Cleanup
            os.unlink(tmp_file.name)
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure situations."""
        cache = TNSIMCache(max_size=10)
        
        # Try to add very large objects
        large_object = {"data": list(range(100000))}
        
        # Should handle without crashing
        for i in range(20):
            cache.set(f"key_{i}", large_object.copy())
        
        # Cache should maintain size limits
        assert len(cache) <= cache.max_size
        
        # Should still be functional
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_max_size_cache(self):
        """Test cache with zero max size."""
        with pytest.raises(ValueError):
            TNSIMCache(max_size=0)
    
    def test_single_item_cache(self):
        """Test cache with max size of 1."""
        cache = TNSIMCache(max_size=1)
        
        # Add first item
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert len(cache) == 1
        
        # Add second item - should evict first
        cache.set("key2", "value2")
        assert cache.get("key2") == "value2"
        assert cache.get("key1") is None
        assert len(cache) == 1
    
    def test_very_short_ttl(self):
        """Test cache with very short TTL."""
        cache = TNSIMCache(ttl=0.001, eviction_policy="ttl")  # 1ms TTL
        
        cache.set("key", "value")
        
        # Should expire almost immediately
        time.sleep(0.01)
        assert cache.get("key") is None
    
    def test_empty_key_handling(self):
        """Test handling of empty keys."""
        cache = TNSIMCache()
        
        # Empty string key should work
        cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"
        
        # Whitespace-only key should work
        cache.set("   ", "whitespace_key_value")
        assert cache.get("   ") == "whitespace_key_value"
    
    def test_none_value_handling(self):
        """Test handling of None values."""
        cache = TNSIMCache()
        
        # Should be able to cache None values
        cache.set("none_key", None)
        assert cache.get("none_key") is None
        assert "none_key" in cache  # Should distinguish from missing key
        
        # Should distinguish None value from missing key
        assert cache.get("missing_key") is None
        assert "missing_key" not in cache
    
    def test_duplicate_key_updates(self):
        """Test updating existing keys."""
        cache = TNSIMCache()
        
        # Set initial value
        cache.set("key", "value1")
        assert cache.get("key") == "value1"
        assert len(cache) == 1
        
        # Update with new value
        cache.set("key", "value2")
        assert cache.get("key") == "value2"
        assert len(cache) == 1  # Size should not increase
    
    def test_concurrent_key_generation(self):
        """Test concurrent cache key generation."""
        cache = TNSIMCache()
        
        # Create identical sets in different threads
        test_set = ZeroSumInfiniteSet(elements=[1, 2, 3, -6])
        keys = []
        
        def generate_key():
            key = cache.generate_key(test_set, method="direct")
            keys.append(key)
        
        # Generate keys concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=generate_key)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All keys should be identical
        assert len(set(keys)) == 1  # All keys should be the same
        assert len(keys) == 5