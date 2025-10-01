"""Configuration file for pytest tests.

This module contains fixtures and configuration for running tests
for the TNSIM (Theory of Numerical Simulation of Infinite Mathematical structures) project.
"""

import pytest
import numpy as np
import torch
from typing import List, Dict, Any, Generator
from decimal import Decimal, getcontext

from tnsim.core.sets import ZeroSumInfiniteSet
from tnsim.core.series import InfiniteSeries
from tnsim.core.compensation import NumericalCompensator


# Set high precision for Decimal calculations
getcontext().prec = 50


@pytest.fixture
def sample_harmonic_set() -> ZeroSumInfiniteSet:
    """Create a sample harmonic series zero-sum set.
    
    Returns:
        ZeroSumInfiniteSet: Harmonic series with alternating signs
    """
    def harmonic_generator(n: int) -> float:
        """Generate harmonic series terms with alternating signs."""
        return (-1) ** (n + 1) / n
    
    return ZeroSumInfiniteSet(
        generator=harmonic_generator,
        name="alternating_harmonic",
        convergence_test="alternating_series"
    )


@pytest.fixture
def sample_alternating_set() -> ZeroSumInfiniteSet:
    """Create a sample alternating series zero-sum set.
    
    Returns:
        ZeroSumInfiniteSet: Simple alternating series
    """
    def alternating_generator(n: int) -> float:
        """Generate simple alternating series."""
        return (-1) ** n / (2 ** n)
    
    return ZeroSumInfiniteSet(
        generator=alternating_generator,
        name="alternating_geometric",
        convergence_test="ratio_test"
    )


@pytest.fixture
def sample_geometric_set() -> ZeroSumInfiniteSet:
    """Create a sample geometric series zero-sum set.
    
    Returns:
        ZeroSumInfiniteSet: Geometric series with compensation
    """
    def geometric_generator(n: int) -> float:
        """Generate geometric series with compensation term."""
        if n == 0:
            return 1.0
        elif n % 2 == 1:
            return 1.0 / (2 ** n)
        else:
            return -1.0 / (2 ** (n - 1))
    
    return ZeroSumInfiniteSet(
        generator=geometric_generator,
        name="compensated_geometric",
        convergence_test="ratio_test"
    )


@pytest.fixture
def sample_series_data() -> Dict[str, List[float]]:
    """Create sample series data for testing.
    
    Returns:
        Dict[str, List[float]]: Dictionary with different series types
    """
    return {
        "harmonic": [1/n for n in range(1, 101)],
        "alternating": [(-1)**n / n for n in range(1, 101)],
        "geometric": [1/(2**n) for n in range(100)],
        "fibonacci_ratios": [1.0, 1.0] + [0.0] * 98  # Will be filled by test
    }


@pytest.fixture
def numerical_compensator() -> NumericalCompensator:
    """Create a numerical compensator instance.
    
    Returns:
        NumericalCompensator: Configured compensator for testing
    """
    return NumericalCompensator(
        precision=1e-15,
        max_iterations=1000,
        algorithm="kahan_babuska"
    )


@pytest.fixture
def torch_device() -> torch.device:
    """Get the appropriate torch device for testing.
    
    Returns:
        torch.device: CUDA if available, otherwise CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_tensor_data(torch_device) -> Dict[str, torch.Tensor]:
    """Create sample tensor data for neural network testing.
    
    Args:
        torch_device: Device to create tensors on
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary with sample tensors
    """
    torch.manual_seed(42)  # For reproducible tests
    
    return {
        "small_batch": torch.randn(2, 10, 64, device=torch_device),
        "medium_batch": torch.randn(8, 50, 128, device=torch_device),
        "large_batch": torch.randn(16, 100, 256, device=torch_device),
        "sequence_data": torch.randn(4, 20, 512, device=torch_device),
        "attention_mask": torch.tril(torch.ones(20, 20, device=torch_device)).bool()
    }


@pytest.fixture
def precision_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for precision testing.
    
    Returns:
        List[Dict[str, Any]]: List of test cases with expected results
    """
    return [
        {
            "name": "small_numbers",
            "values": [1e-15, 1e-16, 1e-17],
            "expected_sum": 1.11e-15,
            "tolerance": 1e-17
        },
        {
            "name": "large_numbers",
            "values": [1e15, -1e15, 1.0],
            "expected_sum": 1.0,
            "tolerance": 1e-10
        },
        {
            "name": "mixed_scale",
            "values": [1.0, 1e-10, -1.0, 1e-11],
            "expected_sum": 1.1e-10,
            "tolerance": 1e-12
        }
    ]


@pytest.fixture
def convergence_test_data() -> Dict[str, Dict[str, Any]]:
    """Create data for convergence testing.
    
    Returns:
        Dict[str, Dict[str, Any]]: Test data for different convergence scenarios
    """
    return {
        "fast_convergence": {
            "series": [1/(n**2) for n in range(1, 1001)],
            "expected_convergence": True,
            "max_terms": 100
        },
        "slow_convergence": {
            "series": [1/n for n in range(1, 10001)],
            "expected_convergence": False,  # Harmonic series diverges
            "max_terms": 1000
        },
        "alternating_convergence": {
            "series": [(-1)**(n+1)/n for n in range(1, 1001)],
            "expected_convergence": True,
            "max_terms": 500
        }
    }


@pytest.fixture
def performance_benchmarks() -> Dict[str, Dict[str, Any]]:
    """Create performance benchmark data.
    
    Returns:
        Dict[str, Dict[str, Any]]: Benchmark configurations
    """
    return {
        "small_scale": {
            "series_length": 1000,
            "max_time_seconds": 0.1,
            "memory_limit_mb": 10
        },
        "medium_scale": {
            "series_length": 10000,
            "max_time_seconds": 1.0,
            "memory_limit_mb": 50
        },
        "large_scale": {
            "series_length": 100000,
            "max_time_seconds": 10.0,
            "memory_limit_mb": 200
        }
    }


@pytest.fixture(scope="session")
def test_configuration() -> Dict[str, Any]:
    """Global test configuration.
    
    Returns:
        Dict[str, Any]: Configuration parameters for all tests
    """
    return {
        "default_precision": 1e-12,
        "max_test_iterations": 10000,
        "timeout_seconds": 30,
        "enable_gpu_tests": torch.cuda.is_available(),
        "random_seed": 42,
        "numerical_tolerance": {
            "float32": 1e-6,
            "float64": 1e-12,
            "decimal": 1e-15
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(test_configuration):
    """Automatically set up test environment for each test.
    
    Args:
        test_configuration: Global test configuration
    """
    # Set random seeds for reproducibility
    np.random.seed(test_configuration["random_seed"])
    torch.manual_seed(test_configuration["random_seed"])
    
    # Set default tensor type
    torch.set_default_dtype(torch.float64)
    
    # Configure decimal precision
    getcontext().prec = 50
    
    yield  # Run the test
    
    # Cleanup after test
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture
def mock_balansis_available():
    """Mock Balansis availability for testing.
    
    Yields:
        bool: Whether Balansis should be considered available
    """
    # This fixture can be used to test both scenarios:
    # when Balansis is available and when it's not
    yield True


@pytest.fixture
def error_injection_scenarios() -> List[Dict[str, Any]]:
    """Create scenarios for error injection testing.
    
    Returns:
        List[Dict[str, Any]]: Error scenarios for robustness testing
    """
    return [
        {
            "name": "nan_values",
            "data": [1.0, float('nan'), 2.0],
            "expected_error": ValueError
        },
        {
            "name": "infinite_values",
            "data": [1.0, float('inf'), -float('inf')],
            "expected_error": OverflowError
        },
        {
            "name": "empty_data",
            "data": [],
            "expected_error": None  # Should handle gracefully
        },
        {
            "name": "wrong_type",
            "data": ["not", "a", "number"],
            "expected_error": TypeError
        }
    ]


@pytest.fixture
def memory_profiler():
    """Memory profiler fixture for performance testing.
    
    Yields:
        callable: Function to get current memory usage
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        return process.memory_info().rss / 1024 / 1024
    
    yield get_memory_usage


@pytest.fixture
def time_profiler():
    """Time profiler fixture for performance testing.
    
    Yields:
        callable: Context manager for timing operations
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        """Context manager for timing operations."""
        start_time = time.time()
        yield lambda: time.time() - start_time
        
    yield timer


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m "not slow"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m "not gpu"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in str(item.fspath) or "integration" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)