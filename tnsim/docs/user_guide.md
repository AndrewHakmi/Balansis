# TNSIM User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Basic Concepts](#basic-concepts)
5. [Working with Zero-Sum Sets](#working-with-zero-sum-sets)
6. [Compensation Algorithms](#compensation-algorithms)
7. [Convergence Analysis](#convergence-analysis)
8. [Parallel Processing](#parallel-processing)
9. [Balansis Integration](#balansis-integration)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Usage](#advanced-usage)
13. [Best Practices](#best-practices)

## Introduction

TNSIM (Theory of Null-Sum Infinite Multitudes) is a Python library for working with zero-sum infinite sets and high-precision numerical computations. The library provides tools for:

- Creating and manipulating zero-sum sets
- High-precision summation with error compensation
- Convergence analysis of infinite series
- Parallel processing of large datasets
- Integration with neural networks through Balansis

### Key Features

- **High Precision**: Advanced algorithms for error compensation in floating-point arithmetic
- **Scalability**: Parallel processing support for large datasets
- **Flexibility**: Multiple compensation methods and convergence tests
- **Integration**: Seamless integration with NumPy, Pandas, and PyTorch
- **Performance**: Optimized algorithms with caching support

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0 (optional)
- PyTorch >= 1.9.0 (optional, for neural network integration)
- Balansis >= 1.0.0 (optional, for high-precision computations)

### Installation via pip

```bash
pip install tnsim
```

### Installation from source

```bash
git clone https://github.com/your-repo/tnsim.git
cd tnsim
pip install -e .
```

### Development installation

```bash
git clone https://github.com/your-repo/tnsim.git
cd tnsim
pip install -e ".[dev]"
```

### Verification

```python
import tnsim
print(tnsim.__version__)

# Run basic test
from tnsim.core.zero_sum_set import ZeroSumInfiniteSet
zs_set = ZeroSumInfiniteSet([1.0, -1.0])
print(f"Is zero-sum: {zs_set.is_zero_sum()}")
```

## Quick Start

### Creating Your First Zero-Sum Set

```python
from tnsim.core.zero_sum_set import ZeroSumInfiniteSet

# Create a simple zero-sum set
zs_set = ZeroSumInfiniteSet([1.0, 2.0, -3.0])

# Check if it's zero-sum
print(f"Is zero-sum: {zs_set.is_zero_sum()}")
print(f"Sum: {zs_set.sum()}")

# Get statistics
stats = zs_set.statistics()
print(f"Mean: {stats['mean']}, Std: {stats['std']}")
```

### Performing Compensated Operations

```python
# Perform compensated summation
result = zs_set.zero_sum_operation(method='compensated')
print(f"Compensated sum: {result['sum']}")
print(f"Error estimate: {result['error']}")
```

### Working with Large Sets

```python
# Create a large harmonic series
harmonic = ZeroSumInfiniteSet.from_function(
    lambda i: 1.0 / i, 
    n=10000, 
    start=1
)

# Use adaptive compensation
result = harmonic.zero_sum_operation(method='adaptive')
print(f"Harmonic series sum: {result['sum']}")
```

## Basic Concepts

### Zero-Sum Sets

A zero-sum set is a collection of numbers whose sum equals zero (within a specified tolerance). In TNSIM, zero-sum sets are represented by the `ZeroSumInfiniteSet` class.

```python
# Perfect zero-sum set
perfect_set = ZeroSumInfiniteSet([1.0, -1.0])
print(perfect_set.is_zero_sum())  # True

# Approximate zero-sum set
approx_set = ZeroSumInfiniteSet([1.0, -0.9999999999999999])
print(approx_set.is_zero_sum())  # True (within tolerance)
```

### Tolerance

Tolerance determines how close to zero the sum must be to consider the set as zero-sum:

```python
# Set with custom tolerance
zs_set = ZeroSumInfiniteSet([1.0, -0.999], tolerance=1e-2)
print(zs_set.is_zero_sum())  # True

# Check with different tolerance
print(zs_set.is_zero_sum(tolerance=1e-6))  # False
```

### Floating-Point Errors

TNSIM addresses floating-point arithmetic errors that can accumulate in large computations:

```python
# Demonstrate floating-point error
numbers = [0.1] * 10
direct_sum = sum(numbers)
print(f"Direct sum: {direct_sum}")  # Not exactly 1.0

# Using TNSIM compensation
zs_set = ZeroSumInfiniteSet(numbers)
result = zs_set.zero_sum_operation(method='compensated')
print(f"Compensated sum: {result['sum']}")  # More accurate
```

## Working with Zero-Sum Sets

### Creating Sets

#### From Lists

```python
# From Python list
zs_set = ZeroSumInfiniteSet([1.0, 2.0, -3.0])

# From generator
zs_set = ZeroSumInfiniteSet(x for x in range(-5, 6))
```

#### From Functions

```python
# Harmonic series
harmonic = ZeroSumInfiniteSet.from_function(
    lambda i: 1.0 / i, 
    n=1000
)

# Alternating series
alternating = ZeroSumInfiniteSet.from_function(
    lambda i: (-1)**i / i, 
    n=1000, 
    start=1
)

# Custom function
custom = ZeroSumInfiniteSet.from_function(
    lambda i: 1.0 / (i**2), 
    n=1000
)
```

#### From NumPy Arrays

```python
import numpy as np

# From NumPy array
array = np.random.randn(1000)
zs_set = ZeroSumInfiniteSet.from_numpy(array)

# From specific NumPy functions
linspace_set = ZeroSumInfiniteSet.from_numpy(
    np.linspace(-1, 1, 1000)
)
```

#### From Pandas Series

```python
import pandas as pd

# From Pandas series
series = pd.Series([1.0, 2.0, -3.0])
zs_set = ZeroSumInfiniteSet.from_pandas(series)
```

### Modifying Sets

#### Adding Elements

```python
zs_set = ZeroSumInfiniteSet([1.0, -1.0])

# Add single element
zs_set.add_element(2.0)

# Add multiple elements
zs_set.add_elements([3.0, -5.0])

print(f"Elements: {zs_set.elements}")
print(f"Sum: {zs_set.sum()}")
```

#### Removing Elements

```python
# Remove specific element
removed = zs_set.remove_element(2.0)
print(f"Element removed: {removed}")
```

### Set Operations

#### Union

```python
set1 = ZeroSumInfiniteSet([1.0, 2.0])
set2 = ZeroSumInfiniteSet([3.0, -6.0])

union_set = set1.union(set2)
print(f"Union elements: {union_set.elements}")
```

#### Intersection

```python
set1 = ZeroSumInfiniteSet([1.0, 2.0, 3.0])
set2 = ZeroSumInfiniteSet([2.0, 3.0, 4.0])

intersection_set = set1.intersection(set2)
print(f"Intersection: {intersection_set.elements}")
```

#### Difference

```python
difference_set = set1.difference(set2)
print(f"Difference: {difference_set.elements}")
```

### Statistical Analysis

```python
zs_set = ZeroSumInfiniteSet(np.random.randn(1000))

# Basic statistics
print(f"Size: {zs_set.size()}")
print(f"Mean: {zs_set.mean()}")
print(f"Std: {zs_set.std()}")
print(f"Min: {zs_set.min()}")
print(f"Max: {zs_set.max()}")
print(f"Median: {zs_set.median()}")

# Advanced statistics
print(f"Skewness: {zs_set.skewness()}")
print(f"Kurtosis: {zs_set.kurtosis()}")
print(f"Entropy: {zs_set.entropy()}")

# Complete statistics
stats = zs_set.statistics()
for key, value in stats.items():
    print(f"{key}: {value}")
```

## Compensation Algorithms

TNSIM provides several algorithms for compensating floating-point errors:

### Direct Method

Simple summation without compensation:

```python
result = zs_set.zero_sum_operation(method='direct')
print(f"Direct sum: {result['sum']}")
```

### Kahan Summation

Kahan algorithm for error compensation:

```python
result = zs_set.zero_sum_operation(method='compensated')
print(f"Kahan sum: {result['sum']}")
print(f"Error: {result['error']}")
```

### Iterative Compensation

Multiple passes for higher precision:

```python
result = zs_set.zero_sum_operation(
    method='iterative',
    max_iterations=10,
    tolerance=1e-16
)
print(f"Iterative sum: {result['sum']}")
print(f"Iterations: {result['iterations']}")
```

### Adaptive Compensation

Adaptive algorithm that adjusts parameters:

```python
result = zs_set.zero_sum_operation(
    method='adaptive',
    target_precision=1e-15
)
print(f"Adaptive sum: {result['sum']}")
print(f"Final precision: {result['precision']}")
```

### Stabilized Method

Stabilized summation for extreme cases:

```python
result = zs_set.zero_sum_operation(method='stabilized')
print(f"Stabilized sum: {result['sum']}")
```

### Comparing Methods

```python
from tnsim.utils.benchmarks import MethodBenchmark

# Create test data
test_data = [0.1] * 1000 + [-0.1] * 1000
zs_set = ZeroSumInfiniteSet(test_data)

# Compare methods
methods = ['direct', 'compensated', 'iterative', 'adaptive']
for method in methods:
    result = zs_set.zero_sum_operation(method=method)
    print(f"{method}: {result['sum']:.2e}, time: {result.get('time', 0):.6f}s")
```

## Convergence Analysis

TNSIM provides tools for analyzing the convergence of infinite series:

### Setting Up Convergence Analysis

```python
from tnsim.core.convergence import ConvergenceAnalyzer

analyzer = ConvergenceAnalyzer(tolerance=1e-12)
```

### Ratio Test

```python
# Geometric series (convergent)
geometric = [0.5**n for n in range(100)]
result = analyzer.ratio_test(geometric)
print(f"Convergent: {result['convergent']}")
print(f"Limit: {result['limit']}")

# Factorial series (divergent)
factorial = [1.0]
for i in range(1, 20):
    factorial.append(factorial[-1] * i)

result = analyzer.ratio_test(factorial)
print(f"Convergent: {result['convergent']}")
```

### Root Test

```python
# Test with root test
result = analyzer.root_test(geometric)
print(f"Root test - Convergent: {result['convergent']}")
print(f"Root test - Limit: {result['limit']}")
```

### Integral Test

```python
# Define function for integral test
def harmonic_function(x):
    return 1.0 / x

# Perform integral test
result = analyzer.integral_test(harmonic_function, 1, float('inf'))
print(f"Integral test - Convergent: {result['convergent']}")
```

### Alternating Series Test

```python
# Test alternating harmonic series
alternating_harmonic = [(-1)**(n+1) / n for n in range(1, 1001)]
result = analyzer.alternating_series_test(alternating_harmonic)
print(f"Alternating series - Convergent: {result['convergent']}")
print(f"Error bound: {result['error_bound']:.6f}")
```

### Comparison Test

```python
# Compare with known convergent series
known_convergent = [1.0 / (n**2) for n in range(1, 101)]
unknown_series = [1.0 / (n**1.5) for n in range(1, 101)]

result = analyzer.comparison_test(unknown_series, known_convergent)
print(f"Comparison test result: {result}")
```

### Partial Sums Analysis

```python
# Analyze behavior of partial sums
analysis = analyzer.partial_sums_analysis(geometric)
print(f"Trend: {analysis['trend']}")
print(f"Growth rate: {analysis['growth_rate']}")
print(f"Estimated limit: {analysis['estimated_limit']}")
```

## Parallel Processing

For large datasets, TNSIM supports parallel processing:

### Setting Up Parallel Processing

```python
from tnsim.core.parallel import ParallelTNSIM

# Using threading (good for I/O bound tasks)
parallel = ParallelTNSIM(num_workers=4, backend='threading')

# Using multiprocessing (good for CPU bound tasks)
parallel = ParallelTNSIM(num_workers=4, backend='multiprocessing')
```

### Parallel Zero-Sum Operations

```python
# Create large set
large_set = ZeroSumInfiniteSet(range(-100000, 100001))

# Parallel processing
result = parallel.parallel_zero_sum(
    large_set, 
    method='compensated',
    chunk_size=10000
)

print(f"Result: {result['sum']}")
print(f"Execution time: {result['execution_time']}s")
print(f"Speedup: {result['speedup']}x")
```

### Batch Processing

```python
# Create multiple sets
sets = [
    ZeroSumInfiniteSet(np.random.randn(1000)) 
    for _ in range(10)
]

# Process in parallel
results = parallel.batch_process(sets, method='adaptive')

for i, result in enumerate(results):
    print(f"Set {i}: sum = {result['sum']:.6e}")
```

### Performance Monitoring

```python
# Get performance statistics
stats = parallel.get_performance_stats()
print(f"Throughput: {stats['throughput']} tasks/sec")
print(f"Average task time: {stats['avg_task_time']}s")
print(f"Total processed: {stats['total_processed']}")

# Shutdown parallel processor
parallel.shutdown()
```

## Balansis Integration

TNSIM integrates with the Balansis library for ultra-high precision computations:

### Setting Up Balansis Integration

```python
from tnsim.integrations.balansis_integration import BalansisCompensator

# Create compensator with different precision levels
compensator = BalansisCompensator(
    precision='ultra',  # 'low', 'medium', 'high', 'ultra'
    algorithm='auto'    # 'auto', 'kahan', 'neumaier', 'klein'
)
```

### High-Precision Series Compensation

```python
# Create problematic series
problematic_series = [1e-16 + i * 1e-17 for i in range(1000)]

# Compensate using Balansis
compensated_sum = compensator.compensate_series(problematic_series)
print(f"Balansis compensated sum: {compensated_sum}")
```

### Zero-Sum Set Compensation

```python
# Create zero-sum set with small errors
zs_set = ZeroSumInfiniteSet([1.0, -1.0 + 1e-15])

# Compensate using Balansis
compensated_set = compensator.compensate_zero_sum_set(zs_set)
print(f"Original sum: {zs_set.sum()}")
print(f"Compensated sum: {compensated_set.sum()}")
```

### Quality Assessment

```python
# Assess compensation quality
original_sum = sum(problematic_series)
compensated_sum = compensator.compensate_series(problematic_series)

quality = compensator.assess_compensation_quality(
    original_sum, 
    compensated_sum
)

print(f"Improvement factor: {quality['improvement_factor']}")
print(f"Error reduction: {quality['error_reduction']:.2%}")
```

### Neural Network Integration

```python
import torch
from tnsim.integrations.balansis_integration import ZeroSumAttention

# Create zero-sum attention layer
attention = ZeroSumAttention(
    d_model=512,
    num_heads=8,
    compensation_method='adaptive',
    dropout=0.1
)

# Use in neural network
query = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
key = torch.randn(32, 100, 512)
value = torch.randn(32, 100, 512)

output, weights = attention(query, key, value)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# Check compensation statistics
stats = attention.get_compensation_stats()
print(f"Compensation applied: {stats['compensation_applied']}")
```

## Performance Optimization

### Caching

TNSIM provides caching to speed up repeated computations:

```python
from tnsim.core.cache import TNSIMCache

# Create cache
cache = TNSIMCache(
    max_size=1000,
    ttl=3600,  # 1 hour
    eviction_policy='lru'
)

# Use cache with computations
key = "harmonic_1000"
result = cache.get(key)

if result is None:
    # Compute if not in cache
    harmonic = ZeroSumInfiniteSet.from_function(
        lambda i: 1.0 / i, 
        n=1000
    )
    result = harmonic.zero_sum_operation(method='compensated')
    cache.put(key, result)

print(f"Cached result: {result['sum']}")

# Cache statistics
stats = cache.get_stats()
print(f"Hit ratio: {stats['hit_ratio']:.2%}")
print(f"Cache size: {stats['size']}")
```

### Profiling

```python
from tnsim.utils.profiler import TNSIMProfiler

# Profile operations
with TNSIMProfiler() as profiler:
    # Your computations here
    zs_set = ZeroSumInfiniteSet(range(10000))
    result = zs_set.zero_sum_operation(method='adaptive')

# Get profiling report
report = profiler.get_report()
print(f"Total time: {report['total_time']}s")
print(f"Memory usage: {report['memory_usage']} MB")
for func, time in report['function_times'].items():
    print(f"{func}: {time}s")
```

### Memory Management

```python
# For very large sets, use generators
def large_harmonic_generator(n):
    for i in range(1, n+1):
        yield 1.0 / i

# Process in chunks
chunk_size = 10000
total_sum = 0.0

for chunk_start in range(0, 1000000, chunk_size):
    chunk = list(large_harmonic_generator(chunk_size))
    zs_set = ZeroSumInfiniteSet(chunk)
    result = zs_set.zero_sum_operation(method='compensated')
    total_sum += result['sum']

print(f"Total sum: {total_sum}")
```

## Troubleshooting

### Common Issues

#### 1. Precision Loss

**Problem**: Results are not as precise as expected.

**Solution**:
```python
# Use higher precision method
result = zs_set.zero_sum_operation(method='adaptive')

# Or use Balansis integration
compensator = BalansisCompensator(precision='ultra')
compensated_sum = compensator.compensate_series(zs_set.elements)
```

#### 2. Performance Issues

**Problem**: Computations are too slow.

**Solution**:
```python
# Use parallel processing
parallel = ParallelTNSIM(num_workers=4)
result = parallel.parallel_zero_sum(zs_set)

# Or use caching
cache = TNSIMCache(max_size=1000)
# Cache your results
```

#### 3. Memory Issues

**Problem**: Out of memory with large sets.

**Solution**:
```python
# Process in chunks
chunk_size = 10000
for i in range(0, len(large_data), chunk_size):
    chunk = large_data[i:i+chunk_size]
    zs_set = ZeroSumInfiniteSet(chunk)
    # Process chunk
```

#### 4. Convergence Issues

**Problem**: Series doesn't converge as expected.

**Solution**:
```python
# Use multiple convergence tests
analyzer = ConvergenceAnalyzer()
tests = ['ratio', 'root', 'integral', 'alternating']

for test_name in tests:
    if hasattr(analyzer, f'{test_name}_test'):
        test_func = getattr(analyzer, f'{test_name}_test')
        result = test_func(series)
        print(f"{test_name}: {result}")
```

### Debugging

#### Enable Verbose Logging

```python
import logging

# Enable TNSIM logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('tnsim')

# Your code here
zs_set = ZeroSumInfiniteSet([1.0, -1.0])
result = zs_set.zero_sum_operation(method='compensated')
```

#### Diagnostic Information

```python
# Get diagnostic information
diag = zs_set.diagnose()
print(f"Status: {diag['status']}")
print(f"Issues: {diag['issues']}")
print(f"Recommendations: {diag['recommendations']}")
```

#### Validation

```python
from tnsim.utils.validation import validate_input, validate_tolerance

# Validate your data
try:
    validate_input(your_data, 'numeric_sequence')
    validate_tolerance(your_tolerance)
except ValueError as e:
    print(f"Validation error: {e}")
```

## Advanced Usage

### Custom Compensation Algorithms

```python
from tnsim.algorithms.base import BaseCompensator

class CustomCompensator(BaseCompensator):
    def __init__(self, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
    
    def compensate(self, elements, **kwargs):
        # Your custom algorithm here
        compensated_sum = sum(elements) * self.custom_param
        return {
            'sum': compensated_sum,
            'error': abs(compensated_sum),
            'method': 'custom'
        }

# Use custom compensator
custom = CustomCompensator(custom_param=0.99)
result = custom.compensate([1.0, 2.0, -3.0])
```

### Custom Convergence Tests

```python
from tnsim.core.convergence import ConvergenceAnalyzer

class CustomAnalyzer(ConvergenceAnalyzer):
    def custom_test(self, series):
        # Your custom convergence test
        # Return dict with 'convergent' and other info
        return {
            'convergent': True,
            'method': 'custom',
            'confidence': 0.95
        }

# Use custom analyzer
analyzer = CustomAnalyzer()
result = analyzer.custom_test(your_series)
```

### Integration with Other Libraries

#### With Dask for Distributed Computing

```python
import dask.array as da
from tnsim.core.zero_sum_set import ZeroSumInfiniteSet

# Create Dask array
dask_array = da.random.random((1000000,), chunks=(10000,))

# Convert chunks to zero-sum sets
def process_chunk(chunk):
    zs_set = ZeroSumInfiniteSet(chunk)
    return zs_set.zero_sum_operation(method='compensated')

# Process in parallel with Dask
results = dask_array.map_blocks(process_chunk, dtype=object)
computed_results = results.compute()
```

#### With Numba for JIT Compilation

```python
import numba
import numpy as np

@numba.jit
def fast_kahan_sum(arr):
    """Fast Kahan summation with Numba JIT."""
    sum_val = 0.0
    c = 0.0
    
    for i in range(len(arr)):
        y = arr[i] - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    
    return sum_val

# Use with TNSIM
zs_set = ZeroSumInfiniteSet(np.random.randn(1000000))
fast_result = fast_kahan_sum(zs_set.to_numpy())
print(f"Fast Kahan sum: {fast_result}")
```

### File I/O and Serialization

#### Saving and Loading Sets

```python
# Save to file
zs_set = ZeroSumInfiniteSet([1.0, 2.0, -3.0])
zs_set.save('my_set.tnsim')

# Load from file
loaded_set = ZeroSumInfiniteSet.load('my_set.tnsim')
print(f"Loaded elements: {loaded_set.elements}")
```

#### JSON Serialization

```python
# Convert to JSON
json_data = zs_set.to_json()
print(f"JSON: {json_data}")

# Create from JSON
restored_set = ZeroSumInfiniteSet.from_json(json_data)
print(f"Restored: {restored_set.elements}")
```

#### Export to Other Formats

```python
# Export to NumPy
np_array = zs_set.to_numpy()
np.save('my_set.npy', np_array)

# Export to Pandas
pd_series = zs_set.to_pandas()
pd_series.to_csv('my_set.csv')

# Export to HDF5
import h5py
with h5py.File('my_set.h5', 'w') as f:
    f.create_dataset('elements', data=zs_set.to_numpy())
    f.attrs['tolerance'] = zs_set.tolerance
```

## Best Practices

### 1. Choose the Right Method

```python
# For small sets (< 1000 elements)
result = zs_set.zero_sum_operation(method='compensated')

# For medium sets (1000-100000 elements)
result = zs_set.zero_sum_operation(method='iterative')

# For large sets (> 100000 elements)
result = zs_set.zero_sum_operation(method='adaptive')

# For extreme precision requirements
compensator = BalansisCompensator(precision='ultra')
result = compensator.compensate_series(zs_set.elements)
```

### 2. Use Appropriate Tolerance

```python
# For financial calculations
zs_set = ZeroSumInfiniteSet(financial_data, tolerance=1e-10)

# For scientific computations
zs_set = ZeroSumInfiniteSet(scientific_data, tolerance=1e-15)

# For approximate calculations
zs_set = ZeroSumInfiniteSet(approximate_data, tolerance=1e-6)
```

### 3. Memory Management

```python
# For large datasets, use generators
def data_generator():
    for i in range(1000000):
        yield compute_value(i)

# Process in chunks
chunk_size = 10000
for chunk in chunked(data_generator(), chunk_size):
    zs_set = ZeroSumInfiniteSet(chunk)
    # Process chunk
    del zs_set  # Explicit cleanup
```

### 4. Error Handling

```python
from tnsim.exceptions import ZeroSumError, CompensationError

try:
    zs_set = ZeroSumInfiniteSet(problematic_data)
    result = zs_set.zero_sum_operation(method='adaptive')
except ZeroSumError as e:
    print(f"Zero-sum error: {e}")
    # Handle zero-sum specific error
except CompensationError as e:
    print(f"Compensation error: {e}")
    # Try different method
    result = zs_set.zero_sum_operation(method='direct')
except Exception as e:
    print(f"Unexpected error: {e}")
    # General error handling
```

### 5. Performance Monitoring

```python
import time
from tnsim.utils.profiler import TNSIMProfiler

# Monitor performance
with TNSIMProfiler() as profiler:
    start_time = time.time()
    
    # Your computations
    result = zs_set.zero_sum_operation(method='adaptive')
    
    end_time = time.time()
    
print(f"Execution time: {end_time - start_time:.6f}s")
report = profiler.get_report()
print(f"Memory usage: {report['memory_usage']} MB")
```

### 6. Testing and Validation

```python
# Always validate your results
def validate_result(zs_set, result, expected_tolerance=1e-12):
    """Validate zero-sum operation result."""
    if abs(result['sum']) > expected_tolerance:
        print(f"Warning: Sum {result['sum']} exceeds tolerance")
    
    if 'error' in result and result['error'] > expected_tolerance:
        print(f"Warning: Error {result['error']} is high")
    
    return abs(result['sum']) <= expected_tolerance

# Use validation
result = zs_set.zero_sum_operation(method='compensated')
is_valid = validate_result(zs_set, result)
print(f"Result is valid: {is_valid}")
```

### 7. Documentation and Logging

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Document your computations
def documented_computation(data, method='compensated'):
    """Perform zero-sum computation with documentation.
    
    Args:
        data: Input data for zero-sum set
        method: Compensation method to use
    
    Returns:
        dict: Computation result with metadata
    """
    logger.info(f"Starting computation with {len(data)} elements")
    logger.info(f"Using method: {method}")
    
    zs_set = ZeroSumInfiniteSet(data)
    result = zs_set.zero_sum_operation(method=method)
    
    logger.info(f"Computation completed: sum={result['sum']:.2e}")
    
    # Add metadata
    result['metadata'] = {
        'input_size': len(data),
        'method': method,
        'timestamp': time.time()
    }
    
    return result
```

This user guide provides comprehensive coverage of TNSIM functionality. For more specific use cases or advanced topics, refer to the API reference and examples in the repository.