# TNSIM API Reference

## Table of Contents

1. [Core Classes](#core-classes)
   - [ZeroSumInfiniteSet](#zerosuminfiniteset)
   - [ConvergenceAnalyzer](#convergenceanalyzer)
   - [ParallelTNSIM](#paralleltnsim)
2. [Compensation Algorithms](#compensation-algorithms)
   - [BaseCompensator](#basecompensator)
   - [KahanCompensator](#kahancompensator)
   - [IterativeCompensator](#iterativecompensator)
   - [AdaptiveCompensator](#adaptivecompensator)
   - [StabilizedCompensator](#stabilizedcompensator)
3. [Utility Classes](#utility-classes)
   - [TNSIMCache](#tnsimcache)
   - [TNSIMProfiler](#tnsimprofiler)
   - [MethodBenchmark](#methodbenchmark)
4. [Integration Classes](#integration-classes)
   - [BalansisCompensator](#balansiscompensator)
   - [ZeroSumAttention](#zerosumatention)
5. [Exceptions](#exceptions)
6. [Constants and Enums](#constants-and-enums)

---

## Core Classes

### ZeroSumInfiniteSet

Main class for working with zero-sum infinite sets.

#### Constructor

```python
ZeroSumInfiniteSet(elements=None, tolerance=1e-15)
```

**Parameters:**
- `elements` (iterable, optional): Initial elements of the set
- `tolerance` (float): Tolerance for zero-sum check

**Example:**
```python
zs_set = ZeroSumInfiniteSet([1.0, 2.0, -3.0], tolerance=1e-12)
```

#### Class Methods

##### `from_function(func, n, start=0, **kwargs)`

Create set from function.

**Parameters:**
- `func` (callable): Function to generate elements
- `n` (int): Number of elements
- `start` (int): Starting index
- `**kwargs`: Additional arguments for function

**Returns:** `ZeroSumInfiniteSet`

**Example:**
```python
harmonic = ZeroSumInfiniteSet.from_function(lambda i: 1.0/i, n=1000, start=1)
```

##### `from_numpy(array, **kwargs)`

Create set from NumPy array.

**Parameters:**
- `array` (numpy.ndarray): NumPy array
- `**kwargs`: Additional arguments for constructor

**Returns:** `ZeroSumInfiniteSet`

##### `from_pandas(series, **kwargs)`

Create set from Pandas series.

**Parameters:**
- `series` (pandas.Series): Pandas series
- `**kwargs`: Additional arguments for constructor

**Returns:** `ZeroSumInfiniteSet`

##### `load(filename)`

Load set from file.

**Parameters:**
- `filename` (str): Path to file

**Returns:** `ZeroSumInfiniteSet`

#### Instance Methods

##### `is_zero_sum(tolerance=None)`

Check if set is zero-sum.

**Parameters:**
- `tolerance` (float, optional): Custom tolerance

**Returns:** `bool`

##### `sum()`

Calculate sum of elements.

**Returns:** `float`

##### `size()`

Get number of elements.

**Returns:** `int`

##### `add_element(element)`

Add element to set.

**Parameters:**
- `element` (float): Element to add

##### `add_elements(elements)`

Add multiple elements to set.

**Parameters:**
- `elements` (iterable): Elements to add

##### `remove_element(element)`

Remove element from set.

**Parameters:**
- `element` (float): Element to remove

**Returns:** `bool` - True if element was removed

##### `zero_sum_operation(method='compensated', **kwargs)`

Perform zero-sum operation with compensation.

**Parameters:**
- `method` (str): Compensation method ('direct', 'compensated', 'iterative', 'adaptive', 'stabilized')
- `**kwargs`: Method-specific parameters

**Returns:** `dict` with keys:
- `sum`: Compensated sum
- `error`: Error estimate
- `method`: Used method
- Additional method-specific keys

**Example:**
```python
result = zs_set.zero_sum_operation(method='adaptive', target_precision=1e-15)
```

##### `statistics()`

Get comprehensive statistics.

**Returns:** `dict` with statistical measures

##### `mean()`, `std()`, `min()`, `max()`, `median()`

Basic statistical methods.

**Returns:** `float`

##### `skewness()`, `kurtosis()`, `entropy()`

Advanced statistical methods.

**Returns:** `float`

##### `union(other)`, `intersection(other)`, `difference(other)`

Set operations.

**Parameters:**
- `other` (ZeroSumInfiniteSet): Other set

**Returns:** `ZeroSumInfiniteSet`

##### `to_numpy()`, `to_pandas()`, `to_json()`

Conversion methods.

**Returns:** Corresponding data type

##### `save(filename)`

Save set to file.

**Parameters:**
- `filename` (str): Path to file

##### `diagnose()`

Get diagnostic information.

**Returns:** `dict` with diagnostic data

#### Properties

- `elements`: List of elements
- `tolerance`: Current tolerance

---

### ConvergenceAnalyzer

Class for analyzing convergence of infinite series.

#### Constructor

```python
ConvergenceAnalyzer(tolerance=1e-12, max_terms=10000)
```

**Parameters:**
- `tolerance` (float): Convergence tolerance
- `max_terms` (int): Maximum number of terms to analyze

#### Methods

##### `ratio_test(series)`

Perform ratio test for convergence.

**Parameters:**
- `series` (list): Series to test

**Returns:** `dict` with convergence information

##### `root_test(series)`

Perform root test for convergence.

**Parameters:**
- `series` (list): Series to test

**Returns:** `dict` with convergence information

##### `integral_test(func, a, b)`

Perform integral test for convergence.

**Parameters:**
- `func` (callable): Function to integrate
- `a` (float): Lower bound
- `b` (float): Upper bound

**Returns:** `dict` with convergence information

##### `alternating_series_test(series)`

Test convergence of alternating series.

**Parameters:**
- `series` (list): Alternating series

**Returns:** `dict` with convergence information

##### `comparison_test(series1, series2)`

Compare convergence of two series.

**Parameters:**
- `series1` (list): First series
- `series2` (list): Second series (known convergence)

**Returns:** `dict` with comparison results

##### `partial_sums_analysis(series)`

Analyze behavior of partial sums.

**Parameters:**
- `series` (list): Series to analyze

**Returns:** `dict` with analysis results

---

### ParallelTNSIM

Class for parallel processing of zero-sum operations.

#### Constructor

```python
ParallelTNSIM(num_workers=None, backend='multiprocessing', chunk_size=1000)
```

**Parameters:**
- `num_workers` (int, optional): Number of worker processes/threads
- `backend` (str): Backend type ('multiprocessing', 'threading')
- `chunk_size` (int): Default chunk size for processing

#### Methods

##### `parallel_zero_sum(zs_set, method='compensated', chunk_size=None, **kwargs)`

Perform parallel zero-sum operation.

**Parameters:**
- `zs_set` (ZeroSumInfiniteSet): Set to process
- `method` (str): Compensation method
- `chunk_size` (int, optional): Chunk size for processing
- `**kwargs`: Method-specific parameters

**Returns:** `dict` with results and performance metrics

##### `batch_process(sets, method='compensated', **kwargs)`

Process multiple sets in parallel.

**Parameters:**
- `sets` (list): List of ZeroSumInfiniteSet objects
- `method` (str): Compensation method
- `**kwargs`: Method-specific parameters

**Returns:** `list` of results

##### `get_performance_stats()`

Get performance statistics.

**Returns:** `dict` with performance metrics

##### `shutdown()`

Shutdown parallel processor.

---

## Compensation Algorithms

### BaseCompensator

Base class for all compensation algorithms.

#### Constructor

```python
BaseCompensator()
```

#### Methods

##### `compensate(elements, **kwargs)`

Perform compensation (abstract method).

**Parameters:**
- `elements` (iterable): Elements to compensate
- `**kwargs`: Algorithm-specific parameters

**Returns:** `dict` with compensation results

---

### KahanCompensator

Kahan summation algorithm implementation.

#### Constructor

```python
KahanCompensator()
```

#### Methods

##### `compensate(elements, **kwargs)`

Perform Kahan summation.

**Parameters:**
- `elements` (iterable): Elements to sum

**Returns:** `dict` with keys:
- `sum`: Compensated sum
- `error`: Error estimate
- `method`: 'kahan'

---

### IterativeCompensator

Iterative compensation algorithm.

#### Constructor

```python
IterativeCompensator(max_iterations=100, tolerance=1e-16)
```

**Parameters:**
- `max_iterations` (int): Maximum number of iterations
- `tolerance` (float): Convergence tolerance

#### Methods

##### `compensate(elements, **kwargs)`

Perform iterative compensation.

**Parameters:**
- `elements` (iterable): Elements to compensate
- `max_iterations` (int, optional): Override max iterations
- `tolerance` (float, optional): Override tolerance

**Returns:** `dict` with compensation results and iteration count

---

### AdaptiveCompensator

Adaptive compensation algorithm.

#### Constructor

```python
AdaptiveCompensator(target_precision=1e-15, adaptation_rate=0.1)
```

**Parameters:**
- `target_precision` (float): Target precision level
- `adaptation_rate` (float): Rate of parameter adaptation

#### Methods

##### `compensate(elements, **kwargs)`

Perform adaptive compensation.

**Parameters:**
- `elements` (iterable): Elements to compensate
- `target_precision` (float, optional): Override target precision

**Returns:** `dict` with compensation results and adapted parameters

---

### StabilizedCompensator

Stabilized compensation for extreme cases.

#### Constructor

```python
StabilizedCompensator(methods=None, weights=None)
```

**Parameters:**
- `methods` (list, optional): List of methods to combine
- `weights` (list, optional): Weights for method combination

#### Methods

##### `compensate(elements, **kwargs)`

Perform stabilized compensation.

**Parameters:**
- `elements` (iterable): Elements to compensate

**Returns:** `dict` with compensation results and method contributions

---

## Utility Classes

### TNSIMCache

Caching system for TNSIM operations.

#### Constructor

```python
TNSIMCache(max_size=1000, ttl=3600, eviction_policy='lru')
```

**Parameters:**
- `max_size` (int): Maximum cache size
- `ttl` (int): Time to live in seconds
- `eviction_policy` (str): Eviction policy ('lru', 'fifo', 'random')

#### Methods

##### `get(key)`

Get value from cache.

**Parameters:**
- `key` (str): Cache key

**Returns:** Cached value or None

##### `put(key, value)`

Put value in cache.

**Parameters:**
- `key` (str): Cache key
- `value`: Value to cache

##### `clear()`

Clear cache.

##### `get_stats()`

Get cache statistics.

**Returns:** `dict` with cache metrics

---

### TNSIMProfiler

Profiling utility for performance analysis.

#### Constructor

```python
TNSIMProfiler(enable_memory_profiling=False)
```

**Parameters:**
- `enable_memory_profiling` (bool): Enable memory profiling

#### Methods

##### `start()`

Start profiling.

##### `stop()`

Stop profiling.

##### `get_report()`

Get profiling report.

**Returns:** `dict` with profiling data

#### Context Manager Usage

```python
with TNSIMProfiler() as profiler:
    # Your code here
    pass

report = profiler.get_report()
```

---

### MethodBenchmark

Benchmarking utility for comparing methods.

#### Constructor

```python
MethodBenchmark(iterations=100, warmup_iterations=10)
```

**Parameters:**
- `iterations` (int): Number of benchmark iterations
- `warmup_iterations` (int): Number of warmup iterations

#### Methods

##### `compare_methods(data, methods, **kwargs)`

Compare multiple compensation methods.

**Parameters:**
- `data` (iterable): Test data
- `methods` (list): List of method names
- `**kwargs`: Method-specific parameters

**Returns:** `dict` with benchmark results for each method

##### `benchmark_method(data, method, **kwargs)`

Benchmark single method.

**Parameters:**
- `data` (iterable): Test data
- `method` (str): Method name
- `**kwargs`: Method-specific parameters

**Returns:** `dict` with benchmark results

---

## Integration Classes

### BalansisCompensator

Integration with Balansis library for ultra-high precision.

#### Constructor

```python
BalansisCompensator(precision='high', algorithm='auto')
```

**Parameters:**
- `precision` (str): Precision level ('low', 'medium', 'high', 'ultra')
- `algorithm` (str): Algorithm choice ('auto', 'kahan', 'neumaier', 'klein')

#### Methods

##### `compensate_series(series)`

Compensate series using Balansis.

**Parameters:**
- `series` (iterable): Series to compensate

**Returns:** `float` - Compensated sum

##### `compensate_zero_sum_set(zs_set)`

Compensate zero-sum set using Balansis.

**Parameters:**
- `zs_set` (ZeroSumInfiniteSet): Set to compensate

**Returns:** `ZeroSumInfiniteSet` - Compensated set

##### `assess_compensation_quality(original, compensated)`

Assess quality of compensation.

**Parameters:**
- `original` (float): Original sum
- `compensated` (float): Compensated sum

**Returns:** `dict` with quality metrics

---

### ZeroSumAttention

Zero-sum attention mechanism for neural networks.

#### Constructor

```python
ZeroSumAttention(d_model, num_heads, compensation_method='adaptive', dropout=0.1)
```

**Parameters:**
- `d_model` (int): Model dimension
- `num_heads` (int): Number of attention heads
- `compensation_method` (str): Compensation method for attention weights
- `dropout` (float): Dropout rate

#### Methods

##### `forward(query, key, value, mask=None)`

Forward pass of attention mechanism.

**Parameters:**
- `query` (torch.Tensor): Query tensor
- `key` (torch.Tensor): Key tensor
- `value` (torch.Tensor): Value tensor
- `mask` (torch.Tensor, optional): Attention mask

**Returns:** `tuple` - (output, attention_weights)

##### `get_compensation_stats()`

Get compensation statistics.

**Returns:** `dict` with compensation metrics

---

## Exceptions

### ZeroSumError

Raised when zero-sum operations fail.

```python
class ZeroSumError(Exception):
    """Exception raised for zero-sum related errors."""
    pass
```

### CompensationError

Raised when compensation algorithms fail.

```python
class CompensationError(Exception):
    """Exception raised for compensation algorithm errors."""
    pass
```

### ConvergenceError

Raised when convergence analysis fails.

```python
class ConvergenceError(Exception):
    """Exception raised for convergence analysis errors."""
    pass
```

### ValidationError

Raised when input validation fails.

```python
class ValidationError(Exception):
    """Exception raised for input validation errors."""
    pass
```

---

## Constants and Enums

### CompensationMethod

Enumeration of available compensation methods.

```python
from enum import Enum

class CompensationMethod(Enum):
    DIRECT = 'direct'
    COMPENSATED = 'compensated'
    ITERATIVE = 'iterative'
    ADAPTIVE = 'adaptive'
    STABILIZED = 'stabilized'
```

### ConvergenceTest

Enumeration of convergence tests.

```python
class ConvergenceTest(Enum):
    RATIO = 'ratio'
    ROOT = 'root'
    INTEGRAL = 'integral'
    ALTERNATING = 'alternating'
    COMPARISON = 'comparison'
```

### PrecisionLevel

Enumeration of precision levels for Balansis integration.

```python
class PrecisionLevel(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    ULTRA = 'ultra'
```

### Constants

```python
# Default tolerance for zero-sum checks
DEFAULT_TOLERANCE = 1e-15

# Maximum number of elements for direct processing
MAX_DIRECT_ELEMENTS = 100000

# Default chunk size for parallel processing
DEFAULT_CHUNK_SIZE = 1000

# Default cache size
DEFAULT_CACHE_SIZE = 1000

# Default TTL for cache entries (seconds)
DEFAULT_CACHE_TTL = 3600
```

---

## Usage Examples

### Basic Usage

```python
from tnsim.core.zero_sum_set import ZeroSumInfiniteSet
from tnsim.algorithms.kahan import KahanCompensator

# Create zero-sum set
zs_set = ZeroSumInfiniteSet([1.0, 2.0, -3.0])

# Check if zero-sum
print(zs_set.is_zero_sum())  # True

# Perform compensated operation
result = zs_set.zero_sum_operation(method='compensated')
print(f"Sum: {result['sum']}, Error: {result['error']}")

# Use compensator directly
compensator = KahanCompensator()
result = compensator.compensate([0.1] * 10)
print(f"Kahan sum: {result['sum']}")
```

### Advanced Usage

```python
from tnsim.core.parallel import ParallelTNSIM
from tnsim.core.convergence import ConvergenceAnalyzer
from tnsim.integrations.balansis_integration import BalansisCompensator

# Parallel processing
parallel = ParallelTNSIM(num_workers=4)
large_set = ZeroSumInfiniteSet.from_function(lambda i: 1.0/i, n=100000, start=1)
result = parallel.parallel_zero_sum(large_set, method='adaptive')
print(f"Parallel result: {result['sum']}")

# Convergence analysis
analyzer = ConvergenceAnalyzer()
geometric = [0.5**n for n in range(100)]
conv_result = analyzer.ratio_test(geometric)
print(f"Convergent: {conv_result['convergent']}")

# Balansis integration
balansis = BalansisCompensator(precision='ultra')
problematic = [1e-16] * 1000
compensated = balansis.compensate_series(problematic)
print(f"Balansis result: {compensated}")

parallel.shutdown()
```

This API reference provides comprehensive documentation for all TNSIM classes and methods. For more examples and detailed usage patterns, refer to the user guide and example notebooks.