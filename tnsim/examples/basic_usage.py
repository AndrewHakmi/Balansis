"""Basic usage examples for TNSIM."""

import asyncio
from decimal import Decimal
import numpy as np
import torch
from typing import List, Dict, Any

from tnsim.core.sets import ZeroSumInfiniteSet
from tnsim.core.cache import TNSIMCache
from tnsim.core.parallel import ParallelTNSIM
from tnsim.integrations.balansis_integration import (
    BalansisCompensator,
    ZeroSumAttention,
    ZeroSumTransformerBlock
)


def example_basic_zero_sum_set():
    """Example of creating and working with zero-sum sets."""
    print("=== Basic ZeroSumInfiniteSet Example ===")
    
    # Creating harmonic series
    harmonic_elements = [1/n for n in range(1, 101)]
    harmonic_set = ZeroSumInfiniteSet(harmonic_elements, series_type='harmonic')
    
    print(f"Created set with {len(harmonic_set.elements)} elements")
    print(f"Series type: {harmonic_set.series_type}")
    
    # Zero-sum operation
    direct_sum = harmonic_set.zero_sum_operation(method='direct')
    compensated_sum = harmonic_set.zero_sum_operation(method='compensated')
    stabilized_sum = harmonic_set.zero_sum_operation(method='stabilized')
    
    print(f"\nZero-sum operation results:")
    print(f"Direct summation: {direct_sum:.10f}")
    print(f"Compensated: {compensated_sum:.10f}")
    print(f"Stabilized: {stabilized_sum:.10f}")
    
    # Finding compensating set
    compensating_set = harmonic_set.find_compensating_set(method='adaptive')
    print(f"\nCompensating set found with {len(compensating_set.elements)} elements")
    
    # Zero-sum validation
    validation_result = harmonic_set.validate_zero_sum(tolerance=1e-10)
    print(f"\nZero-sum validation:")
    print(f"Valid: {validation_result['is_valid']}")
    print(f"Error: {validation_result['error']:.2e}")
    
    # Convergence analysis
    convergence = harmonic_set.convergence_analysis(test='ratio')
    print(f"\nConvergence analysis (ratio test):")
    print(f"Converges: {convergence['converges']}")
    print(f"Type: {convergence['convergence_type']}")
    
    return harmonic_set


def example_different_series_types():
    """Example of working with different series types."""
    print("\n=== Examples of Different Series Types ===")
    
    # Harmonic series
    harmonic = ZeroSumInfiniteSet.create_harmonic_series(100)
    print(f"Harmonic series: sum = {harmonic.zero_sum_operation():.6f}")
    
    # Alternating series
    alternating = ZeroSumInfiniteSet.create_alternating_series(100)
    print(f"Alternating series: sum = {alternating.zero_sum_operation():.6f}")
    
    # Geometric series
    geometric = ZeroSumInfiniteSet.create_geometric_series(0.5, 100)
    print(f"Geometric series (r=0.5): sum = {geometric.zero_sum_operation():.6f}")
    
    # Custom series
    custom_elements = [1/(n**2) for n in range(1, 101)]
    custom_set = ZeroSumInfiniteSet(custom_elements, series_type='custom')
    print(f"Custom series (1/n²): sum = {custom_set.zero_sum_operation():.6f}")
    
    return [harmonic, alternating, geometric, custom_set]


def example_cache_usage():
    """Example of cache usage."""
    print("\n=== Cache Usage Example ===")
    
    # Creating cache
    cache = TNSIMCache(
        max_size=100,
        eviction_strategy='lru',
        ttl_seconds=300
    )
    
    # Creating sets for caching
    sets = []
    for i in range(10):
        elements = [1/(n+i) for n in range(1, 51)]
        zs_set = ZeroSumInfiniteSet(elements)
        sets.append(zs_set)
    
    print(f"Created {len(sets)} sets for caching")
    
    # Computing and caching results
    for i, zs_set in enumerate(sets):
        cache_key = cache.generate_key('zero_sum', zs_set.elements[:5])
        
        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            print(f"Set {i}: result from cache = {cached_result:.6f}")
        else:
            # Compute and save to cache
            result = zs_set.zero_sum_operation()
            cache.put(cache_key, result)
            print(f"Set {i}: computed and saved = {result:.6f}")
    
    # Cache statistics
    stats = cache.get_stats()
    print(f"\nCache statistics:")
    print(f"Size: {stats['size']}")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    
    return cache


def example_parallel_processing():
    """Example of parallel processing."""
    print("\n=== Parallel Processing Example ===")
    
    # Creating sets for parallel processing
    sets = []
    for i in range(20):
        elements = [1/(n+i) for n in range(1, 101)]
        zs_set = ZeroSumInfiniteSet(elements)
        sets.append(zs_set)
    
    print(f"Created {len(sets)} sets for parallel processing")
    
    # Parallel processing
    parallel_tnsim = ParallelTNSIM(max_workers=4, chunk_size=5)
    
    # Parallel zero-sum operations
    results = parallel_tnsim.parallel_zero_sum(sets, method='compensated')
    print(f"\nParallel zero-sum operations:")
    for i, result in enumerate(results[:5]):  # Show first 5
        print(f"Set {i}: {result:.6f}")
    
    # Parallel compensating set search
    compensating_sets = parallel_tnsim.parallel_compensating_sets(
        sets[:5], method='iterative'
    )
    print(f"\nFound {len(compensating_sets)} compensating sets")
    
    # Batch processing
    operations = []
    for zs_set in sets[:10]:
        operations.extend([
            ('zero_sum', zs_set, {'method': 'direct'}),
            ('validate', zs_set, {'tolerance': 1e-8}),
            ('convergence', zs_set, {'test': 'ratio'})
        ])
    
    batch_results = parallel_tnsim.batch_process(operations)
    print(f"\nBatch processing: executed {len(batch_results)} operations")
    
    return parallel_tnsim, results


def example_balansis_integration():
    """Example of Balansis integration."""
    print("\n=== Balansis Integration Example ===")
    
    # Balansis compensator
    compensator = BalansisCompensator(precision='high')
    
    # Creating series for compensation
    series = [1/n - 1/(n+1) for n in range(1, 1001)]
    print(f"Created series with {len(series)} elements")
    
    # Compensated summation
    compensated_sum, metrics = compensator.compensate_series(series)
    print(f"\nCompensated sum: {compensated_sum:.10f}")
    print(f"Compensation error: {metrics.compensation_error:.2e}")
    print(f"Quality score: {metrics.quality_score:.4f}")
    
    # Working with zero-sum sets
    harmonic_set = ZeroSumInfiniteSet.create_harmonic_series(500)
    compensated_zs, zs_metrics = compensator.compensate_zero_sum_set(harmonic_set)
    print(f"\nCompensated zero-sum: {compensated_zs:.10f}")
    print(f"Accuracy improvement: {zs_metrics.accuracy_improvement:.2f}x")
    
    return compensator


def example_attention_mechanism():
    """Example of using zero-sum attention mechanism."""
    print("\n=== Zero-Sum Attention Mechanism Example ===")
    
    # Creating attention model
    d_model = 256
    n_heads = 8
    attention = ZeroSumAttention(
        d_model=d_model,
        n_heads=n_heads,
        compensation_strength=0.1
    )
    
    print(f"Created attention model: d_model={d_model}, n_heads={n_heads}")
    
    # Creating input data
    batch_size, seq_len = 4, 32
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input data: {x.shape}")
    
    # Forward pass
    output, attention_weights = attention(x)
    
    print(f"\nOutput data: {output.shape}")
    print(f"Attention weights: {attention_weights.shape}")
    
    # Checking zero-sum properties
    attention_sum = attention_weights.sum(dim=-1)
    print(f"\nAttention weights sum (should be ~1.0):")
    print(f"Mean: {attention_sum.mean().item():.6f}")
    print(f"Standard deviation: {attention_sum.std().item():.6f}")
    
    # Using mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    masked_output, masked_weights = attention(x, mask=mask)
    
    print(f"\nMasked output: {masked_output.shape}")
    
    return attention


def example_transformer_block():
    """Example of using zero-sum transformer block."""
    print("\n=== Zero-Sum Transformer Block Example ===")
    
    # Creating transformer block
    d_model = 512
    n_heads = 16
    d_ff = 2048
    
    transformer_block = ZeroSumTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    
    print(f"Created transformer block: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    
    # Input data
    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input data: {x.shape}")
    
    # Forward pass
    output = transformer_block(x)
    
    print(f"Output data: {output.shape}")
    
    # Checking residual connections
    residual_norm = (output - x).norm().item()
    print(f"\nResidual connection norm: {residual_norm:.6f}")
    
    return transformer_block


def example_serialization():
    """Example of serialization and deserialization."""
    print("\n=== Serialization and Deserialization Example ===")
    
    # Creating set
    original_set = ZeroSumInfiniteSet.create_alternating_series(100)
    original_set.metadata = {
        'created_by': 'example_script',
        'purpose': 'demonstration',
        'version': '1.0'
    }
    
    print(f"Original set: {len(original_set.elements)} elements")
    print(f"Metadata: {original_set.metadata}")
    
    # Serialization to dictionary
    serialized_data = original_set.to_dict()
    print(f"\nSerialized to dictionary with keys: {list(serialized_data.keys())}")
    
    # Deserialization from dictionary
    restored_set = ZeroSumInfiniteSet.from_dict(serialized_data)
    
    print(f"\nRestored set: {len(restored_set.elements)} elements")
    print(f"Metadata: {restored_set.metadata}")
    
    # Equivalence check
    original_sum = original_set.zero_sum_operation()
    restored_sum = restored_set.zero_sum_operation()
    
    print(f"\nEquivalence check:")
    print(f"Original sum: {original_sum:.10f}")
    print(f"Restored sum: {restored_sum:.10f}")
    print(f"Difference: {abs(original_sum - restored_sum):.2e}")
    
    return original_set, restored_set


def example_advanced_operations():
    """Example of advanced operations."""
    print("\n=== Advanced Operations Example ===")
    
    # Creating complex set
    elements = []
    for n in range(1, 201):
        if n % 2 == 0:
            elements.append(1/n)
        else:
            elements.append(-1/n)
    
    complex_set = ZeroSumInfiniteSet(elements, series_type='custom')
    print(f"Created complex set with {len(elements)} elements")
    
    # Partial sums
    partial_sums = []
    for i in range(0, len(elements), 20):
        end_idx = min(i + 19, len(elements) - 1)
        partial_sum = complex_set.get_partial_sum(i, end_idx)
        partial_sums.append(partial_sum)
    
    print(f"\nComputed {len(partial_sums)} partial sums")
    print(f"First 5 partial sums: {[f'{s:.6f}' for s in partial_sums[:5]]}")
    
    # Convergence analysis with different methods
    convergence_tests = ['ratio', 'root', 'integral']
    for test in convergence_tests:
        result = complex_set.convergence_analysis(test=test)
        print(f"\n{test} test:")
        print(f"  Converges: {result['converges']}")
        print(f"  Type: {result['convergence_type']}")
        print(f"  Rate: {result['convergence_rate']:.6f}")
    
    # Finding compensating set with different methods
    compensation_methods = ['direct', 'iterative', 'adaptive']
    for method in compensation_methods:
        compensating = complex_set.find_compensating_set(
            method=method,
            max_iterations=50
        )
        quality = complex_set.validate_zero_sum()['error']
        print(f"\n{method} method:")
        print(f"  Compensating set size: {len(compensating.elements)}")
        print(f"  Compensation quality: {quality:.2e}")
    
    return complex_set


async def example_async_operations():
    """Example of asynchronous operations."""
    print("\n=== Asynchronous Operations Example ===")
    
    # Creating sets for asynchronous processing
    sets = []
    for i in range(10):
        elements = [1/(n+i) for n in range(1, 101)]
        zs_set = ZeroSumInfiniteSet(elements)
        sets.append(zs_set)
    
    print(f"Created {len(sets)} sets for asynchronous processing")
    
    async def process_set(zs_set, index):
        """Asynchronous set processing."""
        # Simulating asynchronous work
        await asyncio.sleep(0.1)
        
        result = zs_set.zero_sum_operation(method='compensated')
        return index, result
    
    # Parallel asynchronous processing
    tasks = [process_set(zs_set, i) for i, zs_set in enumerate(sets)]
    results = await asyncio.gather(*tasks)
    
    print(f"\nAsynchronous processing completed:")
    for index, result in results:
        print(f"Set {index}: {result:.6f}")
    
    return results


def main():
    """Main function with usage examples."""
    print("TNSIM - Theory of Zero-Sum Infinite Sets")
    print("=" * 60)
    
    # Basic examples
    harmonic_set = example_basic_zero_sum_set()
    series_sets = example_different_series_types()
    cache = example_cache_usage()
    parallel_tnsim, parallel_results = example_parallel_processing()
    
    # Balansis integration
    compensator = example_balansis_integration()
    attention = example_attention_mechanism()
    transformer = example_transformer_block()
    
    # Additional features
    original_set, restored_set = example_serialization()
    complex_set = example_advanced_operations()
    
    # Asynchronous operations
    print("\nRunning asynchronous operations...")
    async_results = asyncio.run(example_async_operations())
    
    print("\n=== Results Summary ===")
    print(f"Processed sets: {len(series_sets) + len(parallel_results) + len(async_results)}")
    print(f"Cache contains: {cache.get_stats()['size']} elements")
    print(f"Cache hit rate: {cache.get_stats()['hit_rate']:.2%}")
    
    print("\nAll examples executed successfully!")


if __name__ == "__main__":
    main()