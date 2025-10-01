"""Advanced examples of TNSIM usage."""

import asyncio
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from pathlib import Path

from tnsim.core.sets import ZeroSumInfiniteSet
from tnsim.core.cache import TNSIMCache
from tnsim.core.parallel import ParallelTNSIM
from tnsim.integrations.balansis_integration import (
    BalansisCompensator,
    ZeroSumAttention,
    ZeroSumTransformerBlock,
    CompensationMetrics
)


class AdvancedTNSIMDemo:
    """Demonstration class for advanced TNSIM capabilities."""
    
    def __init__(self):
        self.cache = TNSIMCache(max_size=1000, eviction_strategy='lru')
        self.parallel_processor = ParallelTNSIM(max_workers=8, chunk_size=10)
        self.compensator = BalansisCompensator(precision='ultra_high')
        self.results_history = []
    
    def benchmark_methods(self, series_sizes: List[int]) -> Dict[str, List[float]]:
        """Benchmark of different summation methods."""
        print("=== Summation Methods Benchmark ===")
        
        methods = ['direct', 'compensated', 'stabilized']
        results = {method: [] for method in methods}
        times = {method: [] for method in methods}
        
        for size in series_sizes:
            print(f"\nTesting size: {size}")
            
            # Creating test series
            elements = [1/n - 1/(n+1) for n in range(1, size+1)]
            test_set = ZeroSumInfiniteSet(elements)
            
            for method in methods:
                start_time = time.time()
                result = test_set.zero_sum_operation(method=method)
                end_time = time.time()
                
                results[method].append(result)
                times[method].append(end_time - start_time)
                
                print(f"  {method}: {result:.10f} ({end_time - start_time:.4f}s)")
        
        # Visualizing results
        self._plot_benchmark_results(series_sizes, results, times)
        
        return {'results': results, 'times': times}
    
    def _plot_benchmark_results(self, sizes: List[int], results: Dict, times: Dict):
        """Plotting benchmark results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        for method, values in results.items():
            ax1.plot(sizes, values, marker='o', label=method)
        ax1.set_xlabel('Series Size')
        ax1.set_ylabel('Summation Result')
        ax1.set_title('Method Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Performance plot
        for method, values in times.items():
            ax2.plot(sizes, values, marker='s', label=method)
        ax2.set_xlabel('Series Size')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Method Performance')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def convergence_analysis_suite(self, test_series: List[Tuple[str, List[float]]]) -> Dict:
        """Comprehensive convergence analysis of various series."""
        print("\n=== Comprehensive Convergence Analysis ===")
        
        analysis_results = {}
        convergence_tests = ['ratio', 'root', 'integral']
        
        for series_name, elements in test_series:
            print(f"\nAnalyzing series: {series_name}")
            test_set = ZeroSumInfiniteSet(elements, series_type='custom')
            
            series_results = {}
            for test in convergence_tests:
                result = test_set.convergence_analysis(test=test)
                series_results[test] = result
                
                print(f"  Test {test}:")
                print(f"    Converges: {result['converges']}")
                print(f"    Type: {result['convergence_type']}")
                print(f"    Rate: {result['convergence_rate']:.6f}")
            
            # Analyzing partial sums
            partial_sums = self._analyze_partial_sums(test_set)
            series_results['partial_sums'] = partial_sums
            
            analysis_results[series_name] = series_results
        
        # Visualizing convergence
        self._plot_convergence_analysis(analysis_results)
        
        return analysis_results
    
    def _analyze_partial_sums(self, zs_set: ZeroSumInfiniteSet) -> Dict:
        """Analysis of partial sums behavior."""
        n_points = min(100, len(zs_set.elements))
        indices = np.linspace(0, len(zs_set.elements)-1, n_points, dtype=int)
        
        partial_sums = []
        for i in indices:
            partial_sum = zs_set.get_partial_sum(0, i)
            partial_sums.append(partial_sum)
        
        # Statistical analysis
        partial_sums_array = np.array(partial_sums)
        return {
            'values': partial_sums,
            'indices': indices.tolist(),
            'mean': float(np.mean(partial_sums_array)),
            'std': float(np.std(partial_sums_array)),
            'trend': self._calculate_trend(partial_sums_array),
            'oscillation': self._calculate_oscillation(partial_sums_array)
        }
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculation of partial sums trend."""
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])  # Linear regression slope
    
    def _calculate_oscillation(self, values: np.ndarray) -> float:
        """Calculation of oscillation degree."""
        if len(values) < 2:
            return 0.0
        
        differences = np.diff(values)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        return float(sign_changes / (len(values) - 2))
    
    def _plot_convergence_analysis(self, analysis_results: Dict):
        """Plotting convergence analysis graphs."""
        n_series = len(analysis_results)
        fig, axes = plt.subplots(n_series, 1, figsize=(12, 4*n_series))
        
        if n_series == 1:
            axes = [axes]
        
        for i, (series_name, results) in enumerate(analysis_results.items()):
            partial_sums = results['partial_sums']
            
            axes[i].plot(partial_sums['indices'], partial_sums['values'], 
                        'b-', alpha=0.7, linewidth=1)
            axes[i].axhline(y=partial_sums['mean'], color='r', 
                           linestyle='--', label=f'Mean: {partial_sums["mean"]:.6f}')
            axes[i].set_title(f'Partial sums: {series_name}')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Partial sum')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compensation_quality_study(self, test_cases: List[Dict]) -> Dict:
        """Study of compensation quality."""
        print("\n=== Compensation Quality Study ===")
        
        results = []
        
        for case in test_cases:
            print(f"\nTest case: {case['name']}")
            
            # Creating set
            zs_set = ZeroSumInfiniteSet(case['elements'])
            
            # Different compensation methods
            compensation_methods = ['direct', 'iterative', 'adaptive']
            case_results = {'name': case['name'], 'methods': {}}
            
            for method in compensation_methods:
                compensating_set = zs_set.find_compensating_set(
                    method=method,
                    max_iterations=case.get('max_iterations', 100)
                )
                
                # Quality assessment
                validation = zs_set.validate_zero_sum(tolerance=1e-12)
                
                # Compensation with Balansis
                balansis_result, balansis_metrics = self.compensator.compensate_zero_sum_set(zs_set)
                
                method_result = {
                    'compensating_size': len(compensating_set.elements),
                    'validation_error': validation['error'],
                    'is_valid': validation['is_valid'],
                    'balansis_compensation': balansis_result,
                    'balansis_quality': balansis_metrics.quality_score,
                    'balansis_error': balansis_metrics.compensation_error
                }
                
                case_results['methods'][method] = method_result
                
                print(f"  Method {method}:")
                print(f"    Compensating set size: {method_result['compensating_size']}")
                print(f"    Validation error: {method_result['validation_error']:.2e}")
                print(f"    Balansis quality: {method_result['balansis_quality']:.4f}")
            
            results.append(case_results)
        
        # Comparative analysis
        self._analyze_compensation_quality(results)
        
        return results
    
    def _analyze_compensation_quality(self, results: List[Dict]):
        """Analysis of compensation quality study results."""
        methods = ['direct', 'iterative', 'adaptive']
        
        # Collecting metrics
        quality_scores = {method: [] for method in methods}
        error_rates = {method: [] for method in methods}
        
        for case_result in results:
            for method in methods:
                method_data = case_result['methods'][method]
                quality_scores[method].append(method_data['balansis_quality'])
                error_rates[method].append(method_data['validation_error'])
        
        # Statistical analysis
        print("\n=== Statistical Analysis of Compensation Quality ===")
        for method in methods:
            quality_mean = np.mean(quality_scores[method])
            quality_std = np.std(quality_scores[method])
            error_mean = np.mean(error_rates[method])
            error_std = np.std(error_rates[method])
            
            print(f"\nMethod {method}:")
            print(f"  Quality: {quality_mean:.4f} ± {quality_std:.4f}")
            print(f"  Error: {error_mean:.2e} ± {error_std:.2e}")
    
    def parallel_scalability_test(self, worker_counts: List[int], data_sizes: List[int]) -> Dict:
        """Parallel processing scalability test."""
        print("\n=== Parallel Processing Scalability Test ===")
        
        results = {}
        
        for data_size in data_sizes:
            print(f"\nTesting data size: {data_size}")
            
            # Creating test data
            test_sets = []
            for i in range(data_size):
                elements = [1/(n+i) for n in range(1, 101)]
                test_sets.append(ZeroSumInfiniteSet(elements))
            
            size_results = {}
            
            for worker_count in worker_counts:
                print(f"  Testing with {worker_count} workers...")
                
                # Creating parallel processor
                parallel_processor = ParallelTNSIM(
                    max_workers=worker_count,
                    chunk_size=max(1, data_size // (worker_count * 2))
                )
                
                # Measuring execution time
                start_time = time.time()
                parallel_results = parallel_processor.parallel_zero_sum(
                    test_sets, method='compensated'
                )
                end_time = time.time()
                
                execution_time = end_time - start_time
                throughput = data_size / execution_time
                
                size_results[worker_count] = {
                    'execution_time': execution_time,
                    'throughput': throughput,
                    'results_count': len(parallel_results)
                }
                
                print(f"    Time: {execution_time:.4f}s, Throughput: {throughput:.2f} ops/s")
            
            results[data_size] = size_results
        
        # Scalability analysis
        self._analyze_scalability(results, worker_counts, data_sizes)
        
        return results
    
    def _analyze_scalability(self, results: Dict, worker_counts: List[int], data_sizes: List[int]):
        """Analysis of scalability results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Execution time plot
        for data_size in data_sizes:
            times = [results[data_size][workers]['execution_time'] for workers in worker_counts]
            ax1.plot(worker_counts, times, marker='o', label=f'Size: {data_size}')
        
        ax1.set_xlabel('Number of workers')
        ax1.set_ylabel('Execution time (s)')
        ax1.set_title('Scalability: Execution time')
        ax1.legend()
        ax1.grid(True)
        
        # Throughput plot
        for data_size in data_sizes:
            throughputs = [results[data_size][workers]['throughput'] for workers in worker_counts]
            ax2.plot(worker_counts, throughputs, marker='s', label=f'Size: {data_size}')
        
        ax2.set_xlabel('Number of workers')
        ax2.set_ylabel('Throughput (ops/s)')
        ax2.set_title('Scalability: Throughput')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def attention_mechanism_study(self, configurations: List[Dict]) -> Dict:
        """Study of zero-sum attention mechanism."""
        print("\n=== Zero-sum attention mechanism study ===")
        
        results = []
        
        for config in configurations:
            print(f"\nConfiguration: {config['name']}")
            
            # Creating attention mechanism
            attention = ZeroSumAttention(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                compensation_strength=config.get('compensation_strength', 0.1)
            )
            
            # Test data
            batch_size, seq_len = config.get('batch_size', 4), config.get('seq_len', 32)
            x = torch.randn(batch_size, seq_len, config['d_model'])
            
            # Performance measurement
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(config.get('iterations', 100)):
                    output, attention_weights = attention(x)
            
            end_time = time.time()
            
            # Results analysis
            avg_time = (end_time - start_time) / config.get('iterations', 100)
            
            # Zero-sum properties check
            attention_sum = attention_weights.sum(dim=-1)
            sum_mean = attention_sum.mean().item()
            sum_std = attention_sum.std().item()
            
            # Weight distribution analysis
            weights_flat = attention_weights.flatten()
            weight_entropy = self._calculate_entropy(weights_flat)
            
            config_result = {
                'name': config['name'],
                'config': config,
                'avg_time': avg_time,
                'attention_sum_mean': sum_mean,
                'attention_sum_std': sum_std,
                'weight_entropy': weight_entropy,
                'output_shape': list(output.shape),
                'attention_shape': list(attention_weights.shape)
            }
            
            results.append(config_result)
            
            print(f"  Average time: {avg_time:.6f}s")
            print(f"  Weight sum: {sum_mean:.6f} ± {sum_std:.6f}")
            print(f"  Weight entropy: {weight_entropy:.4f}")
        
        # Comparative analysis
        self._analyze_attention_results(results)
        
        return results
    
    def _calculate_entropy(self, weights: torch.Tensor) -> float:
        """Calculate entropy of weight distribution."""
        weights_np = weights.detach().cpu().numpy()
        weights_positive = np.abs(weights_np) + 1e-10  # Avoid log(0)
        weights_normalized = weights_positive / np.sum(weights_positive)
        
        entropy = -np.sum(weights_normalized * np.log(weights_normalized))
        return float(entropy)
    
    def _analyze_attention_results(self, results: List[Dict]):
        """Analysis of attention mechanism study results."""
        print("\n=== Comparative Analysis of Attention Configurations ===")
        
        # Performance sorting
        results_sorted = sorted(results, key=lambda x: x['avg_time'])
        
        print("\nPerformance ranking:")
        for i, result in enumerate(results_sorted, 1):
            print(f"{i}. {result['name']}: {result['avg_time']:.6f}s")
        
        # Attention quality analysis
        print("\nAttention quality analysis:")
        for result in results:
            quality_score = 1.0 - abs(1.0 - result['attention_sum_mean'])
            consistency_score = 1.0 / (1.0 + result['attention_sum_std'])
            
            print(f"{result['name']}:")
            print(f"  Sum quality: {quality_score:.4f}")
            print(f"  Consistency: {consistency_score:.4f}")
            print(f"  Entropy: {result['weight_entropy']:.4f}")
    
    def memory_efficiency_analysis(self, test_scenarios: List[Dict]) -> Dict:
        """Memory usage efficiency analysis."""
        print("\n=== Memory Usage Efficiency Analysis ===")
        
        import psutil
        import gc
        
        results = []
        
        for scenario in test_scenarios:
            print(f"\nScenario: {scenario['name']}")
            
            # Measuring initial memory usage
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Executing scenario
            start_time = time.time()
            
            if scenario['type'] == 'large_sets':
                result = self._test_large_sets_memory(scenario['params'])
            elif scenario['type'] == 'parallel_processing':
                result = self._test_parallel_memory(scenario['params'])
            elif scenario['type'] == 'cache_usage':
                result = self._test_cache_memory(scenario['params'])
            elif scenario['type'] == 'attention_memory':
                result = self._test_attention_memory(scenario['params'])
            
            end_time = time.time()
            
            # Measuring peak memory usage
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Memory cleanup
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            scenario_result = {
                'name': scenario['name'],
                'type': scenario['type'],
                'execution_time': end_time - start_time,
                'initial_memory': initial_memory,
                'peak_memory': peak_memory,
                'final_memory': final_memory,
                'memory_increase': peak_memory - initial_memory,
                'memory_leaked': final_memory - initial_memory,
                'result': result
            }
            
            results.append(scenario_result)
            
            print(f"  Execution time: {scenario_result['execution_time']:.4f}s")
            print(f"  Memory increase: {scenario_result['memory_increase']:.2f} MB")
            print(f"  Memory leaked: {scenario_result['memory_leaked']:.2f} MB")
        
        # Results analysis
        self._analyze_memory_efficiency(results)
        
        return results
    
    def _test_large_sets_memory(self, params: Dict) -> Dict:
        """Memory test for large sets."""
        sets = []
        for i in range(params['num_sets']):
            elements = [1/(n+i) for n in range(1, params['set_size']+1)]
            zs_set = ZeroSumInfiniteSet(elements)
            sets.append(zs_set)
        
        # Performing operations
        results = []
        for zs_set in sets:
            result = zs_set.zero_sum_operation(method='compensated')
            results.append(result)
        
        return {'sets_created': len(sets), 'operations_completed': len(results)}
    
    def _test_parallel_memory(self, params: Dict) -> Dict:
        """Memory test for parallel processing."""
        sets = []
        for i in range(params['num_sets']):
            elements = [1/(n+i) for n in range(1, params['set_size']+1)]
            sets.append(ZeroSumInfiniteSet(elements))
        
        parallel_processor = ParallelTNSIM(
            max_workers=params['workers'],
            chunk_size=params['chunk_size']
        )
        
        results = parallel_processor.parallel_zero_sum(sets, method='compensated')
        
        return {'sets_processed': len(sets), 'results_count': len(results)}
    
    def _test_cache_memory(self, params: Dict) -> Dict:
        """Memory test for caching."""
        cache = TNSIMCache(
            max_size=params['cache_size'],
            eviction_strategy='lru'
        )
        
        # Filling cache
        for i in range(params['num_operations']):
            key = f"test_key_{i}"
            value = [1/(n+i) for n in range(1, params['value_size']+1)]
            cache.put(key, value)
        
        stats = cache.get_stats()
        return {'cache_size': stats['size'], 'operations': params['num_operations']}
    
    def _test_attention_memory(self, params: Dict) -> Dict:
        """Memory test for attention mechanism."""
        attention = ZeroSumAttention(
            d_model=params['d_model'],
            n_heads=params['n_heads']
        )
        
        batch_size, seq_len = params['batch_size'], params['seq_len']
        x = torch.randn(batch_size, seq_len, params['d_model'])
        
        outputs = []
        for _ in range(params['iterations']):
            output, _ = attention(x)
            outputs.append(output)
        
        return {'iterations_completed': len(outputs)}
    
    def _analyze_memory_efficiency(self, results: List[Dict]):
        """Memory usage efficiency analysis."""
        print("\n=== Memory Efficiency Analysis ===")
        
        # Sorting by efficiency
        efficiency_scores = []
        for result in results:
            # Efficiency = result / memory increase
            if result['memory_increase'] > 0:
                efficiency = result['execution_time'] / result['memory_increase']
            else:
                efficiency = float('inf')
            efficiency_scores.append((result['name'], efficiency, result))
        
        efficiency_scores.sort(key=lambda x: x[1])
        
        print("\nMemory efficiency ranking:")
        for i, (name, efficiency, result) in enumerate(efficiency_scores, 1):
            leak_status = "LEAK" if result['memory_leaked'] > 1.0 else "OK"
            print(f"{i}. {name}: {efficiency:.4f} (leaked: {result['memory_leaked']:.2f} MB) [{leak_status}]")
    
    def export_results(self, filename: str = "tnsim_analysis_results.json"):
        """Export analysis results."""
        results_data = {
            'timestamp': time.time(),
            'results_history': self.results_history,
            'cache_stats': self.cache.get_stats(),
            'system_info': {
                'python_version': '3.8+',
                'torch_version': torch.__version__,
                'numpy_version': np.__version__
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults exported to {filename}")


def create_test_series() -> List[Tuple[str, List[float]]]:
    """Creating test series for analysis."""
    return [
        ("Harmonic", [1/n for n in range(1, 201)]),
        ("Alternating", [(-1)**(n+1)/n for n in range(1, 201)]),
        ("Geometric (r=0.5)", [0.5**n for n in range(201)]),
        ("Quadratic", [1/(n**2) for n in range(1, 201)]),
        ("Logarithmic", [1/(n * np.log(n+1)) for n in range(1, 201)]),
        ("Exponential", [np.exp(-n/10) for n in range(1, 201)])
    ]


def create_compensation_test_cases() -> List[Dict]:
    """Creating test cases for compensation study."""
    return [
        {
            'name': 'Fast converging',
            'elements': [1/(n**3) for n in range(1, 101)],
            'max_iterations': 50
        },
        {
            'name': 'Slow converging',
            'elements': [1/(n * np.log(n+1)) for n in range(1, 201)],
            'max_iterations': 100
        },
        {
            'name': 'Oscillating',
            'elements': [(-1)**n / np.sqrt(n) for n in range(1, 151)],
            'max_iterations': 75
        },
        {
            'name': 'Mixed',
            'elements': [np.sin(n) / (n**2) for n in range(1, 101)],
            'max_iterations': 60
        }
    ]


def create_attention_configurations() -> List[Dict]:
    """Creating configurations for attention testing."""
    return [
        {
            'name': 'Small model',
            'd_model': 128,
            'n_heads': 4,
            'batch_size': 2,
            'seq_len': 16,
            'iterations': 50
        },
        {
            'name': 'Medium model',
            'd_model': 256,
            'n_heads': 8,
            'batch_size': 4,
            'seq_len': 32,
            'iterations': 30
        },
        {
            'name': 'Large model',
            'd_model': 512,
            'n_heads': 16,
            'batch_size': 2,
            'seq_len': 64,
            'iterations': 20
        }
    ]


def create_memory_test_scenarios() -> List[Dict]:
    """Creating scenarios for memory testing."""
    return [
        {
            'name': 'Large sets',
            'type': 'large_sets',
            'params': {
                'num_sets': 50,
                'set_size': 1000
            }
        },
        {
            'name': 'Parallel processing',
            'type': 'parallel_processing',
            'params': {
                'num_sets': 100,
                'set_size': 500,
                'workers': 4,
                'chunk_size': 10
            }
        },
        {
            'name': 'Intensive caching',
            'type': 'cache_usage',
            'params': {
                'cache_size': 1000,
                'num_operations': 2000,
                'value_size': 100
            }
        },
        {
            'name': 'Attention mechanism',
            'type': 'attention_memory',
            'params': {
                'd_model': 512,
                'n_heads': 8,
                'batch_size': 8,
                'seq_len': 128,
                'iterations': 50
            }
        }
    ]


def main():
    """Main function for running advanced examples."""
    print("TNSIM - Advanced Usage Examples")
    print("=" * 50)
    
    # Creating demonstration object
    demo = AdvancedTNSIMDemo()
    
    # Methods benchmark
    print("\n1. Running summation methods benchmark...")
    series_sizes = [100, 500, 1000, 2000]
    benchmark_results = demo.benchmark_methods(series_sizes)
    demo.results_history.append(('benchmark', benchmark_results))
    
    # Convergence analysis
    print("\n2. Comprehensive convergence analysis...")
    test_series = create_test_series()
    convergence_results = demo.convergence_analysis_suite(test_series)
    demo.results_history.append(('convergence', convergence_results))
    
    # Compensation quality study
    print("\n3. Compensation quality study...")
    test_cases = create_compensation_test_cases()
    compensation_results = demo.compensation_quality_study(test_cases)
    demo.results_history.append(('compensation', compensation_results))
    
    # Scalability test
    print("\n4. Parallel processing scalability test...")
    worker_counts = [1, 2, 4, 8]
    data_sizes = [50, 100, 200]
    scalability_results = demo.parallel_scalability_test(worker_counts, data_sizes)
    demo.results_history.append(('scalability', scalability_results))
    
    # Attention mechanism analysis
    print("\n5. Attention mechanism analysis...")
    attention_configs = create_attention_configurations()
    attention_results = demo.attention_mechanism_study(attention_configs)
    demo.results_history.append(('attention', attention_results))
    
    # Memory efficiency analysis
    print("\n6. Memory usage efficiency analysis...")
    memory_scenarios = create_memory_test_scenarios()
    memory_results = demo.memory_efficiency_analysis(memory_scenarios)
    demo.results_history.append(('memory', memory_results))
    
    # Results export
    print("\n7. Exporting results...")
    demo.export_results("advanced_tnsim_results.json")
    
    print("\n=== All advanced examples completed successfully! ===")
    print(f"Total number of tests: {len(demo.results_history)}")
    print("Results saved to files and plots.")


if __name__ == "__main__":
    main()