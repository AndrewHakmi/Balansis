"""
Balansis Benchmarks Package

Этот пакет содержит бенчмарки для сравнения производительности и точности
теории абсолютной компенсации (ACT) с классическими численными методами.

Модули:
- accuracy_benchmarks: Тесты точности вычислений
- performance_benchmarks: Тесты производительности
- stability_benchmarks: Тесты стабильности
- visualization: Визуализация результатов
- utils: Вспомогательные функции
"""

from .accuracy_benchmarks import AccuracyBenchmark
from .performance_benchmarks import PerformanceBenchmark
from .stability_benchmarks import StabilityBenchmark
from .visualization import BenchmarkVisualizer
from .utils import BenchmarkUtils

__all__ = [
    'AccuracyBenchmark',
    'PerformanceBenchmark', 
    'StabilityBenchmark',
    'BenchmarkVisualizer',
    'BenchmarkUtils'
]

__version__ = "0.1.0"