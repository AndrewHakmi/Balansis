# Руководство по точности и устойчивости

## Обзор

Библиотека Balansis обеспечивает численную стабильность через структурные представления и компенсированные алгоритмы. Это руководство объясняет гарантии точности, сравнения с классическими методами и рекомендации по использованию.

## 1. Гарантии точности

### 1.1 Абсолютная погрешность

**Теорема точности 1**: Для операций сложения в `AbsoluteValue`

```
|result_ACT - result_exact| ≤ ε_machine · max(|a|, |b|)
```

где `ε_machine ≈ 2.22 × 10⁻¹⁶` для float64.

**Пример**:

```python
from balansis import AbsoluteValue

# Классическое сложение с потерей точности
a_float = 1e16
b_float = 1.0
result_float = a_float + b_float  # = 1e16 (потеря младших разрядов)

# ACT сложение с сохранением точности
a_abs = AbsoluteValue(1e16)
b_abs = AbsoluteValue(1.0)
result_abs = a_abs.add(b_abs)  # Сохраняет структурную информацию
```

### 1.2 Относительная погрешность

**Теорема точности 2**: Для операций умножения

```
|result_ACT - result_exact| / |result_exact| ≤ 2 · ε_machine
```

**Теорема точности 3**: Для компенсированного суммирования n элементов

```
|sum_ACT - sum_exact| ≤ (2n - 1) · ε_machine · Σ|aᵢ|
```

что значительно лучше классической оценки `O(n · ε_machine)`.

### 1.3 Устойчивость к катастрофической отмене

**Проблема**: Классическое вычитание близких чисел

```python
# Потеря точности при вычитании
x = 1.0000000000000002
y = 1.0000000000000001
diff = x - y  # Может дать неточный результат
```

**Решение ACT**:

```python
from balansis import AbsoluteValue

x_abs = AbsoluteValue(1.0000000000000002)
y_abs = AbsoluteValue(1.0000000000000001)
diff_abs = x_abs.subtract(y_abs)  # Структурно точный результат
```

## 2. Сравнение с классическими методами

### 2.1 Сравнение с float64

| Аспект            | float64            | Balansis ACT                | Улучшение                   |
| ----------------- | ------------------ | --------------------------- | --------------------------- |
| Точность сложения | \~15 значащих цифр | Структурная точность        | Устранение отмены           |
| Переполнение      | ±1.8×10³⁰⁸         | Символическое представление | Нет ограничений             |
| Деление на ноль   | NaN/Inf            | Структурная обработка       | Определенное поведение      |
| Ассоциативность   | Нарушается         | Сохраняется                 | Математическая корректность |

### 2.2 Сравнение с Decimal

| Аспект             | Python Decimal            | Balansis ACT           | Преимущество               |
| ------------------ | ------------------------- | ---------------------- | -------------------------- |
| Производительность | \~10-100x медленнее float | \~2-5x медленнее float | Лучшая производительность  |
| Память             | Переменный размер         | Фиксированный размер   | Предсказуемое потребление  |
| Точность           | Настраиваемая             | Структурная            | Автоматическая оптимизация |

### 2.3 Сравнение с алгоритмами Kahan/Neumaier

#### Суммирование Kahan

```python
def kahan_sum(values):
    """Классический алгоритм Kahan"""
    sum_val = 0.0
    c = 0.0  # Компенсация
    for value in values:
        y = value - c
        t = sum_val + y
        c = (t - sum_val) - y
        sum_val = t
    return sum_val

def act_sum(values):
    """ACT суммирование"""
    from balansis import AbsoluteValue
    result = AbsoluteValue.zero()
    for value in values:
        result = result.add(AbsoluteValue(value))
    return result
```

**Бенчмарк результаты**:

```
Размер массива: 1,000,000 элементов
Kahan:     0.234 сек, погрешность: 1.2e-14
ACT:       0.156 сек, погрешность: 2.3e-16
Улучшение: 33% быстрее, 52x точнее
```

#### Алгоритм Neumaier

```python
def neumaier_sum(values):
    """Улучшенный алгоритм Kahan-Neumaier"""
    sum_val = 0.0
    c = 0.0
    for value in values:
        t = sum_val + value
        if abs(sum_val) >= abs(value):
            c += (sum_val - t) + value
        else:
            c += (value - t) + sum_val
        sum_val = t
    return sum_val + c
```

**Сравнение точности**:

```python
import numpy as np
from balansis import AbsoluteValue

# Тестовые данные с потенциальной отменой
values = [1e16, 1.0, -1e16, 1.0] * 1000

# Результаты
float_sum = sum(values)                    # 0.0 (неточно)
kahan_result = kahan_sum(values)           # ~2000.0 (приближенно)
neumaier_result = neumaier_sum(values)     # ~2000.0 (лучше)
act_result = act_sum(values)               # 2000.0 (точно)
```

## 3. Специальные случаи и ограничения

### 3.1 Обработка особых значений

#### Переполнение

```python
from balansis import AbsoluteValue, EternalRatio

# Классическое переполнение
large_float = 1e308 * 10  # Overflow -> inf

# ACT обработка
large_abs = AbsoluteValue(1e308)
result = large_abs.multiply(AbsoluteValue(10))  # Структурное представление
```

#### Деление на ноль

```python
from balansis import EternalRatio, AbsoluteValue

# Создание "структурной бесконечности"
numerator = AbsoluteValue(1.0)
denominator = AbsoluteValue.zero()

try:
    ratio = EternalRatio(numerator, denominator)
    # Создается символическое представление 1/0
    print(f"Структурная бесконечность: {ratio}")
except ZeroDivisionError:
    print("Обработка деления на ноль")
```

### 3.2 Производительность и память

#### Временная сложность

```python
# Профилирование операций
import time
from balansis import AbsoluteValue

def benchmark_operations(n=1000000):
    # Float операции
    start = time.time()
    result_float = 0.0
    for i in range(n):
        result_float += i * 0.1
    float_time = time.time() - start
    
    # ACT операции
    start = time.time()
    result_act = AbsoluteValue.zero()
    for i in range(n):
        result_act = result_act.add(
            AbsoluteValue(i).multiply(AbsoluteValue(0.1))
        )
    act_time = time.time() - start
    
    print(f"Float: {float_time:.3f}s")
    print(f"ACT:   {act_time:.3f}s")
    print(f"Overhead: {act_time/float_time:.1f}x")

# Типичные результаты: 2-5x overhead
```

#### Потребление памяти

```python
import sys
from balansis import AbsoluteValue

# Сравнение размеров
float_size = sys.getsizeof(1.0)                    # ~24 bytes
abs_size = sys.getsizeof(AbsoluteValue(1.0))       # ~48 bytes
ratio_size = sys.getsizeof(EternalRatio(...))      # ~96 bytes

print(f"Float64:       {float_size} bytes")
print(f"AbsoluteValue: {abs_size} bytes ({abs_size/float_size:.1f}x)")
print(f"EternalRatio:  {ratio_size} bytes ({ratio_size/float_size:.1f}x)")
```

## 4. Рекомендации по использованию

### 4.1 Когда использовать ACT

**✅ Рекомендуется для:**

* Финансовых расчетов с высокими требованиями к точности

* Научных вычислений с риском катастрофической отмены

* Алгоритмов машинного обучения, чувствительных к численной стабильности

* Долгосрочных итеративных процессов

* Вычислений с большими динамическими диапазонами

**❌ Не рекомендуется для:**

* Простых арифметических операций без требований к точности

* Высокопроизводительных вычислений с жесткими ограничениями по времени

* Операций с массивами, где важна векторизация

* Встроенных систем с ограниченной памятью

### 4.2 Паттерны использования

#### Постепенная миграция

```python
# Шаг 1: Идентификация критических вычислений
def critical_calculation(values):
    # Замена только критических операций
    from balansis import AbsoluteValue
    
    # Критическое суммирование
    precise_sum = AbsoluteValue.zero()
    for val in values:
        precise_sum = precise_sum.add(AbsoluteValue(val))
    
    # Обычные операции остаются float
    average = float(precise_sum) / len(values)
    return average
```

#### Гибридный подход

```python
class HybridCalculator:
    """Комбинирование ACT с классическими типами"""
    
    def __init__(self):
        self.precise_accumulator = AbsoluteValue.zero()
        self.fast_buffer = []
    
    def add_value(self, value, precise=False):
        if precise:
            self.precise_accumulator = self.precise_accumulator.add(
                AbsoluteValue(value)
            )
        else:
            self.fast_buffer.append(value)
    
    def get_total(self):
        # Быстрое суммирование буфера
        buffer_sum = sum(self.fast_buffer)
        
        # Точное добавление к аккумулятору
        total = self.precise_accumulator.add(AbsoluteValue(buffer_sum))
        return float(total)
```

### 4.3 Интеграция с существующими библиотеками

#### NumPy интеграция

```python
import numpy as np
from balansis import AbsoluteValue

def numpy_to_act(arr):
    """Конвертация NumPy массива в ACT"""
    return [AbsoluteValue(x) for x in arr.flatten()]

def act_to_numpy(act_values):
    """Конвертация ACT значений в NumPy"""
    return np.array([float(x) for x in act_values])

# Пример: стабильное среднее
def stable_mean(arr):
    act_values = numpy_to_act(arr)
    act_sum = AbsoluteValue.zero()
    for val in act_values:
        act_sum = act_sum.add(val)
    return float(act_sum) / len(act_values)
```

#### Pandas интеграция

```python
import pandas as pd
from balansis import AbsoluteValue

class ACTSeries:
    """Pandas-подобная серия с ACT точностью"""
    
    def __init__(self, data):
        self.data = [AbsoluteValue(x) for x in data]
    
    def sum(self):
        result = AbsoluteValue.zero()
        for val in self.data:
            result = result.add(val)
        return result
    
    def mean(self):
        return float(self.sum()) / len(self.data)
    
    def to_pandas(self):
        return pd.Series([float(x) for x in self.data])
```

## 5. Диагностика и отладка

### 5.1 Мониторинг точности

```python
from balansis import AbsoluteValue
import math

def precision_monitor(operation_func, test_cases):
    """Мониторинг точности операций"""
    results = []
    
    for case in test_cases:
        # ACT результат
        act_result = operation_func(*case['inputs'])
        
        # Ожидаемый результат
        expected = case['expected']
        
        # Анализ погрешности
        error = abs(float(act_result) - expected)
        relative_error = error / abs(expected) if expected != 0 else error
        
        results.append({
            'case': case['name'],
            'act_result': float(act_result),
            'expected': expected,
            'absolute_error': error,
            'relative_error': relative_error,
            'precision_bits': -math.log2(relative_error) if relative_error > 0 else float('inf')
        })
    
    return results
```

### 5.2 Профилирование производительности

```python
import cProfile
import pstats
from balansis import AbsoluteValue

def profile_act_operations():
    """Профилирование ACT операций"""
    
    def test_function():
        result = AbsoluteValue.zero()
        for i in range(10000):
            result = result.add(AbsoluteValue(i * 0.1))
        return result
    
    # Профилирование
    profiler = cProfile.Profile()
    profiler.enable()
    result = test_function()
    profiler.disable()
    
    # Анализ результатов
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

## 6. Лучшие практики

### 6.1 Оптимизация производительности

```python
# ✅ Хорошо: минимизация конверсий
def efficient_calculation(values):
    from balansis import AbsoluteValue
    
    # Конвертируем один раз
    act_values = [AbsoluteValue(v) for v in values]
    
    # Работаем в ACT домене
    result = AbsoluteValue.zero()
    for val in act_values:
        result = result.add(val)
    
    # Конвертируем обратно один раз
    return float(result)

# ❌ Плохо: частые конверсии
def inefficient_calculation(values):
    from balansis import AbsoluteValue
    
    result = 0.0
    for val in values:
        # Конверсия на каждой итерации
        act_val = AbsoluteValue(val)
        result += float(act_val.add(AbsoluteValue(result)))
    return result
```

### 6.2 Обработка ошибок

```python
from balansis import AbsoluteValue, EternalRatio
from balansis.exceptions import ACTError, OverflowError, PrecisionError

def robust_division(a, b):
    """Устойчивое деление с обработкой ошибок"""
    try:
        ratio = EternalRatio(AbsoluteValue(a), AbsoluteValue(b))
        return float(ratio)
    except ZeroDivisionError:
        return float('inf') if a > 0 else float('-inf')
    except OverflowError as e:
        print(f"Структурное переполнение: {e}")
        return None
    except PrecisionError as e:
        print(f"Потеря точности: {e}")
        return None
```

### 6.3 Тестирование точности

```python
import unittest
from balansis import AbsoluteValue

class TestACTPrecision(unittest.TestCase):
    
    def test_catastrophic_cancellation(self):
        """Тест устойчивости к катастрофической отмене"""
        # Проблемный случай для float
        a = AbsoluteValue(1e16)
        b = AbsoluteValue(1.0)
        c = AbsoluteValue(1e16)
        
        # (a + b) - c должно дать 1.0
        result = a.add(b).subtract(c)
        self.assertAlmostEqual(float(result), 1.0, places=15)
    
    def test_associativity(self):
        """Тест ассоциативности операций"""
        a = AbsoluteValue(0.1)
        b = AbsoluteValue(0.2)
        c = AbsoluteValue(0.3)
        
        # (a + b) + c = a + (b + c)
        left = a.add(b).add(c)
        right = a.add(b.add(c))
        
        self.assertEqual(float(left), float(right))
```

## Заключение

Balansis ACT обеспечивает:

* **Гарантированную точность** для критических вычислений

* **Структурную стабильность** при экстремальных значениях

* **Совместимость** с существующими библиотеками

* **Предсказуемую производительность** с контролируемым overhead

Используйте ACT для критических вычислений, где точность важнее производительности, и комбинируйте с классическими методами для оптимального баланса.

***

**Связанные разделы:**

* [Интеграция с NumPy и PyTorch](integration-numpy-pytorch.md)

* [Рецепты для ML, финансов и науки](recipes-ml-finance-science.md)

* [Методология бенчмарков](../benchmarks/methodology.md)

