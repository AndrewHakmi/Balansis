# Balansis API Reference

Полная документация API для библиотеки Balansis - Python реализации теории абсолютной компенсации (ACT).

## Обзор

Balansis предоставляет новый математический фреймворк, заменяющий традиционные понятия нуля и бесконечности концепциями Absolute и Eternity для повышения вычислительной стабильности.

## Основные модули

### [Core Module](core/index.md)
Основные математические типы и операции:
- [`AbsoluteValue`](core/absolute_value.md) - Значения с величиной и направлением
- [`EternalRatio`](core/eternal_ratio.md) - Структурные отношения между AbsoluteValue
- [`Operations`](core/operations.md) - Компенсированные арифметические операции

### [Logic Module](logic/index.md)
Логика компенсации и стабилизации:
- [`Compensator`](logic/compensator.md) - Движок компенсации для поддержания стабильности

### [Algebra Module](algebra/index.md)
Алгебраические структуры:
- [`AbsoluteGroup`](algebra/absolute_group.md) - Групповые операции для Absolute значений
- [`EternityField`](algebra/eternity_field.md) - Полевые операции для вечных отношений

### [Utils Module](utils/index.md)
Утилиты и вспомогательные функции:
- [`PlotUtils`](utils/plot_utils.md) - Визуализация математических структур

## Константы и настройки

### ACT Константы
```python
from balansis import (
    ABSOLUTE,              # AbsoluteValue(magnitude=0.0, direction=1)
    UNIT_POSITIVE,         # AbsoluteValue(magnitude=1.0, direction=1)
    UNIT_NEGATIVE,         # AbsoluteValue(magnitude=1.0, direction=-1)
    DEFAULT_TOLERANCE,     # 1e-10
    STABILITY_THRESHOLD,   # 1e-8
    ACT_EPSILON,          # 1e-15
    ACT_STABILITY_THRESHOLD, # 1e-12
    ACT_ABSOLUTE_THRESHOLD,  # 1e-20
    ACT_COMPENSATION_FACTOR  # 0.1
)
```

## Быстрый старт

### Базовое использование
```python
from balansis import AbsoluteValue, EternalRatio, Operations

# Создание AbsoluteValue
a = AbsoluteValue(magnitude=5.0, direction=1)
b = AbsoluteValue(magnitude=3.0, direction=-1)

# Компенсированные операции
result = a + b  # Автоматическая компенсация
ratio = EternalRatio(numerator=a, denominator=b)

# Использование Operations для сложных вычислений
ops = Operations()
compensated_sum = ops.compensated_add([a, b])
```

### Работа с алгебраическими структурами
```python
from balansis import AbsoluteGroup, EternityField

# Групповые операции
group = AbsoluteGroup()
group.add_element(a)
group.add_element(b)
group_result = group.operate(a, b)

# Полевые операции
field = EternityField()
ratio1 = EternalRatio(numerator=a, denominator=b)
ratio2 = EternalRatio(numerator=b, denominator=a)
field_result = field.multiply(ratio1, ratio2)
```

## Примеры использования

- [Базовые операции с AbsoluteValue](../examples/basic_operations.md)
- [Работа с EternalRatio](../examples/eternal_ratios.md)
- [Алгебраические структуры](../examples/algebraic_structures.md)
- [Компенсация и стабилизация](../examples/compensation.md)
- [Визуализация](../examples/visualization.md)

## Интеграция с другими библиотеками

- [NumPy интеграция](../integration/numpy.md)
- [Pandas интеграция](../integration/pandas.md)
- [PyTorch интеграция](../integration/pytorch.md)
- [SciPy интеграция](../integration/scipy.md)

## Производительность и бенчмарки

- [Сравнение производительности](../benchmarks/performance.md)
- [Точность вычислений](../benchmarks/accuracy.md)
- [Стабильность операций](../benchmarks/stability.md)

## Версионирование

Текущая версия: **0.1.0**

Balansis следует [семантическому версионированию](https://semver.org/). См. [CHANGELOG](../../CHANGELOG.md) для деталей изменений.