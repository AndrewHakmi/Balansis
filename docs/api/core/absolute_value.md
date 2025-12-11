# AbsoluteValue API Reference

## Обзор

`AbsoluteValue` - это основной математический тип в библиотеке Balansis, который заменяет традиционный ноль стабильным представлением с величиной и направлением. Это позволяет выполнять компенсированные операции, избегая математических нестабильностей.

## Класс AbsoluteValue

```python
class AbsoluteValue(BaseModel):
    """Основной математический тип, представляющий значения с величиной и направлением."""
    
    magnitude: float  # Неотрицательная величина
    direction: Literal[-1, 1]  # Направление: +1 или -1
```

### Атрибуты

#### `magnitude: float`
- **Описание**: Неотрицательное число с плавающей точкой, представляющее размер значения
- **Ограничения**: Должно быть конечным и неотрицательным (≥ 0)
- **Валидация**: Автоматически проверяется на конечность и неотрицательность

#### `direction: Literal[-1, 1]`
- **Описание**: Индикатор направления/знака
- **Значения**: 
  - `+1` для положительных значений
  - `-1` для отрицательных значений
- **Валидация**: Должно быть точно +1 или -1

### Конструктор

```python
def __init__(self, magnitude: float, direction: Literal[-1, 1]) -> None:
    """Создает новый AbsoluteValue.
    
    Args:
        magnitude: Неотрицательная величина значения
        direction: Направление (+1 или -1)
        
    Raises:
        ValueError: Если magnitude отрицательная или не конечная
        ValueError: Если direction не равно +1 или -1
    """
```

**Примеры:**
```python
from balansis import AbsoluteValue

# Абсолютный ноль
absolute_zero = AbsoluteValue(magnitude=0.0, direction=1)

# Положительное значение
positive_five = AbsoluteValue(magnitude=5.0, direction=1)

# Отрицательное значение
negative_three = AbsoluteValue(magnitude=3.0, direction=-1)
```

## Арифметические операции

### Сложение (`+`)

```python
def __add__(self, other: AbsoluteValue) -> AbsoluteValue:
    """Компенсированное сложение по принципам ACT."""
```

**Правила сложения:**
- **Одинаковое направление**: величины складываются
- **Разные направления**: величины вычитаются, направление большей величины побеждает
- **Равные величины, разные направления**: результат - Absolute (magnitude=0)

**Примеры:**
```python
a = AbsoluteValue(magnitude=5.0, direction=1)   # +5
b = AbsoluteValue(magnitude=3.0, direction=1)   # +3
result1 = a + b  # AbsoluteValue(magnitude=8.0, direction=1)  # +8

c = AbsoluteValue(magnitude=5.0, direction=1)   # +5
d = AbsoluteValue(magnitude=3.0, direction=-1)  # -3
result2 = c + d  # AbsoluteValue(magnitude=2.0, direction=1)  # +2

e = AbsoluteValue(magnitude=5.0, direction=1)   # +5
f = AbsoluteValue(magnitude=5.0, direction=-1)  # -5
result3 = e + f  # AbsoluteValue(magnitude=0.0, direction=1)  # Absolute
```

### Вычитание (`-`)

```python
def __sub__(self, other: AbsoluteValue) -> AbsoluteValue:
    """Компенсированное вычитание."""
```

Реализовано как сложение с инвертированным направлением второго операнда.

**Пример:**
```python
a = AbsoluteValue(magnitude=5.0, direction=1)   # +5
b = AbsoluteValue(magnitude=3.0, direction=1)   # +3
result = a - b  # AbsoluteValue(magnitude=2.0, direction=1)  # +2
```

### Скалярное умножение (`*`)

```python
def __mul__(self, scalar: Union[float, int]) -> AbsoluteValue:
    """Скалярное умножение."""

def __rmul__(self, scalar: Union[float, int]) -> AbsoluteValue:
    """Правое скалярное умножение (коммутативное)."""
```

**Правила:**
- Величина умножается на абсолютное значение скаляра
- Направление инвертируется, если скаляр отрицательный

**Примеры:**
```python
a = AbsoluteValue(magnitude=3.0, direction=1)   # +3
result1 = a * 2      # AbsoluteValue(magnitude=6.0, direction=1)   # +6
result2 = a * -2     # AbsoluteValue(magnitude=6.0, direction=-1)  # -6
result3 = 2.5 * a    # AbsoluteValue(magnitude=7.5, direction=1)   # +7.5
```

### Скалярное деление (`/`)

```python
def __truediv__(self, scalar: Union[float, int]) -> AbsoluteValue:
    """Скалярное деление."""
```

**Правила:**
- Деление на ноль запрещено (ValueError)
- Скаляр должен быть конечным

**Пример:**
```python
a = AbsoluteValue(magnitude=6.0, direction=1)   # +6
result = a / 2  # AbsoluteValue(magnitude=3.0, direction=1)  # +3
```

### Унарные операции

#### Отрицание (`-`)

```python
def __neg__(self) -> AbsoluteValue:
    """Унарное отрицание (инверсия направления)."""
```

**Пример:**
```python
a = AbsoluteValue(magnitude=5.0, direction=1)   # +5
result = -a  # AbsoluteValue(magnitude=5.0, direction=-1)  # -5
```

#### Абсолютное значение (`abs()`)

```python
def __abs__(self) -> AbsoluteValue:
    """Возвращает абсолютное значение (положительное направление)."""
```

**Пример:**
```python
a = AbsoluteValue(magnitude=5.0, direction=-1)  # -5
result = abs(a)  # AbsoluteValue(magnitude=5.0, direction=1)  # +5
```

## Операции сравнения

### Равенство (`==`)

```python
def __eq__(self, other: Any) -> bool:
    """Сравнение на равенство."""
```

Два AbsoluteValue равны, если равны их величина и направление.

### Сравнение по величине (`<`, `<=`, `>`, `>=`)

```python
def __lt__(self, other: AbsoluteValue) -> bool:
    """Сравнение "меньше чем" на основе знакового значения."""
```

Сравнение основано на знаковом значении (magnitude × direction).

**Примеры:**
```python
a = AbsoluteValue(magnitude=3.0, direction=1)   # +3
b = AbsoluteValue(magnitude=5.0, direction=1)   # +5
c = AbsoluteValue(magnitude=2.0, direction=-1)  # -2

print(a < b)   # True  (+3 < +5)
print(c < a)   # True  (-2 < +3)
print(a == a)  # True
```

## Методы проверки состояния

### `is_absolute()`

```python
def is_absolute(self) -> bool:
    """Проверяет, является ли значение Absolute (magnitude = 0)."""
```

**Пример:**
```python
absolute_zero = AbsoluteValue(magnitude=0.0, direction=1)
regular_value = AbsoluteValue(magnitude=5.0, direction=1)

print(absolute_zero.is_absolute())  # True
print(regular_value.is_absolute())  # False
```

### `is_positive()` / `is_negative()`

```python
def is_positive(self) -> bool:
    """Проверяет, является ли направление положительным."""

def is_negative(self) -> bool:
    """Проверяет, является ли направление отрицательным."""
```

**Примеры:**
```python
positive = AbsoluteValue(magnitude=5.0, direction=1)
negative = AbsoluteValue(magnitude=3.0, direction=-1)

print(positive.is_positive())  # True
print(negative.is_negative())  # True
```

## Методы преобразования

### `to_float()`

```python
def to_float(self) -> float:
    """Преобразует в обычное число с плавающей точкой."""
```

**Пример:**
```python
a = AbsoluteValue(magnitude=5.0, direction=-1)
float_value = a.to_float()  # -5.0
```

### `to_complex()`

```python
def to_complex(self) -> complex:
    """Преобразует в комплексное число."""
```

### Строковое представление

```python
def __str__(self) -> str:
    """Человекочитаемое строковое представление."""

def __repr__(self) -> str:
    """Техническое строковое представление."""
```

**Примеры:**
```python
a = AbsoluteValue(magnitude=5.0, direction=1)
print(str(a))   # "+5.0"
print(repr(a))  # "AbsoluteValue(magnitude=5.0, direction=1)"

absolute = AbsoluteValue(magnitude=0.0, direction=1)
print(str(absolute))  # "Absolute"
```

## Константы

```python
from balansis import ABSOLUTE, UNIT_POSITIVE, UNIT_NEGATIVE

# Предопределенные константы
ABSOLUTE = AbsoluteValue(magnitude=0.0, direction=1)      # Абсолютный ноль
UNIT_POSITIVE = AbsoluteValue(magnitude=1.0, direction=1)  # +1
UNIT_NEGATIVE = AbsoluteValue(magnitude=1.0, direction=-1) # -1
```

## Исключения

### `ValueError`
Возникает в следующих случаях:
- Отрицательная или не конечная величина
- Неверное направление (не +1 и не -1)
- Деление на ноль
- Умножение/деление на не конечный скаляр

## Примеры использования

### Базовые операции
```python
from balansis import AbsoluteValue

# Создание значений
a = AbsoluteValue(magnitude=10.0, direction=1)   # +10
b = AbsoluteValue(magnitude=3.0, direction=-1)   # -3

# Арифметические операции
sum_result = a + b      # +7
diff_result = a - b     # +13
scaled = a * 0.5        # +5
divided = a / 2         # +5

# Проверки
print(f"a положительное: {a.is_positive()}")  # True
print(f"b отрицательное: {b.is_negative()}")  # True
print(f"Сумма: {sum_result}")                  # +7.0
```

### Работа с компенсацией
```python
# Демонстрация компенсации при равных величинах
x = AbsoluteValue(magnitude=5.0, direction=1)   # +5
y = AbsoluteValue(magnitude=5.0, direction=-1)  # -5

compensated = x + y  # Результат: Absolute (magnitude=0.0, direction=1)
print(f"Компенсированный результат: {compensated}")  # Absolute
print(f"Это Absolute: {compensated.is_absolute()}")  # True
```

### Цепочка операций
```python
# Сложные вычисления с автоматической компенсацией
values = [
    AbsoluteValue(magnitude=2.0, direction=1),   # +2
    AbsoluteValue(magnitude=3.0, direction=-1),  # -3
    AbsoluteValue(magnitude=1.0, direction=1),   # +1
    AbsoluteValue(magnitude=4.0, direction=1),   # +4
]

# Последовательное сложение
result = values[0]
for val in values[1:]:
    result = result + val

print(f"Итоговый результат: {result}")  # +4.0
```