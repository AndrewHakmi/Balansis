from balansis.core.absolute import AbsoluteValue
from balansis.algebra.absolute_group import AbsoluteGroup, GroupElement
from balansis.logic.compensator import Compensator

# Test direct AbsoluteValue addition
value1 = AbsoluteValue(magnitude=4.0, direction=1)
value2 = AbsoluteValue(magnitude=4.0, direction=-1)

print(f"Value1: {value1}")
print(f"Value2: {value2}")

# Direct addition
direct_sum = value1 + value2
print(f"Direct sum: {direct_sum}")
print(f"Direct sum is_absolute: {direct_sum.is_absolute()}")

# Test compensator
compensator = Compensator()
compensated_sum = compensator.compensate_addition(value1, value2)
print(f"\nCompensated sum: {compensated_sum}")
print(f"Compensated sum is_absolute: {compensated_sum.is_absolute()}")

# Test with group
group = AbsoluteGroup.additive_group()
element1 = GroupElement(value=value1)
element2 = GroupElement(value=value2)

result = group.operate(element1, element2)
print(f"\nGroup operation result: {result}")
print(f"Group result value: {result.value}")
print(f"Group result is_absolute: {result.value.is_absolute()}")