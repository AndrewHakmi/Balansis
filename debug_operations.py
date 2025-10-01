from balansis.core.absolute_value import AbsoluteValue
from balansis.core.operations import Operations
import math

# Test compensated_power with large values
print("Testing compensated_power with large values:")
large_base = AbsoluteValue(magnitude=1e50, direction=1)
result, comp = Operations.compensated_power(large_base, 10.0)
print(f"Result magnitude: {result.magnitude}, compensation: {comp}")

# Test compensated_power with zero magnitude and negative exponent
print("\nTesting compensated_power with zero magnitude and negative exponent:")
zero_base = AbsoluteValue(magnitude=0.0, direction=1)
try:
    result2, comp2 = Operations.compensated_power(zero_base, -2.0)
    print(f"Zero result magnitude: {result2.magnitude}, compensation: {comp2}, is_absolute: {result2.is_absolute()}")
except Exception as e:
    print(f"Exception: {e}")

# Test compensated_exp with large values
print("\nTesting compensated_exp with large values:")
large_val = AbsoluteValue(magnitude=200.0, direction=1)
result3, comp3 = Operations.compensated_exp(large_val)
print(f"Exp result magnitude: {result3.magnitude}, compensation: {comp3}")