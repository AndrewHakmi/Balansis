#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations
import pytest

print("Testing compensated_power with zero magnitude and negative exponent:")
try:
    zero_base = AbsoluteValue(magnitude=0.0, direction=1)
    result, compensation = Operations.compensated_power(zero_base, -2.0)
    print(f"Result: {result}, Compensation: {compensation}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting compensated_exp with large value:")
try:
    large_val = AbsoluteValue(magnitude=200.0, direction=1)
    result, compensation = Operations.compensated_exp(large_val)
    print(f"Result magnitude: {result.magnitude}, Compensation: {compensation}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")