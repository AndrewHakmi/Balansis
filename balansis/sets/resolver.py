import itertools
import math
from typing import Iterator
from balansis.core.absolute import AbsoluteValue
from balansis.sets.eternal_set import EternalSet

def global_compensate(set_a: EternalSet, set_b: EternalSet) -> EternalSet:
    zero = AbsoluteValue.absolute()
    def generator() -> Iterator[AbsoluteValue]:
        for a, b in itertools.zip_longest(set_a, set_b, fillvalue=zero):
            r = a + b
            if hasattr(r, "is_absolute"):
                if r.is_absolute():
                    if set_a.is_infinite or set_b.is_infinite:
                        yield zero
                    else:
                        continue
            else:
                if math.isclose(r.magnitude, 0.0, rel_tol=1e-12, abs_tol=1e-12):
                    if set_a.is_infinite or set_b.is_infinite:
                        yield zero
                    else:
                        continue
            yield r
    return EternalSet(generator(), is_infinite=(set_a.is_infinite or set_b.is_infinite), rule_name="global_compensate")

def verify_zero_sum(result_set: EternalSet, threshold: int = 1000):
    residuals = []
    it = iter(result_set)
    for _ in range(int(threshold)):
        try:
            x = next(it)
        except StopIteration:
            break
        if hasattr(x, "is_absolute"):
            if x.is_absolute():
                continue
        else:
            if math.isclose(x.magnitude, 0.0, rel_tol=1e-12, abs_tol=1e-12):
                continue
        residuals.append(x)
    return residuals
