import numpy as np
from balansis.core.absolute import AbsoluteValue
from balansis.numpy_integration import to_numpy, from_numpy, ufunc_add, ufunc_log

def test_to_from_numpy_roundtrip():
    values = [AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(-2.0)]
    arr = to_numpy(values)
    back = from_numpy(arr)
    assert back[0] == values[0]
    assert back[1] == values[1]

def test_numpy_ufunc_add():
    a = np.array([AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(2.0)], dtype=object)
    b = np.array([AbsoluteValue.from_float(3.0), AbsoluteValue.from_float(-1.0)], dtype=object)
    c = ufunc_add(a, b)
    assert c[0].to_float() == 4.0
    assert c[1].to_float() == 1.0

def test_numpy_ufunc_log():
    a = np.array([AbsoluteValue.from_float(1.0), AbsoluteValue.from_float(4.0)], dtype=object)
    l = ufunc_log(a)
    assert abs(l[0] - 0.0) < 1e-12
    assert abs(l[1] - np.log(4.0)) < 1e-12
