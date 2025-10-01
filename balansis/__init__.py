"""Balansis: Python mathematical library implementing Absolute Compensation Theory (ACT).

This library provides a novel framework replacing traditional zero and infinity
with Absolute and Eternity concepts for enhanced computational stability.

Core Components:
    - AbsoluteValue: Values with magnitude and direction
    - EternalRatio: Structural ratios between AbsoluteValues
    - Operations: Compensated arithmetic operations
    - Compensator: Balance and stability calculations
    - AbsoluteGroup: Group theory for Absolute values
    - EternityField: Field operations for eternal ratios

Example:
    >>> from balansis import AbsoluteValue, EternalRatio
    >>> a = AbsoluteValue(magnitude=5.0, direction=1)
    >>> b = AbsoluteValue(magnitude=3.0, direction=-1)
    >>> result = a + b  # Compensated addition
    >>> ratio = EternalRatio(numerator=a, denominator=b)
"""

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.core.operations import Operations
from balansis.logic.compensator import Compensator
from balansis.algebra.absolute_group import AbsoluteGroup
from balansis.algebra.eternity_field import EternityField
# from balansis.utils.plot import PlotUtils  # Temporarily disabled

__version__ = "0.1.0"
__author__ = "Balansis Team"
__email__ = "team@balansis.org"

# ACT Constants
ABSOLUTE = AbsoluteValue(magnitude=0.0, direction=1)
UNIT_POSITIVE = AbsoluteValue(magnitude=1.0, direction=1)
UNIT_NEGATIVE = AbsoluteValue(magnitude=1.0, direction=-1)

# Mathematical limits and tolerances
DEFAULT_TOLERANCE = 1e-10
STABILITY_THRESHOLD = 1e-8
MAX_MAGNITUDE = 1e308
MIN_MAGNITUDE = 1e-308

# ACT-specific constants
ACT_EPSILON = 1e-15
ACT_STABILITY_THRESHOLD = 1e-12
ACT_ABSOLUTE_THRESHOLD = 1e-20
ACT_COMPENSATION_FACTOR = 0.1

__all__ = [
    "AbsoluteValue",
    "EternalRatio",
    "Operations",
    "Compensator",
    "AbsoluteGroup",
    "EternityField",
    # "PlotUtils",  # Temporarily disabled
    "ABSOLUTE",
    "UNIT_POSITIVE",
    "UNIT_NEGATIVE",
    "DEFAULT_TOLERANCE",
    "STABILITY_THRESHOLD",
    "MAX_MAGNITUDE",
    "MIN_MAGNITUDE",
    "ACT_EPSILON",
    "ACT_STABILITY_THRESHOLD",
    "ACT_ABSOLUTE_THRESHOLD",
    "ACT_COMPENSATION_FACTOR",
]