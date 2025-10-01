"""Core mathematical types and operations for Balansis library.

This module contains the fundamental mathematical constructs of Absolute Compensation Theory:
- AbsoluteValue: Core value type with magnitude and direction
- EternalRatio: Structural ratios between AbsoluteValues
- Operations: Basic mathematical operations following ACT principles
"""

from .absolute import AbsoluteValue
from .eternity import EternalRatio
from .operations import Operations

__all__ = ["AbsoluteValue", "EternalRatio", "Operations"]