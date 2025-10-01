"""TNSIM integrations with external libraries."""

from .balansis_integration import ZeroSumAttention, BalansisCompensator

__all__ = [
    'ZeroSumAttention',
    'BalansisCompensator'
]

__version__ = '1.0.0'
__author__ = 'TNSIM Team'
__description__ = 'External library integrations for Zero Sum Theory of Infinite Sets'