"""Database module for TNSIM."""

from .config import DatabaseConfig, db_config, get_config, EnvironmentConfig
from .repository import DatabaseConnection, InfiniteSetRepository

__all__ = [
    'DatabaseConfig',
    'db_config', 
    'get_config',
    'EnvironmentConfig',
    'DatabaseConnection',
    'InfiniteSetRepository'
]

__version__ = '1.0.0'
__author__ = 'TNSIM Team'
__description__ = 'Database layer for Zero Sum Theory of Infinite Sets'