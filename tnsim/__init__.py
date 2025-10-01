"""TNSIM (Theory of Zero-Sum Infinite Sets).

Library for working with compensated infinite sets,
implementing principles of zero-sum and numerical stability.

Main components:
- ZeroSumInfiniteSet: Class for working with infinite sets
- TNSIMCache: Operation caching system
- ParallelTNSIM: Parallel computations
- ZeroSumAttention: Neural network integration
- FastAPI application for web interface

Usage example:
    >>> from tnsim import ZeroSumInfiniteSet
    >>> zs_set = ZeroSumInfiniteSet.create_harmonic_series(100)
    >>> result = zs_set.zero_sum_operation()
    >>> print(f"Result: {result}")
"""

# Package version
__version__ = "1.0.0"
__author__ = "TNSIM Team"
__email__ = "contact@tnsim.org"
__description__ = "Theory of Zero-Sum Infinite Sets"
__url__ = "https://github.com/tnsim/tnsim"
__license__ = "MIT"

# Import main classes
try:
    from .core import (
        ZeroSumInfiniteSet,
        TNSIMCache,
        ParallelTNSIM,
        cached_operation,
        get_global_cache,
        get_global_parallel_processor,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import main classes: {e}")
    ZeroSumInfiniteSet = None
    TNSIMCache = None
    ParallelTNSIM = None
    cached_operation = None
    get_global_cache = None
    get_global_parallel_processor = None

# Import integrations
try:
    from .integrations import (
        ZeroSumAttention,
        BalansisCompensator,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import integrations: {e}")
    ZeroSumAttention = None
    BalansisCompensator = None

# Import database configuration
try:
    from .database import (
        DatabaseConfig,
        db_config,
        get_config,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import database configuration: {e}")
    DatabaseConfig = None
    db_config = None
    get_config = None

# Exported symbols
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
    "__license__",
    
    # Main classes
    "ZeroSumInfiniteSet",
    "TNSIMCache",
    "ParallelTNSIM",
    "cached_operation",
    "get_global_cache",
    "get_global_parallel_processor",
    
    # Integrations
    "ZeroSumAttention",
    "BalansisCompensator",
    
    # Database
    "DatabaseConfig",
    "db_config",
    "get_config",
    
    # Utilities
    "create_harmonic_series",
    "create_alternating_series",
    "create_geometric_series",
    "run_server",
    "get_version_info",
]

# Utility functions
def create_harmonic_series(n_terms: int = 1000) -> 'ZeroSumInfiniteSet':
    """Create a harmonic series.
    
    Args:
        n_terms: Number of elements to generate
        
    Returns:
        ZeroSumInfiniteSet: Harmonic series object
    """
    if ZeroSumInfiniteSet is None:
        raise ImportError("ZeroSumInfiniteSet is not available")
    return ZeroSumInfiniteSet.create_harmonic_series(n_terms)

def create_alternating_series(n_terms: int = 1000) -> 'ZeroSumInfiniteSet':
    """Create an alternating series.
    
    Args:
        n_terms: Number of elements to generate
        
    Returns:
        ZeroSumInfiniteSet: Alternating series object
    """
    if ZeroSumInfiniteSet is None:
        raise ImportError("ZeroSumInfiniteSet is not available")
    return ZeroSumInfiniteSet.create_alternating_series(n_terms)

def create_geometric_series(ratio: float = 0.5, n_terms: int = 1000) -> 'ZeroSumInfiniteSet':
    """Create a geometric series.
    
    Args:
        ratio: Common ratio of the progression
        n_terms: Number of elements to generate
        
    Returns:
        ZeroSumInfiniteSet: Geometric series object
    """
    if ZeroSumInfiniteSet is None:
        raise ImportError("ZeroSumInfiniteSet is not available")
    return ZeroSumInfiniteSet.create_geometric_series(ratio, n_terms)

def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Start FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional parameters for uvicorn
    """
    try:
        import uvicorn
        from .api.main import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            **kwargs
        )
    except ImportError:
        raise ImportError("uvicorn and FastAPI are required to run the server")

def get_version_info() -> dict:
    """Get version and dependency information.
    
    Returns:
        dict: Dictionary with version information
    """
    import sys
    import platform
    
    info = {
        "tnsim_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }
    
    # Check availability of main dependencies
    dependencies = {
        "numpy": None,
        "scipy": None,
        "torch": None,
        "fastapi": None,
        "asyncpg": None,
    }
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            dependencies[dep] = getattr(module, "__version__", "unknown")
        except ImportError:
            dependencies[dep] = "not installed"
    
    info["dependencies"] = dependencies
    
    # Check availability of TNSIM components
    components = {
        "ZeroSumInfiniteSet": ZeroSumInfiniteSet is not None,
        "TNSIMCache": TNSIMCache is not None,
        "ParallelTNSIM": ParallelTNSIM is not None,
        "ZeroSumAttention": ZeroSumAttention is not None,
        "BalansisCompensator": BalansisCompensator is not None,
        "DatabaseConfig": DatabaseConfig is not None,
    }
    
    info["components"] = components
    
    return info

# Logging setup
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Python compatibility check
import sys

if sys.version_info < (3, 9):
    raise RuntimeError(
        f"TNSIM requires Python 3.9 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Initialization on import
def _initialize():
    """Initialize package on import."""
    # Default logging setup
    import os
    
    log_level = os.getenv("TNSIM_LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Check environment variables
    required_env_vars = [
        "DATABASE_URL",
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars and os.getenv("TNSIM_STRICT_ENV", "false").lower() == "true":
        raise EnvironmentError(
            f"Missing required environment variables: {missing_vars}"
        )

# Execute initialization
_initialize()