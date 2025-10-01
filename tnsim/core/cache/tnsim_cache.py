"""Caching system for TNSIM operations."""

import json
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from decimal import Decimal
import threading
from pathlib import Path

class TNSIMCache:
    """Cache for operations with infinite sets within TNSIM framework.
    
    Provides fast access to computation results and ⊕ operations.
    """
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24, 
                 persistent: bool = True, cache_dir: str = "cache"):
        """Cache initialization.
        
        Args:
            max_size: Maximum cache size
            ttl_hours: Entry time-to-live in hours
            persistent: Whether to save cache to disk
            cache_dir: Directory for cache storage
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.persistent = persistent
        self.cache_dir = Path(cache_dir)
        
        # Main cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
        
        # Create cache directory
        if self.persistent:
            self.cache_dir.mkdir(exist_ok=True)
            self._load_persistent_cache()
    
    def _generate_key(self, operation: str, *args, **kwargs) -> str:
        """Generate key for caching.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            **kwargs: Named arguments
            
        Returns:
            Hash key for caching
        """
        # Create string for hashing
        key_data = {
            'operation': operation,
            'args': self._serialize_args(args),
            'kwargs': self._serialize_args(kwargs)
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _serialize_args(self, args) -> Any:
        """Serialize arguments for hashing."""
        if isinstance(args, (list, tuple)):
            return [self._serialize_args(arg) for arg in args]
        elif isinstance(args, dict):
            return {k: self._serialize_args(v) for k, v in args.items()}
        elif isinstance(args, Decimal):
            return str(args)
        elif hasattr(args, 'to_dict'):
            return args.to_dict()
        elif hasattr(args, '__dict__'):
            return str(args)
        else:
            return args
    
    def get(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            **kwargs: Named arguments
            
        Returns:
            Cached value or None
        """
        key = self._generate_key(operation, *args, **kwargs)
        
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if self._is_expired(entry['timestamp']):
                del self._cache[key]
                del self._access_times[key]
                self._stats['misses'] += 1
                self._stats['evictions'] += 1
                return None
            
            # Update access time
            self._access_times[key] = datetime.now()
            self._stats['hits'] += 1
            
            return entry['value']
    
    def set(self, operation: str, value: Any, *args, **kwargs) -> None:
        """Save value to cache.
        
        Args:
            operation: Operation name
            value: Value to cache
            *args: Positional arguments
            **kwargs: Named arguments
        """
        key = self._generate_key(operation, *args, **kwargs)
        
        with self._lock:
            # Check cache size
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            # Save to cache
            self._cache[key] = {
                'value': value,
                'timestamp': datetime.now(),
                'operation': operation
            }
            self._access_times[key] = datetime.now()
            self._stats['size'] = len(self._cache)
            
            # Save to disk if enabled
            if self.persistent:
                self._save_entry_to_disk(key, self._cache[key])
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check TTL expiration."""
        return datetime.now() - timestamp > self.ttl
    
    def _evict_lru(self) -> None:
        """Remove least recently used entry."""
        if not self._access_times:
            return
        
        # Find key with least access time
        lru_key = min(self._access_times.keys(), 
                     key=lambda k: self._access_times[k])
        
        # Remove entry
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self._stats['evictions'] += 1
        
        # Remove from disk
        if self.persistent:
            self._remove_entry_from_disk(lru_key)
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'size': 0
            }
            
            # Clear disk
            if self.persistent:
                self._clear_disk_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self._stats,
                'hit_rate_percent': round(hit_rate, 2),
                'total_requests': total_requests,
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'ttl_hours': self.ttl.total_seconds() / 3600
            }
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries.
        
        Returns:
            Number of removed entries
        """
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if self._is_expired(entry['timestamp']):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                del self._access_times[key]
                if self.persistent:
                    self._remove_entry_from_disk(key)
            
            self._stats['evictions'] += len(expired_keys)
            self._stats['size'] = len(self._cache)
        
        return len(expired_keys)
    
    def _save_entry_to_disk(self, key: str, entry: Dict[str, Any]) -> None:
        """Save entry to disk."""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            # Error logging (in real application)
            pass
    
    def _remove_entry_from_disk(self, key: str) -> None:
        """Remove entry from disk."""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            # Error logging
            pass
    
    def _load_persistent_cache(self) -> None:
        """Load cache from disk."""
        try:
            for file_path in self.cache_dir.glob("*.pkl"):
                key = file_path.stem
                
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check TTL
                if not self._is_expired(entry['timestamp']):
                    self._cache[key] = entry
                    self._access_times[key] = entry['timestamp']
                else:
                    # Remove expired files
                    file_path.unlink()
            
            self._stats['size'] = len(self._cache)
        except Exception as e:
            # Error logging
            pass
    
    def _clear_disk_cache(self) -> None:
        """Clear cache on disk."""
        try:
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()
        except Exception as e:
            # Error logging
            pass
    
    def get_cached_operations(self) -> List[str]:
        """Get list of cached operations."""
        with self._lock:
            operations = set()
            for entry in self._cache.values():
                operations.add(entry['operation'])
            return list(operations)
    
    def invalidate_operation(self, operation: str) -> int:
        """Invalidate all entries for a specific operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Number of removed entries
        """
        keys_to_remove = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if entry['operation'] == operation:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
                del self._access_times[key]
                if self.persistent:
                    self._remove_entry_from_disk(key)
            
            self._stats['evictions'] += len(keys_to_remove)
            self._stats['size'] = len(self._cache)
        
        return len(keys_to_remove)
    
    def __enter__(self):
        """Context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Can add final cleanup or saving
        pass

# Global cache instance
_global_cache = None

def get_global_cache() -> TNSIMCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = TNSIMCache()
    return _global_cache

def cached_operation(operation_name: str):
    """Decorator for caching operations.
    
    Args:
        operation_name: Operation name for caching
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()
            
            # Try to get from cache
            result = cache.get(operation_name, *args, **kwargs)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(operation_name, result, *args, **kwargs)
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator