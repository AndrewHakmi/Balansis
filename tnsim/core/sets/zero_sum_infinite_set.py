"""Main class for working with infinite sets within the TNSIM framework."""

import numpy as np
from typing import List, Optional, Union, Dict, Any
from decimal import Decimal, getcontext
import uuid
from datetime import datetime

# Import from Balansis library
try:
    from balansis.core.operations import compensated_sum, compensated_matmul
    from balansis.logic.compensator import Compensator
except ImportError:
    # Fallbacks for when Balansis is not available
    class Compensator:
        def compensate(self, a, b):
            return a + b
        
        def stabilize(self, x):
            return x
    
    def compensated_sum(arr):
        return np.sum(arr)

# Set precision for Decimal
getcontext().prec = 50

class ZeroSumInfiniteSet:
    """Class for representing infinite sets within the TNSIM framework.
    
    Implements the ⊕ operation and compensation principles for working with infinite sets.
    """
    
    def __init__(self, elements: List[Union[float, Decimal]], 
                 set_type: str = 'custom',
                 name: Optional[str] = None,
                 properties: Optional[Dict[str, Any]] = None):
        """Initialize infinite set.
        
        Args:
            elements: List of set elements
            set_type: Set type ('harmonic', 'alternating', 'geometric', 'custom')
            name: Set name
            properties: Additional set properties
        """
        self.id = str(uuid.uuid4())
        self.elements = np.array([Decimal(str(x)) for x in elements])
        self.set_type = set_type
        self.name = name or f"{set_type}_set_{self.id[:8]}"
        self.properties = properties or {}
        self.created_at = datetime.now()
        
        # Initialize compensator from Balansis
        self.compensator = Compensator()
        self._compensating_set = None
        self._cached_sum = None
        
    def __repr__(self) -> str:
        return f"ZeroSumInfiniteSet(name='{self.name}', type='{self.set_type}', elements={len(self.elements)})"
    
    def zero_sum_operation(self, other: 'ZeroSumInfiniteSet', 
                          method: str = 'compensated') -> Decimal:
        """Execute ⊕ operation between two sets.
        
        Args:
            other: Another infinite set
            method: Calculation method ('direct', 'compensated', 'stabilized')
            
        Returns:
            Result of ⊕ operation
        """
        if method == 'direct':
            return self._direct_sum(other)
        elif method == 'compensated':
            return self._compensated_sum(other)
        elif method == 'stabilized':
            return self._stabilized_sum(other)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _direct_sum(self, other: 'ZeroSumInfiniteSet') -> Decimal:
        """Direct summation of set elements."""
        sum_a = sum(self.elements)
        sum_b = sum(other.elements)
        return sum_a + sum_b
    
    def _compensated_sum(self, other: 'ZeroSumInfiniteSet') -> Decimal:
        """Compensated summation through Balansis."""
        sum_a = float(sum(self.elements))
        sum_b = float(sum(other.elements))
        result = self.compensator.compensate(sum_a, sum_b)
        return Decimal(str(result))
    
    def _stabilized_sum(self, other: 'ZeroSumInfiniteSet') -> Decimal:
        """Stabilized summation with additional compensation."""
        # Apply stabilization to each set
        stabilized_a = self.compensator.stabilize(self.elements.astype(float))
        stabilized_b = self.compensator.stabilize(other.elements.astype(float))
        
        # Compensated summation
        sum_a = compensated_sum(stabilized_a)
        sum_b = compensated_sum(stabilized_b)
        
        return Decimal(str(sum_a + sum_b))
    
    def find_compensating_set(self, method: str = 'direct') -> 'ZeroSumInfiniteSet':
        """Find compensating set.
        
        Args:
            method: Search method ('direct', 'iterative', 'adaptive')
            
        Returns:
            Compensating set
        """
        if self._compensating_set is None:
            if method == 'direct':
                self._compensating_set = self._direct_compensating_set()
            elif method == 'iterative':
                self._compensating_set = self._iterative_compensating_set()
            elif method == 'adaptive':
                self._compensating_set = self._adaptive_compensating_set()
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return self._compensating_set
    
    def _direct_compensating_set(self) -> 'ZeroSumInfiniteSet':
        """Direct creation of compensating set."""
        compensating_elements = [-x for x in self.elements]
        return ZeroSumInfiniteSet(
            compensating_elements,
            f"compensating_{self.set_type}",
            f"Compensating_{self.name}",
            {**self.properties, 'compensates': self.id}
        )
    
    def _iterative_compensating_set(self) -> 'ZeroSumInfiniteSet':
        """Iterative search for compensating set."""
        # Start with direct negation
        compensating_elements = [-x for x in self.elements]
        
        # Iterative correction to achieve precise compensation
        for i in range(10):  # Maximum 10 iterations
            temp_set = ZeroSumInfiniteSet(compensating_elements)
            result = self.zero_sum_operation(temp_set, 'compensated')
            
            if abs(result) < Decimal('1e-10'):
                break
                
            # Element correction
            correction = result / len(compensating_elements)
            compensating_elements = [x - float(correction) for x in compensating_elements]
        
        return ZeroSumInfiniteSet(
            compensating_elements,
            f"iterative_compensating_{self.set_type}",
            f"Iterative_Compensating_{self.name}",
            {**self.properties, 'compensates': self.id, 'method': 'iterative'}
        )
    
    def _adaptive_compensating_set(self) -> 'ZeroSumInfiniteSet':
        """Adaptive search for compensating set using Balansis."""
        # Use compensator for adaptive search
        stabilized_elements = self.compensator.stabilize(self.elements.astype(float))
        compensating_elements = [-self.compensator.stabilize(x) for x in stabilized_elements]
        
        return ZeroSumInfiniteSet(
            compensating_elements,
            f"adaptive_compensating_{self.set_type}",
            f"Adaptive_Compensating_{self.name}",
            {**self.properties, 'compensates': self.id, 'method': 'adaptive'}
        )
    
    def validate_zero_sum(self, tolerance: Decimal = Decimal('1e-10')) -> Dict[str, Any]:
        """Validate zero sum with compensating set.
        
        Args:
            tolerance: Acceptable error tolerance
            
        Returns:
            Dictionary with validation results
        """
        compensating = self.find_compensating_set()
        result = self.zero_sum_operation(compensating, 'compensated')
        
        is_zero_sum = abs(result) < tolerance
        
        return {
            'is_zero_sum': is_zero_sum,
            'result': result,
            'tolerance': tolerance,
            'error_margin': abs(result),
            'compensating_set_id': compensating.id,
            'validation_timestamp': datetime.now()
        }
    
    def get_partial_sum(self, n_elements: int) -> Decimal:
        """Get partial sum of first n elements.
        
        Args:
            n_elements: Number of elements to sum
            
        Returns:
            Partial sum
        """
        if n_elements > len(self.elements):
            n_elements = len(self.elements)
        
        return sum(self.elements[:n_elements])
    
    def convergence_analysis(self, max_terms: int = 1000) -> Dict[str, Any]:
        """Analyze series convergence.
        
        Args:
            max_terms: Maximum number of terms for analysis
            
        Returns:
            Convergence analysis results
        """
        partial_sums = []
        n_terms = min(max_terms, len(self.elements))
        
        for i in range(1, n_terms + 1):
            partial_sum = self.get_partial_sum(i)
            partial_sums.append(float(partial_sum))
        
        # Simple convergence analysis
        if len(partial_sums) > 10:
            last_10 = partial_sums[-10:]
            variance = np.var(last_10)
            is_convergent = variance < 1e-6
        else:
            is_convergent = False
            variance = float('inf')
        
        return {
            'is_convergent': is_convergent,
            'partial_sums': partial_sums,
            'variance': variance,
            'final_sum': partial_sums[-1] if partial_sums else 0,
            'n_terms_analyzed': n_terms
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'set_type': self.set_type,
            'elements': [float(x) for x in self.elements],
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'element_count': len(self.elements)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZeroSumInfiniteSet':
        """Create object from dictionary."""
        obj = cls(
            elements=data['elements'],
            set_type=data['set_type'],
            name=data['name'],
            properties=data.get('properties', {})
        )
        obj.id = data['id']
        obj.created_at = datetime.fromisoformat(data['created_at'])
        return obj

    @staticmethod
    def create_harmonic_series(n_terms: int = 1000) -> 'ZeroSumInfiniteSet':
        """Create harmonic series."""
        elements = [Decimal(1) / Decimal(i) for i in range(1, n_terms + 1)]
        return ZeroSumInfiniteSet(
            elements,
            'harmonic',
            'Harmonic Series',
            {'formula': '1/n', 'divergent': True}
        )
    
    @staticmethod
    def create_alternating_series(n_terms: int = 1000) -> 'ZeroSumInfiniteSet':
        """Create alternating series."""
        elements = [Decimal((-1) ** i) / Decimal(i + 1) for i in range(n_terms)]
        return ZeroSumInfiniteSet(
            elements,
            'alternating',
            'Alternating Series',
            {'formula': '(-1)^n/(n+1)', 'convergent': True}
        )
    
    @staticmethod
    def create_geometric_series(ratio: float = 0.5, n_terms: int = 1000) -> 'ZeroSumInfiniteSet':
        """Create geometric series."""
        elements = [Decimal(ratio) ** i for i in range(n_terms)]
        return ZeroSumInfiniteSet(
            elements,
            'geometric',
            f'Geometric Series (r={ratio})',
            {'ratio': ratio, 'convergent': abs(ratio) < 1}
        )