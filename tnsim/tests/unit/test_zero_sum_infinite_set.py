"""Unit tests for ZeroSumInfiniteSet class."""

import math
import pytest
from decimal import Decimal
from typing import List

from tnsim.core.sets import ZeroSumInfiniteSet


class TestZeroSumInfiniteSetCreation:
    """Tests for ZeroSumInfiniteSet creation."""
    
    def test_create_empty_set(self):
        """Test creating an empty set."""
        zs_set = ZeroSumInfiniteSet()
        assert len(zs_set.elements) == 0
        assert zs_set.series_type == "custom"
        assert zs_set.name is None
    
    def test_create_with_elements(self, sample_elements):
        """Test creating a set with elements."""
        zs_set = ZeroSumInfiniteSet(
            elements=sample_elements,
            series_type="custom",
            name="test_series"
        )
        assert zs_set.elements == sample_elements
        assert zs_set.series_type == "custom"
        assert zs_set.name == "test_series"
    
    def test_create_harmonic_series(self):
        """Test creating a harmonic series."""
        n_terms = 10
        harmonic = ZeroSumInfiniteSet.create_harmonic_series(n_terms)
        
        assert len(harmonic.elements) == n_terms
        assert harmonic.series_type == "harmonic"
        assert harmonic.elements[0] == 1.0
        assert harmonic.elements[1] == 0.5
        assert harmonic.elements[9] == 0.1
    
    def test_create_alternating_series(self):
        """Test creating an alternating series."""
        n_terms = 10
        alternating = ZeroSumInfiniteSet.create_alternating_series(n_terms)
        
        assert len(alternating.elements) == n_terms
        assert alternating.series_type == "alternating"
        assert alternating.elements[0] == 1.0
        assert alternating.elements[1] == -0.5
        assert alternating.elements[2] == 1/3
        assert alternating.elements[3] == -0.25
    
    def test_create_geometric_series(self):
        """Test creating a geometric series."""
        ratio = 0.5
        n_terms = 5
        geometric = ZeroSumInfiniteSet.create_geometric_series(ratio, n_terms)
        
        assert len(geometric.elements) == n_terms
        assert geometric.series_type == "geometric"
        assert geometric.elements[0] == 1.0
        assert geometric.elements[1] == 0.5
        assert geometric.elements[2] == 0.25
        assert geometric.elements[3] == 0.125
        assert geometric.elements[4] == 0.0625
    
    def test_invalid_series_creation(self):
        """Test creating series with invalid parameters."""
        with pytest.raises(ValueError):
            ZeroSumInfiniteSet.create_harmonic_series(0)
        
        with pytest.raises(ValueError):
            ZeroSumInfiniteSet.create_harmonic_series(-5)
        
        with pytest.raises(ValueError):
            ZeroSumInfiniteSet.create_geometric_series(1.0, 10)  # |ratio| >= 1


class TestZeroSumOperations:
    """Tests for zero-sum operations."""
    
    def test_zero_sum_operation_direct(self, zero_sum_set):
        """Test direct zero-sum operation."""
        result = zero_sum_set.zero_sum_operation(method="direct")
        
        assert "sum" in result
        assert "method" in result
        assert "compensation_error" in result
        assert result["method"] == "direct"
        assert isinstance(result["sum"], (int, float, Decimal))
    
    def test_zero_sum_operation_compensated(self, zero_sum_set):
        """Test compensated zero-sum operation."""
        result = zero_sum_set.zero_sum_operation(method="compensated")
        
        assert result["method"] == "compensated"
        assert "compensation_error" in result
        assert "iterations" in result
        
        # Compensated sum should be more accurate
        direct_result = zero_sum_set.zero_sum_operation(method="direct")
        assert abs(result["compensation_error"]) <= abs(direct_result["compensation_error"])
    
    def test_zero_sum_operation_stabilized(self, zero_sum_set):
        """Test stabilized zero-sum operation."""
        result = zero_sum_set.zero_sum_operation(method="stabilized")
        
        assert result["method"] == "stabilized"
        assert "stability_factor" in result
        assert "numerical_precision" in result
    
    def test_zero_sum_operation_invalid_method(self, zero_sum_set):
        """Test operation with invalid method."""
        with pytest.raises(ValueError):
            zero_sum_set.zero_sum_operation(method="invalid_method")
    
    @pytest.mark.parametrize("method", ["direct", "compensated", "stabilized"])
    def test_zero_sum_operation_methods(self, geometric_set, method):
        """Parameterized test of all operation methods."""
        result = geometric_set.zero_sum_operation(method=method)
        
        assert "sum" in result
        assert "method" in result
        assert result["method"] == method
        
        # For geometric series with ratio=0.5 expect sum close to 2
        expected_sum = 2.0
        assert abs(result["sum"] - expected_sum) < 0.1


class TestCompensatingSet:
    """Tests for finding compensating sets."""
    
    def test_find_compensating_set_direct(self, zero_sum_set):
        """Test direct compensating set search."""
        result = zero_sum_set.find_compensating_set(method="direct")
        
        assert "compensating_elements" in result
        assert "compensation_quality" in result
        assert "method" in result
        assert result["method"] == "direct"
        assert isinstance(result["compensating_elements"], list)
    
    def test_find_compensating_set_iterative(self, alternating_set):
        """Test iterative compensating set search."""
        result = alternating_set.find_compensating_set(
            method="iterative",
            max_iterations=50
        )
        
        assert result["method"] == "iterative"
        assert "iterations_used" in result
        assert "convergence_achieved" in result
        assert result["iterations_used"] <= 50
    
    def test_find_compensating_set_adaptive(self, custom_set):
        """Test adaptive compensating set search."""
        result = custom_set.find_compensating_set(
            method="adaptive",
            tolerance=1e-10
        )
        
        assert result["method"] == "adaptive"
        assert "adaptation_steps" in result
        assert "final_tolerance" in result
        assert result["final_tolerance"] <= 1e-10
    
    def test_compensating_set_quality(self, zero_sum_set):
        """Test compensating set quality."""
        result = zero_sum_set.find_compensating_set(method="direct")
        
        # Check that compensating elements actually compensate
        original_sum = sum(zero_sum_set.elements)
        compensating_sum = sum(result["compensating_elements"])
        total_sum = original_sum + compensating_sum
        
        assert abs(total_sum) < 1e-10  # Should be close to zero
        assert result["compensation_quality"] > 0.9  # High quality


class TestValidation:
    """Tests for zero-sum validation."""
    
    def test_validate_zero_sum_basic(self, zero_sum_set):
        """Basic zero-sum validation test."""
        result = zero_sum_set.validate_zero_sum()
        
        assert "is_zero_sum" in result
        assert "total_sum" in result
        assert "tolerance_used" in result
        assert "validation_method" in result
        assert isinstance(result["is_zero_sum"], bool)
    
    def test_validate_zero_sum_with_tolerance(self, alternating_set):
        """Test validation with specified tolerance."""
        tolerance = 1e-8
        result = alternating_set.validate_zero_sum(tolerance=tolerance)
        
        assert result["tolerance_used"] == tolerance
        
        # Check validation logic
        if abs(result["total_sum"]) <= tolerance:
            assert result["is_zero_sum"] is True
        else:
            assert result["is_zero_sum"] is False
    
    def test_validate_zero_sum_detailed(self, geometric_set):
        """Test detailed zero-sum validation."""
        result = geometric_set.validate_zero_sum(detailed=True)
        
        assert "element_contributions" in result
        assert "cumulative_sums" in result
        assert "error_analysis" in result
        assert len(result["element_contributions"]) == len(geometric_set.elements)
    
    def test_validate_compensated_sum(self):
        """Test validation of compensated sum."""
        # Create a set that should give zero sum
        elements = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25]
        zs_set = ZeroSumInfiniteSet(elements=elements)
        
        result = zs_set.validate_zero_sum(tolerance=1e-15)
        
        assert result["is_zero_sum"] is True
        assert abs(result["total_sum"]) < 1e-15


class TestPartialSums:
    """Tests for partial sums."""
    
    def test_get_partial_sum_basic(self, zero_sum_set):
        """Basic test for getting partial sum."""
        n = 5
        partial_sum = zero_sum_set.get_partial_sum(n)
        
        expected_sum = sum(zero_sum_set.elements[:n])
        assert abs(partial_sum - expected_sum) < 1e-15
    
    def test_get_partial_sum_range(self, alternating_set):
        """Test getting partial sum in range."""
        start, end = 2, 7
        partial_sum = alternating_set.get_partial_sum(end, start)
        
        expected_sum = sum(alternating_set.elements[start:end])
        assert abs(partial_sum - expected_sum) < 1e-15
    
    def test_get_partial_sum_invalid_range(self, zero_sum_set):
        """Test getting partial sum with invalid range."""
        with pytest.raises(ValueError):
            zero_sum_set.get_partial_sum(-1)
        
        with pytest.raises(ValueError):
            zero_sum_set.get_partial_sum(len(zero_sum_set.elements) + 1)
    
    def test_partial_sums_convergence(self, convergent_series):
        """Test convergence of partial sums."""
        zs_set = ZeroSumInfiniteSet(elements=convergent_series)
        
        partial_sums = []
        for i in range(1, len(convergent_series) + 1):
            partial_sums.append(zs_set.get_partial_sum(i))
        
        # Check that partial sums stabilize
        last_few_sums = partial_sums[-5:]
        variance = sum((s - sum(last_few_sums)/len(last_few_sums))**2 for s in last_few_sums)
        assert variance < 1e-10  # Low variation means convergence


class TestConvergenceAnalysis:
    """Tests for convergence analysis."""
    
    def test_analyze_convergence_basic(self, geometric_set):
        """Basic convergence analysis test."""
        result = geometric_set.analyze_convergence()
        
        assert "converges" in result
        assert "convergence_type" in result
        assert "convergence_rate" in result
        assert isinstance(result["converges"], bool)
    
    def test_analyze_convergence_ratio_test(self, alternating_set):
        """Test convergence analysis with ratio test."""
        result = alternating_set.analyze_convergence(method="ratio_test")
        
        assert result["method"] == "ratio_test"
        assert "ratio_limit" in result
        assert "test_result" in result
    
    def test_analyze_convergence_root_test(self, harmonic_set):
        """Test convergence analysis with root test."""
        result = harmonic_set.analyze_convergence(method="root_test")
        
        assert result["method"] == "root_test"
        assert "root_limit" in result
        assert "test_result" in result
    
    def test_analyze_convergence_integral_test(self, custom_set):
        """Test convergence analysis with integral test."""
        result = custom_set.analyze_convergence(method="integral_test")
        
        assert result["method"] == "integral_test"
        assert "integral_convergence" in result
    
    @pytest.mark.parametrize("method", ["ratio_test", "root_test", "integral_test"])
    def test_convergence_methods_parametrized(self, zero_sum_set, method):
        """Parametrized test for different convergence analysis methods."""
        result = zero_sum_set.analyze_convergence(method=method)
        
        assert "converges" in result
        assert "method" in result
        assert result["method"] == method


class TestSerialization:
    """Tests for serialization and deserialization."""
    
    def test_to_dict(self, zero_sum_set):
        """Test conversion to dictionary."""
        data = zero_sum_set.to_dict()
        
        assert isinstance(data, dict)
        assert "elements" in data
        assert "metadata" in data
        assert "class_name" in data
        assert data["class_name"] == "ZeroSumInfiniteSet"
    
    def test_from_dict(self, zero_sum_set):
        """Test creation from dictionary."""
        data = zero_sum_set.to_dict()
        restored_set = ZeroSumInfiniteSet.from_dict(data)
        
        assert len(restored_set.elements) == len(zero_sum_set.elements)
        assert all(a == b for a, b in zip(restored_set.elements, zero_sum_set.elements))
    
    def test_serialization_roundtrip(self, alternating_set):
        """Test complete serialization cycle."""
        # Serialization
        data = alternating_set.to_dict()
        
        # Deserialization
        restored_set = ZeroSumInfiniteSet.from_dict(data)
        
        # Check identity
        assert restored_set.elements == alternating_set.elements
        assert restored_set.metadata == alternating_set.metadata
    
    def test_serialization_with_metadata(self, custom_set):
        """Test serialization with metadata."""
        # Add metadata
        custom_set.metadata["test_key"] = "test_value"
        custom_set.metadata["creation_time"] = "2024-01-01"
        
        # Serialization and deserialization
        data = custom_set.to_dict()
        restored_set = ZeroSumInfiniteSet.from_dict(data)
        
        assert restored_set.metadata["test_key"] == "test_value"
        assert restored_set.metadata["creation_time"] == "2024-01-01"


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_set_operations(self):
        """Test operations with empty set."""
        empty_set = ZeroSumInfiniteSet()
        
        result = empty_set.zero_sum_operation()
        assert result["sum"] == 0.0
        
        validation = empty_set.validate_zero_sum()
        assert validation["is_zero_sum"] is True
        assert validation["total_sum"] == 0.0
    
    def test_single_element_set(self):
        """Test set with single element."""
        single_set = ZeroSumInfiniteSet(elements=[42.0])
        
        result = single_set.zero_sum_operation()
        assert result["sum"] == 42.0
        
        partial_sum = single_set.get_partial_sum(1)
        assert partial_sum == 42.0
    
    def test_very_large_numbers(self):
        """Test with very large numbers."""
        large_elements = [1e15, -1e15, 1e10, -1e10]
        large_set = ZeroSumInfiniteSet(elements=large_elements)
        
        result = large_set.zero_sum_operation(method="compensated")
        
        # Compensated method should handle large numbers
        assert abs(result["sum"]) < 1e5  # Relatively small error
    
    def test_very_small_numbers(self):
        """Test with very small numbers."""
        small_elements = [1e-15, -1e-15, 1e-10, -1e-10]
        small_set = ZeroSumInfiniteSet(elements=small_elements)
        
        result = small_set.zero_sum_operation(method="stabilized")
        
        # Stabilized method should handle small numbers
        assert "numerical_precision" in result
        assert result["numerical_precision"] > 1e-16
    
    def test_mixed_precision_numbers(self):
        """Test with mixed precision numbers."""
        mixed_elements = [1.0, 1e-15, 1e15, -1e15, -1.0, -1e-15]
        mixed_set = ZeroSumInfiniteSet(elements=mixed_elements)
        
        # All methods should work
        for method in ["direct", "compensated", "stabilized"]:
            result = mixed_set.zero_sum_operation(method=method)
            assert "sum" in result
            assert not math.isnan(result["sum"])
            assert not math.isinf(result["sum"])
    
    def test_infinite_and_nan_handling(self):
        """Test handling of infinity and NaN."""
        # Test with infinity
        with pytest.raises((ValueError, OverflowError)):
            ZeroSumInfiniteSet(elements=[float('inf'), 1, 2])
        
        # Test with NaN
        with pytest.raises((ValueError, TypeError)):
            ZeroSumInfiniteSet(elements=[float('nan'), 1, 2])
    
    def test_zero_elements_handling(self):
        """Test handling of zero elements."""
        zero_elements = [0.0, 0.0, 0.0, 1.0, -1.0]
        zero_set = ZeroSumInfiniteSet(elements=zero_elements)
        
        result = zero_set.zero_sum_operation()
        assert result["sum"] == 0.0
        
        validation = zero_set.validate_zero_sum()
        assert validation["is_zero_sum"] is True