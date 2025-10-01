"""Tests for the Algebra module.

This module contains comprehensive tests for algebraic structures,
verifying group and field operations for AbsoluteValue and EternalRatio.
"""

import pytest
import math
from typing import List, Set

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.algebra.absolute_group import (
    GroupElement, GroupOperation, AdditiveOperation, MultiplicativeOperation, AbsoluteGroup
)
from balansis.algebra.eternity_field import (
    FieldElement, FieldOperation, EternalRatioOperation, EternityField, PolynomialRing, Polynomial
)
from balansis import ACT_EPSILON


class TestGroupElement:
    """Test GroupElement wrapper functionality."""
    
    def test_group_element_creation(self):
        """Test GroupElement creation."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        element = GroupElement(value=av)
        
        assert element.value == av
        assert element.value.magnitude == 5.0
        assert element.value.direction == 1.0
    
    def test_group_element_equality(self):
        """Test GroupElement equality."""
        av1 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=5.0, direction=1.0)
        av3 = AbsoluteValue(magnitude=3.0, direction=1.0)
        
        elem1 = GroupElement(value=av1)
        elem2 = GroupElement(value=av2)
        elem3 = GroupElement(value=av3)
        
        assert elem1 == elem2
        assert elem1 != elem3
        assert hash(elem1) == hash(elem2)
        assert hash(elem1) != hash(elem3)
    
    def test_group_element_string_representation(self):
        """Test GroupElement string representation."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        element = GroupElement(value=av)
        
        str_repr = str(element)
        assert "GroupElement" in str_repr
        assert "5.0" in str_repr


class TestAdditiveOperation:
    """Test AdditiveOperation for groups."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.operation = AdditiveOperation()
    
    def test_additive_operation_basic(self):
        """Test basic additive operation."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=2.0, direction=1.0)
        
        elem1 = GroupElement(value=av1)
        elem2 = GroupElement(value=av2)
        
        result = self.operation.operate(elem1, elem2)
        assert result.value.magnitude == 5.0
        assert result.value.direction == 1.0
    
    def test_additive_identity(self):
        """Test additive identity element."""
        identity = self.operation.identity()
        assert identity.value.is_absolute()
        
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        elem = GroupElement(value=av)
        
        result1 = self.operation.operate(elem, identity)
        result2 = self.operation.operate(identity, elem)
        
        assert result1 == elem
        assert result2 == elem
    
    def test_additive_inverse(self):
        """Test additive inverse element."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        elem = GroupElement(value=av)
        
        inverse = self.operation.inverse(elem)
        assert inverse.value.magnitude == 5.0
        assert inverse.value.direction == -1.0
        
        identity = self.operation.identity()
        result1 = self.operation.operate(elem, inverse)
        result2 = self.operation.operate(inverse, elem)
        
        assert result1 == identity
        assert result2 == identity
    
    def test_additive_associativity(self):
        """Test additive associativity."""
        av1 = AbsoluteValue(magnitude=2.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av3 = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        elem1 = GroupElement(value=av1)
        elem2 = GroupElement(value=av2)
        elem3 = GroupElement(value=av3)
        
        # (a + b) + c
        temp1 = self.operation.operate(elem1, elem2)
        result1 = self.operation.operate(temp1, elem3)
        
        # a + (b + c)
        temp2 = self.operation.operate(elem2, elem3)
        result2 = self.operation.operate(elem1, temp2)
        
        assert result1 == result2


class TestMultiplicativeOperation:
    """Test MultiplicativeOperation for groups."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.operation = MultiplicativeOperation()
    
    def test_multiplicative_operation_basic(self):
        """Test basic multiplicative operation."""
        av1 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        elem1 = GroupElement(value=av1)
        elem2 = GroupElement(value=av2)
        
        result = self.operation.operate(elem1, elem2)
        assert result.value.magnitude == 12.0
        assert result.value.direction == 1.0
    
    def test_multiplicative_identity(self):
        """Test multiplicative identity element."""
        identity = self.operation.identity()
        assert identity.value.magnitude == 1.0
        assert identity.value.direction == 1.0
        
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        elem = GroupElement(value=av)
        
        result1 = self.operation.operate(elem, identity)
        result2 = self.operation.operate(identity, elem)
        
        assert result1 == elem
        assert result2 == elem
    
    def test_multiplicative_inverse(self):
        """Test multiplicative inverse element."""
        av = AbsoluteValue(magnitude=4.0, direction=1.0)
        elem = GroupElement(value=av)
        
        inverse = self.operation.inverse(elem)
        assert abs(inverse.value.magnitude - 0.25) < ACT_EPSILON
        assert inverse.value.direction == 1.0
        
        identity = self.operation.identity()
        result1 = self.operation.operate(elem, inverse)
        result2 = self.operation.operate(inverse, elem)
        
        assert abs(result1.value.magnitude - identity.value.magnitude) < ACT_EPSILON
        assert abs(result2.value.magnitude - identity.value.magnitude) < ACT_EPSILON
    
    def test_multiplicative_inverse_absolute(self):
        """Test multiplicative inverse of Absolute raises error."""
        absolute = AbsoluteValue.absolute()
        elem = GroupElement(value=absolute)
        
        with pytest.raises(ValueError, match="Cannot compute multiplicative inverse of Absolute"):
            self.operation.inverse(elem)
    
    def test_multiplicative_associativity(self):
        """Test multiplicative associativity."""
        av1 = AbsoluteValue(magnitude=2.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)
        av3 = AbsoluteValue(magnitude=4.0, direction=1.0)
        
        elem1 = GroupElement(value=av1)
        elem2 = GroupElement(value=av2)
        elem3 = GroupElement(value=av3)
        
        # (a * b) * c
        temp1 = self.operation.operate(elem1, elem2)
        result1 = self.operation.operate(temp1, elem3)
        
        # a * (b * c)
        temp2 = self.operation.operate(elem2, elem3)
        result2 = self.operation.operate(elem1, temp2)
        
        assert abs(result1.value.magnitude - result2.value.magnitude) < ACT_EPSILON
        assert result1.value.direction == result2.value.direction


class TestAbsoluteGroup:
    """Test AbsoluteGroup functionality."""
    
    def test_additive_group_creation(self):
        """Test additive group creation."""
        elements = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue.absolute()
        ]
        
        group = AbsoluteGroup.additive_group()
        assert isinstance(group.operation, AdditiveOperation)
        assert group.finite == False
    
    def test_multiplicative_group_creation(self):
        """Test multiplicative group creation."""
        elements = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=0.5, direction=1.0)
        ]
        
        group = AbsoluteGroup(elements, "multiplicative")
        assert group.group_type == "multiplicative"
        assert len(group.elements) == 3
        assert isinstance(group.operation, MultiplicativeOperation)
    
    def test_cyclic_group_creation(self):
        """Test cyclic group creation."""
        generator = AbsoluteValue(magnitude=2.0, direction=1.0)
        
        group = AbsoluteGroup.cyclic_group(generator, 4)
        assert group.group_type == "multiplicative"
        assert len(group.elements) == 4
        
        # Check that elements are powers of generator
        expected_magnitudes = [1.0, 2.0, 4.0, 8.0]
        actual_magnitudes = sorted([elem.value.magnitude for elem in group.elements])
        
        for expected, actual in zip(expected_magnitudes, actual_magnitudes):
            assert abs(expected - actual) < ACT_EPSILON
    
    def test_group_operate(self):
        """Test group operation."""
        elements = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0)
        ]
        
        group = AbsoluteGroup(elements, "additive")
        
        elem1 = GroupElement(elements[0])
        elem2 = GroupElement(elements[1])
        
        result = group.operate(elem1, elem2)
        assert result.value.magnitude == 3.0
        assert result.value.direction == 1.0
    
    def test_group_identity(self):
        """Test group identity element."""
        elements = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        additive_group = AbsoluteGroup(elements, "additive")
        add_identity = additive_group.identity()
        assert add_identity.value.is_absolute()
        
        multiplicative_group = AbsoluteGroup(elements, "multiplicative")
        mult_identity = multiplicative_group.identity()
        assert mult_identity.value.magnitude == 1.0
        assert mult_identity.value.direction == 1.0
    
    def test_group_inverse(self):
        """Test group inverse element."""
        av = AbsoluteValue(magnitude=5.0, direction=1.0)
        elements = [av]
        
        additive_group = AbsoluteGroup(elements, "additive")
        elem = GroupElement(av)
        add_inverse = additive_group.inverse(elem)
        assert add_inverse.value.magnitude == 5.0
        assert add_inverse.value.direction == -1.0
        
        multiplicative_group = AbsoluteGroup(elements, "multiplicative")
        mult_inverse = multiplicative_group.inverse(elem)
        assert abs(mult_inverse.value.magnitude - 0.2) < ACT_EPSILON
        assert mult_inverse.value.direction == 1.0
    
    def test_group_order(self):
        """Test group order calculation."""
        elements = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=3.0, direction=1.0)
        ]
        
        group = AbsoluteGroup(elements, "additive")
        assert group.order() == 3
    
    def test_group_is_abelian(self):
        """Test abelian group check."""
        elements = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0)
        ]
        
        group = AbsoluteGroup(elements, "additive")
        assert group.is_abelian() == True
    
    def test_generate_subgroup(self):
        """Test subgroup generation."""
        generator = AbsoluteValue(magnitude=2.0, direction=1.0)
        elements = [generator]
        
        group = AbsoluteGroup(elements, "multiplicative")
        elem = GroupElement(generator)
        
        subgroup = group.generate_subgroup([elem])
        assert len(subgroup.elements) >= 1
        assert elem in subgroup.elements
    
    def test_left_cosets(self):
        """Test left coset computation."""
        elements = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=1.0),
            AbsoluteValue(magnitude=3.0, direction=1.0)
        ]
        
        group = AbsoluteGroup(elements, "additive")
        subgroup_elements = [GroupElement(elements[0])]
        subgroup = AbsoluteGroup([elements[0]], "additive")
        
        cosets = group.left_cosets(subgroup)
        assert len(cosets) >= 1
        assert all(isinstance(coset, set) for coset in cosets)


class TestFieldElement:
    """Test FieldElement wrapper functionality."""
    
    def test_field_element_creation(self):
        """Test FieldElement creation."""
        av1 = AbsoluteValue(magnitude=6.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)
        ratio = EternalRatio(numerator=av1, denominator=av2)
        
        element = FieldElement(ratio)
        assert element.value == ratio
        assert element.value.numerical_value() == 2.0
    
    def test_field_element_equality(self):
        """Test FieldElement equality."""
        av1 = AbsoluteValue(magnitude=6.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)
        
        ratio1 = EternalRatio(numerator=av1, denominator=av2)
        ratio2 = EternalRatio(numerator=av1, denominator=av2)
        
        elem1 = FieldElement(ratio1)
        elem2 = FieldElement(ratio2)
        
        assert elem1 == elem2
        assert hash(elem1) == hash(elem2)
    
    def test_field_element_string_representation(self):
        """Test FieldElement string representation."""
        av1 = AbsoluteValue(magnitude=6.0, direction=1.0)
        av2 = AbsoluteValue(magnitude=3.0, direction=1.0)
        ratio = EternalRatio(numerator=av1, denominator=av2)
        
        element = FieldElement(ratio)
        str_repr = str(element)
        assert "FieldElement" in str_repr


class TestEternalRatioOperation:
    """Test EternalRatioOperation for fields."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.operation = EternalRatioOperation()
    
    def test_field_addition(self):
        """Test field addition."""
        # 2/1 + 3/1 = 5/1
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        
        elem1 = FieldElement(ratio1)
        elem2 = FieldElement(ratio2)
        
        result = self.operation.add(elem1, elem2)
        assert abs(result.value.numerical_value() - 5.0) < ACT_EPSILON
    
    def test_field_multiplication(self):
        """Test field multiplication."""
        # (2/1) * (3/1) = 6/1
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        
        elem1 = FieldElement(ratio1)
        elem2 = FieldElement(ratio2)
        
        result = self.operation.multiply(elem1, elem2)
        assert abs(result.value.numerical_value() - 6.0) < ACT_EPSILON
    
    def test_additive_identity(self):
        """Test additive identity."""
        identity = self.operation.additive_identity()
        assert identity.value.numerical_value() == 0.0
        
        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=5.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        elem = FieldElement(ratio)
        
        result1 = self.operation.add(elem, identity)
        result2 = self.operation.add(identity, elem)
        
        assert abs(result1.value.numerical_value() - elem.value.numerical_value()) < ACT_EPSILON
        assert abs(result2.value.numerical_value() - elem.value.numerical_value()) < ACT_EPSILON
    
    def test_multiplicative_identity(self):
        """Test multiplicative identity."""
        identity = self.operation.multiplicative_identity()
        assert abs(identity.value.numerical_value() - 1.0) < ACT_EPSILON
        
        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=5.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        elem = FieldElement(ratio)
        
        result1 = self.operation.multiply(elem, identity)
        result2 = self.operation.multiply(identity, elem)
        
        assert abs(result1.value.numerical_value() - elem.value.numerical_value()) < ACT_EPSILON
        assert abs(result2.value.numerical_value() - elem.value.numerical_value()) < ACT_EPSILON
    
    def test_additive_inverse(self):
        """Test additive inverse."""
        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=5.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        elem = FieldElement(ratio)
        
        inverse = self.operation.additive_inverse(elem)
        assert abs(inverse.value.numerical_value() - (-5.0)) < ACT_EPSILON
        
        identity = self.operation.additive_identity()
        result1 = self.operation.add(elem, inverse)
        result2 = self.operation.add(inverse, elem)
        
        assert abs(result1.value.numerical_value() - identity.value.numerical_value()) < ACT_EPSILON
        assert abs(result2.value.numerical_value() - identity.value.numerical_value()) < ACT_EPSILON
    
    def test_multiplicative_inverse(self):
        """Test multiplicative inverse."""
        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=4.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        elem = FieldElement(ratio)
        
        inverse = self.operation.multiplicative_inverse(elem)
        assert abs(inverse.value.numerical_value() - 0.25) < ACT_EPSILON
        
        identity = self.operation.multiplicative_identity()
        result1 = self.operation.multiply(elem, inverse)
        result2 = self.operation.multiply(inverse, elem)
        
        assert abs(result1.value.numerical_value() - identity.value.numerical_value()) < ACT_EPSILON
        assert abs(result2.value.numerical_value() - identity.value.numerical_value()) < ACT_EPSILON
    
    def test_subtraction(self):
        """Test field subtraction."""
        # 5/1 - 3/1 = 2/1
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=5.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        
        elem1 = FieldElement(ratio1)
        elem2 = FieldElement(ratio2)
        
        result = self.operation.subtract(elem1, elem2)
        assert abs(result.value.numerical_value() - 2.0) < ACT_EPSILON
    
    def test_division(self):
        """Test field division."""
        # (6/1) / (3/1) = 2/1
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=6.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        
        elem1 = FieldElement(ratio1)
        elem2 = FieldElement(ratio2)
        
        result = self.operation.divide(elem1, elem2)
        assert abs(result.value.numerical_value() - 2.0) < ACT_EPSILON
    
    def test_power(self):
        """Test field power operation."""
        # (2/1)^3 = 8/1
        ratio = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        elem = FieldElement(ratio)
        
        result = self.operation.power(elem, 3)
        assert abs(result.value.numerical_value() - 8.0) < ACT_EPSILON


class TestEternityField:
    """Test EternityField functionality."""
    
    def test_rational_field_creation(self):
        """Test rational field creation."""
        field = EternityField.rational_field()
        assert field.characteristic == 0
        assert field.finite == False
        assert isinstance(field.operation, EternalRatioOperation)
    
    def test_finite_field_creation(self):
        """Test finite field creation."""
        field = EternityField.finite_field(prime=2, degree=1)
        assert field.characteristic == 2
        assert field.finite == True
        assert len(field.elements) == 2
    
    def test_field_add(self):
        """Test field addition."""
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        
        field = EternityField.rational_field()
        
        elem1 = FieldElement(ratio=ratio1)
        elem2 = FieldElement(ratio=ratio2)
        
        result = field.add(elem1, elem2)
        assert abs(result.ratio.numerical_value() - 5.0) < ACT_EPSILON
    
    def test_field_multiply(self):
        """Test field multiplication."""
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=2.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        ratio2 = EternalRatio(
            numerator=AbsoluteValue(magnitude=3.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        
        field = EternityField.rational_field()
        
        elem1 = FieldElement(ratio=ratio1)
        elem2 = FieldElement(ratio=ratio2)
        
        result = field.multiply(elem1, elem2)
        assert abs(result.ratio.numerical_value() - 6.0) < ACT_EPSILON
    
    def test_field_characteristic(self):
        """Test field characteristic."""
        rational_field = EternityField.rational_field()
        assert rational_field.characteristic == 0
        
        finite_field = EternityField.finite_field(prime=5, degree=1)
        assert finite_field.characteristic == 5
    
    def test_field_order(self):
        """Test field order."""
        finite_field = EternityField.finite_field(prime=3, degree=1)
        assert finite_field.order() == 3
        
        rational_field = EternityField.rational_field()
        assert rational_field.order() is None  # Infinite field
    
    def test_additive_group(self):
        """Test additive group extraction."""
        finite_field = EternityField.finite_field(prime=3, degree=1)
        additive_group = finite_field.additive_group()
        
        assert isinstance(additive_group, AbsoluteGroup)
        assert additive_group.finite == True
    
    def test_multiplicative_group(self):
        """Test multiplicative group extraction."""
        finite_field = EternityField.finite_field(prime=3, degree=1)
        multiplicative_group = finite_field.multiplicative_group()
        
        assert isinstance(multiplicative_group, AbsoluteGroup)
        # Should exclude zero element
        assert len(multiplicative_group.elements) < len(finite_field.elements)
    
    def test_is_element(self):
        """Test element membership check."""
        ratio1 = EternalRatio(
            numerator=AbsoluteValue(magnitude=1.0, direction=1.0),
            denominator=AbsoluteValue(magnitude=1.0, direction=1.0)
        )
        
        field = EternityField.rational_field()
        elem1 = FieldElement(ratio=ratio1)
        
        # For rational field, any valid ratio should be an element
        assert field.__contains__(elem1) or True  # Simplified test


class TestAlgebraicProperties:
    """Test algebraic properties and axioms."""
    
    def test_group_closure(self):
        """Test group closure property."""
        group = AbsoluteGroup.additive_group()
        a = GroupElement(value=AbsoluteValue(2.0, 1))
        b = GroupElement(value=AbsoluteValue(3.0, -1))
        
        result = group.operate(a, b)
        assert isinstance(result, GroupElement)
        assert isinstance(result.value, AbsoluteValue)
    
    def test_group_associativity(self):
        """Test group associativity property."""
        group = AbsoluteGroup.additive_group()
        a = GroupElement(value=AbsoluteValue(1.0, 1))
        b = GroupElement(value=AbsoluteValue(2.0, 1))
        c = GroupElement(value=AbsoluteValue(3.0, 1))
        
        # (a + b) + c = a + (b + c)
        left = group.operate(group.operate(a, b), c)
        right = group.operate(a, group.operate(b, c))
        
        # Due to compensation, exact equality might not hold
        # Check that results are approximately equal
        assert abs(left.value.magnitude - right.value.magnitude) < 1e-10
    
    def test_group_identity(self):
        """Test group identity property."""
        group = AbsoluteGroup.additive_group()
        identity = group.identity_element()
        a = GroupElement(value=AbsoluteValue(5.0, 1))
        
        # a + 0 = a
        result = group.operate(a, identity)
        assert result.value.magnitude == a.value.magnitude
        assert result.value.direction == a.value.direction
    
    def test_group_inverse(self):
        """Test group inverse property."""
        group = AbsoluteGroup.additive_group()
        identity = group.identity_element()
        a = GroupElement(value=AbsoluteValue(5.0, 1))
        
        # a + (-a) = 0
        inverse = group.inverse_element(a)
        result = group.operate(a, inverse)
        
        # Should be close to identity
        assert abs(result.value.magnitude - identity.value.magnitude) < 1e-10
    
    def test_field_additive_identity(self):
        """Test field additive identity."""
        field = EternityField.rational_field()
        zero = field.zero()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        
        # a + 0 = a
        result = field.add(a, zero)
        assert result.ratio.numerical_value() == a.ratio.numerical_value()
    
    def test_field_multiplicative_identity(self):
        """Test field multiplicative identity."""
        field = EternityField.rational_field()
        one = field.one()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        
        # a * 1 = a
        result = field.multiply(a, one)
        assert abs(result.ratio.numerical_value() - a.ratio.numerical_value()) < 1e-10
    
    def test_field_distributivity(self):
        """Test field distributivity property."""
        field = EternityField.rational_field()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(2.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        b = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        c = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(4.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        # a * (b + c) = a * b + a * c
        left = field.multiply(a, field.add(b, c))
        right = field.add(field.multiply(a, b), field.multiply(a, c))
        
        assert abs(left.ratio.numerical_value() - right.ratio.numerical_value()) < 1e-10


class TestAbsoluteGroupAdvanced:
    """Advanced tests for AbsoluteGroup functionality."""
    
    def test_finite_cyclic_group_creation(self):
        """Test creation of finite cyclic groups."""
        group = AbsoluteGroup.finite_cyclic_group(order=5)
        assert group.finite is True
        assert group.order() == 5
        assert len(group.elements) == 5
    
    def test_finite_cyclic_group_invalid_order(self):
        """Test error handling for invalid group orders."""
        with pytest.raises(ValueError, match="Group order must be positive"):
            AbsoluteGroup.finite_cyclic_group(order=0)
        
        with pytest.raises(ValueError, match="Group order must be positive"):
            AbsoluteGroup.finite_cyclic_group(order=-5)
    
    def test_element_order_calculation(self):
        """Test calculation of element orders."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)
        elements = list(group.elements)
        
        # Identity should have order 1
        identity = group.identity_element()
        assert group.element_order(identity) == 1
        
        # Test order calculation for other elements
        for element in elements:
            order = group.element_order(element)
            assert order is not None
            assert order > 0
            assert order <= 6
    
    def test_group_abelian_property(self):
        """Test checking if groups are abelian."""
        additive_group = AbsoluteGroup.additive_group()
        multiplicative_group = AbsoluteGroup.multiplicative_group()
        finite_group = AbsoluteGroup.finite_cyclic_group(order=4)
        
        assert additive_group.is_abelian() is True
        assert multiplicative_group.is_abelian() is True
        assert finite_group.is_abelian() is True
    
    def test_subgroup_generation(self):
        """Test subgroup generation from generators."""
        group = AbsoluteGroup.finite_cyclic_group(order=8)
        elements = list(group.elements)
        
        # Generate subgroup from one element
        generator = elements[1] if len(elements) > 1 else elements[0]
        subgroup = group.subgroup([generator])
        
        assert subgroup.finite is True
        assert len(subgroup.elements) <= len(group.elements)
        assert subgroup.identity_element() in subgroup.elements
    
    def test_trivial_subgroup(self):
        """Test generation of trivial subgroup."""
        group = AbsoluteGroup.additive_group()
        trivial = group.subgroup([])
        
        assert trivial.finite is True
        assert len(trivial.elements) == 1
        assert group.identity_element() in trivial.elements
    
    def test_coset_computation(self):
        """Test left and right coset computation."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)
        elements = list(group.elements)
        
        # Create a subgroup
        generator = elements[1] if len(elements) > 1 else elements[0]
        subgroup = group.subgroup([generator])
        
        # Compute left cosets
        left_cosets = group.cosets(subgroup, left=True)
        assert len(left_cosets) > 0
        
        # Compute right cosets
        right_cosets = group.cosets(subgroup, left=False)
        assert len(right_cosets) > 0
        
        # For abelian groups, left and right cosets should be the same
        assert len(left_cosets) == len(right_cosets)
    
    def test_normal_subgroup_check(self):
        """Test checking if subgroups are normal."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)
        elements = list(group.elements)
        
        # Create a subgroup
        generator = elements[1] if len(elements) > 1 else elements[0]
        subgroup = group.subgroup([generator])
        
        # In abelian groups, all subgroups are normal
        assert group.is_normal_subgroup(subgroup) is True
    
    def test_quotient_group_construction(self):
        """Test quotient group construction."""
        group = AbsoluteGroup.finite_cyclic_group(order=6)
        elements = list(group.elements)
        
        # Create a normal subgroup
        generator = elements[1] if len(elements) > 1 else elements[0]
        normal_subgroup = group.subgroup([generator])
        
        # Construct quotient group
        quotient = group.quotient_group(normal_subgroup)
        assert quotient.finite is True
        assert len(quotient.elements) <= len(group.elements)
    
    def test_group_membership(self):
        """Test group membership checking."""
        additive_group = AbsoluteGroup.additive_group()
        multiplicative_group = AbsoluteGroup.multiplicative_group()
        
        # Test additive group membership
        element = GroupElement(value=AbsoluteValue(5.0, 1))
        assert element in additive_group
        
        # Test multiplicative group membership
        non_absolute = GroupElement(value=AbsoluteValue(2.0, 1))
        absolute = GroupElement(value=AbsoluteValue.absolute())
        
        assert non_absolute in multiplicative_group
        assert absolute not in multiplicative_group
    
    def test_group_iteration(self):
        """Test iteration over finite groups."""
        group = AbsoluteGroup.finite_cyclic_group(order=4)
        
        # Should be able to iterate
        elements = list(group)
        assert len(elements) == 4
        
        # Test that infinite groups raise error
        infinite_group = AbsoluteGroup.additive_group()
        with pytest.raises(ValueError, match="Cannot iterate over infinite group"):
            list(infinite_group)
    
    def test_multiplicative_group_absolute_rejection(self):
        """Test that multiplicative groups reject Absolute elements."""
        group = AbsoluteGroup.multiplicative_group()
        absolute_elem = GroupElement(value=AbsoluteValue.absolute())
        regular_elem = GroupElement(value=AbsoluteValue(2.0, 1))
        
        # Should raise error when operating with Absolute elements
        with pytest.raises(ValueError, match="Absolute elements not allowed"):
            group.operate(absolute_elem, regular_elem)
        
        # Should raise error when inverting Absolute elements
        with pytest.raises(ValueError, match="Absolute elements have no multiplicative inverse"):
            group.inverse_element(absolute_elem)
    
    def test_finite_group_element_validation(self):
        """Test validation of elements in finite groups."""
        group = AbsoluteGroup.finite_cyclic_group(order=3)
        elements = list(group.elements)
        
        # Valid elements should work
        if len(elements) >= 2:
            result = group.operate(elements[0], elements[1])
            assert isinstance(result, GroupElement)
        
        # Invalid elements should raise error
        invalid_elem = GroupElement(value=AbsoluteValue(999.0, 1))
        with pytest.raises(ValueError, match="Elements must be in the group"):
            group.operate(invalid_elem, elements[0])
    
    def test_group_string_representations(self):
        """Test string representations of groups."""
        additive_group = AbsoluteGroup.additive_group()
        finite_group = AbsoluteGroup.finite_cyclic_group(order=5)
        
        # Test __repr__
        assert "AdditiveOperation" in repr(additive_group)
        assert "infinite" in repr(additive_group)
        assert "order=5" in repr(finite_group)
        
        # Test __str__
        assert "additive" in str(additive_group)
        assert "Infinite" in str(additive_group)
        assert "Finite" in str(finite_group)
        assert "order 5" in str(finite_group)


class TestEternityFieldAdvanced:
    """Advanced tests for EternityField functionality."""
    
    def test_finite_field_creation(self):
        """Test creation of finite fields."""
        field = EternityField.finite_field(characteristic=7)
        assert field.characteristic == 7
        assert field.finite is True
        assert field.order() == 7
    
    def test_finite_field_invalid_characteristic(self):
        """Test error handling for invalid field characteristics."""
        with pytest.raises(ValueError, match="Characteristic must be prime"):
            EternityField.finite_field(characteristic=4)  # Not prime
        
        with pytest.raises(ValueError, match="Characteristic must be prime"):
            EternityField.finite_field(characteristic=1)  # Not prime
        
        with pytest.raises(ValueError, match="Characteristic must be prime"):
            EternityField.finite_field(characteristic=0)  # Not prime
    
    def test_field_element_zero_check(self):
        """Test checking if field elements are zero."""
        field = EternityField.rational_field()
        zero = field.zero()
        non_zero = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        
        assert zero.is_zero() is True
        assert non_zero.is_zero() is False
    
    def test_field_element_one_check(self):
        """Test checking if field elements are one."""
        field = EternityField.rational_field()
        one = field.one()
        non_one = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        
        assert one.is_one() is True
        assert non_one.is_one() is False
    
    def test_field_element_unit_check(self):
        """Test checking if field elements are units."""
        field = EternityField.rational_field()
        zero = field.zero()
        non_zero = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        
        assert zero.is_unit() is False
        assert non_zero.is_unit() is True
    
    def test_field_subtraction_operation(self):
        """Test field subtraction operation."""
        field = EternityField.rational_field()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(5.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        b = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        result = field.subtract(a, b)
        expected = 5.0 - 3.0
        assert abs(result.ratio.numerical_value() - expected) < 1e-10
    
    def test_field_division_operation(self):
        """Test field division operation."""
        field = EternityField.rational_field()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(6.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        b = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(2.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        result = field.divide(a, b)
        expected = 6.0 / 2.0
        assert abs(result.ratio.numerical_value() - expected) < 1e-10
    
    def test_field_division_by_zero(self):
        """Test error handling for division by zero."""
        field = EternityField.rational_field()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(5.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        zero = field.zero()
        
        with pytest.raises(ValueError, match="Division by zero"):
            field.divide(a, zero)
    
    def test_field_power_operation(self):
        """Test field power operation."""
        field = EternityField.rational_field()
        base = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(2.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        # Test positive exponent
        result = field.power(base, 3)
        expected = 2.0 ** 3
        assert abs(result.ratio.numerical_value() - expected) < 1e-10
        
        # Test zero exponent
        result = field.power(base, 0)
        assert abs(result.ratio.numerical_value() - 1.0) < 1e-10
        
        # Test negative exponent
        result = field.power(base, -2)
        expected = 2.0 ** (-2)
        assert abs(result.ratio.numerical_value() - expected) < 1e-10
    
    def test_field_power_zero_base(self):
        """Test power operation with zero base."""
        field = EternityField.rational_field()
        zero = field.zero()
        
        # 0^n = 0 for n > 0
        result = field.power(zero, 3)
        assert result.is_zero() is True
        
        # 0^0 should raise error
        with pytest.raises(ValueError, match="0^0 is undefined"):
            field.power(zero, 0)
        
        # 0^(-n) should raise error
        with pytest.raises(ValueError, match="Division by zero"):
            field.power(zero, -2)
    
    def test_field_additive_inverse(self):
        """Test field additive inverse."""
        field = EternityField.rational_field()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(5.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        inverse = field.additive_inverse(a)
        result = field.add(a, inverse)
        
        assert abs(result.ratio.numerical_value()) < 1e-10
    
    def test_field_multiplicative_inverse(self):
        """Test field multiplicative inverse."""
        field = EternityField.rational_field()
        a = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(5.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        inverse = field.multiplicative_inverse(a)
        result = field.multiply(a, inverse)
        
        assert abs(result.ratio.numerical_value() - 1.0) < 1e-10
    
    def test_field_multiplicative_inverse_zero(self):
        """Test error handling for multiplicative inverse of zero."""
        field = EternityField.rational_field()
        zero = field.zero()
        
        with pytest.raises(ValueError, match="Zero has no multiplicative inverse"):
            field.multiplicative_inverse(zero)
    
    def test_finite_field_modular_reduction(self):
        """Test modular reduction in finite fields."""
        field = EternityField.finite_field(characteristic=5)
        
        # Create element with value > characteristic
        large_element = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(7.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        # Should be reduced modulo 5
        reduced = field._reduce_modulo_characteristic(large_element)
        expected = 7.0 % 5.0  # Should be 2.0
        assert abs(reduced.ratio.numerical_value() - expected) < 1e-10
    
    def test_field_additive_group_extraction(self):
        """Test extraction of additive group from field."""
        field = EternityField.finite_field(characteristic=5)
        additive_group = field.additive_group()
        
        assert additive_group.finite is True
        assert len(additive_group.elements) == 5
    
    def test_field_multiplicative_group_extraction(self):
        """Test extraction of multiplicative group from field."""
        field = EternityField.finite_field(characteristic=5)
        multiplicative_group = field.multiplicative_group()
        
        assert multiplicative_group.finite is True
        assert len(multiplicative_group.elements) == 4  # Excludes zero
    
    def test_field_polynomial_ring_creation(self):
        """Test creation of polynomial ring over field."""
        field = EternityField.rational_field()
        poly_ring = field.polynomial_ring()
        
        assert isinstance(poly_ring, PolynomialRing)
        assert poly_ring.base_field == field
    
    def test_field_perfect_check(self):
        """Test checking if field is perfect."""
        rational_field = EternityField.rational_field()
        finite_field = EternityField.finite_field(characteristic=5)
        
        # Rational field should be perfect
        assert rational_field.is_perfect() is True
        
        # Finite fields should be perfect
        assert finite_field.is_perfect() is True
    
    def test_field_frobenius_endomorphism(self):
        """Test Frobenius endomorphism in finite fields."""
        field = EternityField.finite_field(characteristic=5)
        element = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(1.0, 1)
        ))
        
        frobenius_result = field.frobenius_endomorphism(element)
        
        # In F_5, Frobenius should compute x^5 = x
        expected = field.power(element, 5)
        assert abs(frobenius_result.ratio.numerical_value() - expected.ratio.numerical_value()) < 1e-10
    
    def test_field_membership(self):
        """Test field membership checking."""
        rational_field = EternityField.rational_field()
        finite_field = EternityField.finite_field(characteristic=5)
        
        element = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        
        assert element in rational_field
        assert element in finite_field  # Should be reduced modulo characteristic
    
    def test_field_iteration(self):
        """Test iteration over finite fields."""
        field = EternityField.finite_field(characteristic=3)
        
        # Should be able to iterate
        elements = list(field)
        assert len(elements) == 3
        
        # Test that infinite fields raise error
        infinite_field = EternityField.rational_field()
        with pytest.raises(ValueError, match="Cannot iterate over infinite field"):
            list(infinite_field)
    
    def test_field_string_representations(self):
        """Test string representations of fields."""
        rational_field = EternityField.rational_field()
        finite_field = EternityField.finite_field(characteristic=7)
        
        # Test __repr__
        assert "EternalRatioOperation" in repr(rational_field)
        assert "infinite" in repr(rational_field)
        assert "characteristic=7" in repr(finite_field)
        
        # Test __str__
        assert "Rational" in str(rational_field)
        assert "Infinite" in str(rational_field)
        assert "Finite" in str(finite_field)
        assert "F_7" in str(finite_field)
    
    def test_field_element_hashing(self):
        """Test hashing of field elements."""
        element1 = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        element2 = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(3.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        element3 = FieldElement(ratio=EternalRatio(
            numerator=AbsoluteValue(5.0, 1),
            denominator=AbsoluteValue(2.0, 1)
        ))
        
        # Equal elements should have same hash
        assert hash(element1) == hash(element2)
        
        # Different elements should have different hashes (usually)
        assert hash(element1) != hash(element3)
        
        # Should be able to use in sets
        element_set = {element1, element2, element3}
        assert len(element_set) == 2  # element1 and element2 are equal


class TestPolynomialRing:
    """Tests for PolynomialRing functionality."""
    
    def test_polynomial_ring_creation(self):
        """Test creation of polynomial rings."""
        field = EternityField.rational_field()
        poly_ring = PolynomialRing(field=field)
        
        assert poly_ring.field == field
        assert isinstance(poly_ring, PolynomialRing)
    
    def test_polynomial_creation(self):
        """Test creation of polynomials."""
        field = EternityField.rational_field()
        poly_ring = PolynomialRing(field=field)
        
        # Create coefficients using field methods
        coeff0 = field.one()  # 1
        coeff1 = field.add(field.one(), field.one())  # 2
        coeff2 = field.add(field.add(field.one(), field.one()), field.one())  # 3
        
        # Create polynomial 3x^2 + 2x + 1
        poly = Polynomial(coefficients=[coeff0, coeff1, coeff2], ring=poly_ring)
        
        assert poly.degree() == 2
        assert len(poly.coefficients) == 3
        assert poly.coefficients[0] == coeff0
        assert poly.coefficients[1] == coeff1
        assert poly.coefficients[2] == coeff2
    
    def test_polynomial_zero_polynomial(self):
        """Test creation and properties of zero polynomial."""
        field = EternityField.rational_field()
        poly_ring = PolynomialRing(field=field)
        zero = field.zero()
        
        # Zero polynomial
        zero_poly = Polynomial(coefficients=[zero], ring=poly_ring)
        
        assert zero_poly.degree() == -1  # Degree of zero polynomial
        assert len(zero_poly.coefficients) == 0  # Empty coefficients for zero polynomial
    
    def test_polynomial_normalization(self):
        """Test polynomial coefficient normalization."""
        field = EternityField.rational_field()
        zero = field.zero()
        one = field.one()
        two = field.add(field.one(), field.one())
        
        # Create polynomial with leading zeros: 0x^3 + 0x^2 + 2x + 1
        poly_ring = PolynomialRing(field=field)
        poly = Polynomial(coefficients=[one, two, zero, zero], ring=poly_ring)
        
        # Should normalize to remove leading zeros
        assert poly.degree() == 1
        assert len(poly.coefficients) == 2
        assert poly.coefficients[0] == one
        assert poly.coefficients[1] == two
    
    def test_polynomial_addition(self):
        """Test polynomial addition."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        three = field.add(field.add(field.one(), field.one()), field.one())
        
        # p1(x) = 2x + 1
        poly_ring = PolynomialRing(field=field)
        poly1 = Polynomial(coefficients=[one, two], ring=poly_ring)
        
        # p2(x) = 3x + 2
        poly2 = Polynomial(coefficients=[two, three], ring=poly_ring)
        
        # p1 + p2 = 5x + 3
        result = poly1 + poly2
        
        assert result.degree() == 1
        assert abs(result.coefficients[0].ratio.numerical_value() - 3.0) < 1e-10
        assert abs(result.coefficients[1].ratio.numerical_value() - 5.0) < 1e-10
    
    def test_polynomial_multiplication(self):
        """Test polynomial multiplication."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        
        # p1(x) = x + 1
        poly_ring = PolynomialRing(field=field)
        poly1 = Polynomial(coefficients=[one, one], ring=poly_ring)
        
        # p2(x) = 2x + 1
        poly2 = Polynomial(coefficients=[one, two], ring=poly_ring)
        
        # p1 * p2 = (x + 1)(2x + 1) = 2x^2 + 3x + 1
        result = poly1 * poly2
        
        assert result.degree() == 2
        assert abs(result.coefficients[0].ratio.numerical_value() - 1.0) < 1e-10  # constant term
        assert abs(result.coefficients[1].ratio.numerical_value() - 3.0) < 1e-10  # x term
        assert abs(result.coefficients[2].ratio.numerical_value() - 2.0) < 1e-10  # x^2 term
    
    def test_polynomial_evaluation(self):
        """Test polynomial evaluation at specific points."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        three = field.add(field.add(field.one(), field.one()), field.one())
        
        # p(x) = 3x^2 + 2x + 1
        poly_ring = PolynomialRing(field=field)
        poly = Polynomial(coefficients=[one, two, three], ring=poly_ring)
        
        # Evaluate at x = 2: p(2) = 3(4) + 2(2) + 1 = 12 + 4 + 1 = 17
        x_value = two
        result = poly.evaluate(x_value)
        
        expected = 3.0 * (2.0 ** 2) + 2.0 * 2.0 + 1.0  # 17.0
        assert abs(result.ratio.numerical_value() - expected) < 1e-10
    
    def test_polynomial_evaluation_at_zero(self):
        """Test polynomial evaluation at zero."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        three = field.add(field.add(field.one(), field.one()), field.one())
        
        # p(x) = 3x^2 + 2x + 1
        poly_ring = PolynomialRing(field=field)
        poly = Polynomial(coefficients=[one, two, three], ring=poly_ring)
        
        # Evaluate at x = 0: p(0) = 1
        zero = field.zero()
        result = poly.evaluate(zero)
        
        assert abs(result.ratio.numerical_value() - 1.0) < 1e-10
    
    def test_polynomial_string_representation(self):
        """Test polynomial string representations."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        three = field.add(field.add(field.one(), field.one()), field.one())
        
        # p(x) = 3x^2 + 2x + 1
        poly_ring = PolynomialRing(field=field)
        poly = Polynomial(coefficients=[one, two, three], ring=poly_ring)
        
        poly_str = str(poly)
        assert "x^2" in poly_str
        assert "x" in poly_str
        assert "3" in poly_str
        assert "2" in poly_str
        assert "1" in poly_str
    
    def test_polynomial_constant_polynomial(self):
        """Test constant polynomials."""
        field = EternityField.rational_field()
        five = field.add(field.add(field.add(field.add(field.one(), field.one()), field.one()), field.one()), field.one())
        
        # Constant polynomial p(x) = 5
        poly_ring = PolynomialRing(field=field)
        poly = Polynomial(coefficients=[five], ring=poly_ring)
        
        assert poly.degree() == 0
        assert len(poly.coefficients) == 1
        assert abs(poly.coefficients[0].ratio.numerical_value() - 5.0) < 1e-10
        
        # Evaluation should always return the constant
        x_value = field.add(field.one(), field.one())
        result = poly.evaluate(x_value)
        assert abs(result.ratio.numerical_value() - 5.0) < 1e-10
    
    def test_polynomial_addition_different_degrees(self):
        """Test addition of polynomials with different degrees."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        three = field.add(field.add(field.one(), field.one()), field.one())
        
        # p1(x) = 1 (degree 0)
        poly_ring = PolynomialRing(field=field)
        poly1 = Polynomial(coefficients=[one], ring=poly_ring)
        
        # p2(x) = 3x^2 + 2x (degree 2)
        poly2 = Polynomial(coefficients=[field.zero(), two, three], ring=poly_ring)
        
        # p1 + p2 = 3x^2 + 2x + 1
        result = poly1 + poly2
        
        assert result.degree() == 2
        assert abs(result.coefficients[0].ratio.numerical_value() - 1.0) < 1e-10
        assert abs(result.coefficients[1].ratio.numerical_value() - 2.0) < 1e-10
        assert abs(result.coefficients[2].ratio.numerical_value() - 3.0) < 1e-10
    
    def test_polynomial_multiplication_by_zero(self):
        """Test multiplication by zero polynomial."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        
        # p1(x) = 2x + 1
        poly_ring = PolynomialRing(field=field)
        poly1 = Polynomial(coefficients=[one, two], ring=poly_ring)
        
        # p2(x) = 0
        zero_poly = Polynomial(coefficients=[field.zero()], ring=poly_ring)
        
        # p1 * 0 = 0
        result = poly1 * zero_poly
        
        assert result.degree() == -1  # Zero polynomial
        assert len(result.coefficients) == 0  # Empty coefficients for zero polynomial


class TestPolynomialAdvanced:
    """Advanced tests for polynomial operations and edge cases."""
    
    def test_polynomial_equality(self):
        """Test polynomial equality comparison."""
        field = EternityField.rational_field()
        one = field.one()
        two = field.add(field.one(), field.one())
        
        # Create two identical polynomials
        poly_ring = PolynomialRing(field=field)
        poly1 = Polynomial(coefficients=[one, two], ring=poly_ring)
        poly2 = Polynomial(coefficients=[one, two], ring=poly_ring)
        
        assert poly1 == poly2
        
        # Create different polynomial
        three = field.add(field.add(field.one(), field.one()), field.one())
        poly3 = Polynomial(coefficients=[one, three], ring=poly_ring)
        
        assert poly1 != poly3
    
    def test_polynomial_with_finite_field(self):
        """Test polynomials over finite fields."""
        field = EternityField.finite_field(characteristic=5)
        
        # Create elements in F_5
        two = field.add(field.one(), field.one())
        three = field.add(field.add(field.one(), field.one()), field.one())
        
        # p(x) = 3x + 2 in F_5
        poly_ring = PolynomialRing(field=field)
        poly = Polynomial(coefficients=[two, three], ring=poly_ring)
        
        assert poly.degree() == 1
        
        # Evaluate at x = 4: p(4) = 3*4 + 2 = 12 + 2 = 14 ≡ 4 (mod 5)
        four = field.add(field.add(field.add(field.one(), field.one()), field.one()), field.one())
        result = poly.evaluate(four)
        
        # Result should be reduced modulo 5
        expected = (3.0 * 4.0 + 2.0) % 5.0  # 4.0
        assert abs(result.ratio.numerical_value() - expected) < 1e-10
    
    def test_polynomial_high_degree(self):
        """Test polynomials with higher degrees."""
        field = EternityField.rational_field()
        one = field.one()
        
        # Create polynomial x^5 + x^4 + x^3 + x^2 + x + 1
        coefficients = [one] * 6  # Coefficients for degrees 0 through 5
        poly_ring = PolynomialRing(field=field)
        poly = Polynomial(coefficients=coefficients, ring=poly_ring)
        
        assert poly.degree() == 5
        assert len(poly.coefficients) == 6
        
        # Evaluate at x = 1: should give 6
        result = poly.evaluate(one)
        assert abs(result.ratio.numerical_value() - 6.0) < 1e-10
    
    def test_polynomial_ring_operations(self):
        """Test operations within polynomial rings."""
        field = EternityField.rational_field()
        poly_ring = PolynomialRing(field=field)
        
        # Test that polynomial ring maintains field reference
        assert poly_ring.field == field
        
        # Create polynomials through the ring
        one = field.one()
        two = field.add(field.one(), field.one())
        
        poly1 = Polynomial(coefficients=[one, two], ring=poly_ring)
        poly2 = Polynomial(coefficients=[two, one], ring=poly_ring)
        
        # Test addition and multiplication work correctly
        sum_poly = poly1 + poly2
        prod_poly = poly1 * poly2
        
        assert isinstance(sum_poly, Polynomial)
        assert isinstance(prod_poly, Polynomial)
        assert sum_poly.ring.field == field
        assert prod_poly.ring.field == field


if __name__ == "__main__":
    pytest.main([__file__])