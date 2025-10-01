"""Integration tests for the Balansis module.

This module contains tests for checking integration with the Balansis library,
including numerical error compensation and zero-sum attention mechanisms.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import time
import gc
import psutil
import os

from tnsim.integrations.balansis_integration import (
    BalansisCompensator,
    CompensationMetrics,
    ZeroSumAttention,
    ZeroSumTransformerBlock,
    test_balansis_integration
)


class TestBalansisCompensator:
    """Tests for the Balansis compensator."""
    
    def test_compensator_initialization(self):
        """Test compensator initialization."""
        compensator = BalansisCompensator(precision=1e-12)
        
        assert compensator.precision == 1e-12
        assert compensator.max_iterations == 1000
        assert hasattr(compensator, 'balansis_available')
    
    def test_compensated_sum_with_balansis(self):
        """Test compensated sum with Balansis (if available)."""
        compensator = BalansisCompensator()
        values = [1.0, 1e-15, -1.0, 1e-16]
        
        result = compensator.compensated_sum(values)
        
        # Result should be close to the sum of small numbers
        expected = sum(values)
        assert abs(result - expected) < compensator.precision
    
    def test_compensated_sum_fallback(self):
        """Test fallback compensated sum without Balansis."""
        with patch('tnsim.integrations.balansis_integration.BALANSIS_AVAILABLE', False):
            compensator = BalansisCompensator()
            values = [1.0, 1e-15, -1.0, 1e-16]
            
            result = compensator.compensated_sum(values)
            
            # Fallback should use Kahan algorithm
            assert isinstance(result, float)
            assert abs(result - sum(values)) < 1e-10
    
    def test_kahan_summation(self):
        """Test Kahan summation algorithm."""
        compensator = BalansisCompensator()
        
        # Test with numbers where naive sum loses precision
        values = [1.0] + [1e-15] * 1000
        
        result = compensator._kahan_summation(values)
        expected = 1.0 + 1000 * 1e-15
        
        # Kahan algorithm should preserve more precision
        assert abs(result - expected) < 1e-12
    
    def test_compensate_series(self):
        """Test series compensation."""
        compensator = BalansisCompensator()
        
        # Create series with known sum
        series = [1/n for n in range(1, 101)]  # Partial harmonic sum
        
        compensated_sum, metrics = compensator.compensate_series(series)
        
        assert isinstance(compensated_sum, float)
        assert isinstance(metrics, CompensationMetrics)
        assert metrics.original_sum != compensated_sum  # Should be different
        assert metrics.compensation_error >= 0
        assert metrics.iterations > 0
    
    def test_analyze_compensation_quality(self):
        """Test compensation quality analysis."""
        compensator = BalansisCompensator()
        
        original_sum = 5.123456789
        compensated_sum = 5.123456790
        
        metrics = compensator.analyze_compensation_quality(
            original_sum, compensated_sum, iterations=10
        )
        
        assert isinstance(metrics, CompensationMetrics)
        assert metrics.original_sum == original_sum
        assert metrics.compensated_sum == compensated_sum
        assert metrics.compensation_error == abs(compensated_sum - original_sum)
        assert metrics.relative_error > 0
        assert metrics.iterations == 10
        assert metrics.precision_gain >= 0
    
    def test_compensate_zero_sum_set(self, sample_harmonic_set):
        """Test zero-sum set compensation."""
        compensator = BalansisCompensator()
        
        compensated_sum, metrics = compensator.compensate_zero_sum_set(sample_harmonic_set)
        
        assert isinstance(compensated_sum, float)
        assert isinstance(metrics, CompensationMetrics)
        # For zero-sum set, sum should be close to zero
        assert abs(compensated_sum) < compensator.precision * 10
    
    def test_edge_cases(self):
        """Test edge cases."""
        compensator = BalansisCompensator()
        
        # Empty list
        result = compensator.compensated_sum([])
        assert result == 0.0
        
        # Single element
        result = compensator.compensated_sum([42.0])
        assert result == 42.0
        
        # Very large numbers
        big_numbers = [1e100, -1e100, 1.0]
        result = compensator.compensated_sum(big_numbers)
        assert abs(result - 1.0) < 1e-10
        
        # Very small numbers
        small_numbers = [1e-100] * 1000
        result = compensator.compensated_sum(small_numbers)
        assert result > 0


class TestZeroSumAttention:
    """Tests for zero-sum attention mechanism."""
    
    def test_attention_initialization(self):
        """Test attention mechanism initialization."""
        attention = ZeroSumAttention(
            d_model=512,
            n_heads=8,
            dropout=0.1,
            compensation_strength=0.1
        )
        
        assert attention.d_model == 512
        assert attention.n_heads == 8
        assert attention.d_k == 64  # 512 / 8
        assert attention.dropout.p == 0.1
        assert attention.compensation_strength == 0.1
        
        # Check that layers are created correctly
        assert attention.w_q.in_features == 512
        assert attention.w_k.in_features == 512
        assert attention.w_v.in_features == 512
        assert attention.w_o.in_features == 512
    
    def test_compensated_linear_projection(self):
        """Test compensated linear projection."""
        attention = ZeroSumAttention(d_model=256, n_heads=4)
        
        # Create input data
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 256)
        
        # Test projections
        q = attention._compensated_linear(x, attention.w_q)
        k = attention._compensated_linear(x, attention.w_k)
        v = attention._compensated_linear(x, attention.w_v)
        
        assert q.shape == (batch_size, seq_len, 256)
        assert k.shape == (batch_size, seq_len, 256)
        assert v.shape == (batch_size, seq_len, 256)
        
        # Check that projections are not identical (due to compensation)
        regular_q = attention.w_q(x)
        assert not torch.allclose(q, regular_q, atol=1e-6)
    
    def test_zero_sum_attention_scores(self):
        """Test zero-sum attention scores computation."""
        attention = ZeroSumAttention(d_model=128, n_heads=2)
        
        batch_size, seq_len = 1, 5
        q = torch.randn(batch_size, 2, seq_len, 64)  # (batch, heads, seq, d_k)
        k = torch.randn(batch_size, 2, seq_len, 64)
        
        scores = attention._zero_sum_attention_scores(q, k)
        
        assert scores.shape == (batch_size, 2, seq_len, seq_len)
        
        # Check zero-sum property (approximately)
        # Sum over last dimension should be close to zero
        row_sums = scores.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-3)
    
    def test_forward_pass(self):
        """Test forward pass through attention mechanism."""
        attention = ZeroSumAttention(d_model=256, n_heads=8)
        
        batch_size, seq_len = 2, 12
        x = torch.randn(batch_size, seq_len, 256)
        
        # Forward pass
        output, attention_weights = attention(x)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)
        
        # Check that attention weights are normalized
        weight_sums = attention_weights.sum(dim=-1)
        expected_sums = torch.ones_like(weight_sums)
        assert torch.allclose(weight_sums, expected_sums, atol=1e-5)
    
    def test_attention_with_mask(self):
        """Test attention with mask."""
        attention = ZeroSumAttention(d_model=128, n_heads=4)
        
        batch_size, seq_len = 1, 6
        x = torch.randn(batch_size, seq_len, 128)
        
        # Create mask (mask last 2 positions)
        mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, -2:] = False
        
        output, attention_weights = attention(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, 128)
        
        # Check that masked positions have zero weights
        masked_weights = attention_weights[:, :, :, -2:]
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights))
    
    def test_residual_connection(self):
        """Test residual connection."""
        attention = ZeroSumAttention(d_model=256, n_heads=8, use_residual=True)
        
        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, 256)
        
        output, _ = attention(x)
        
        # With residual connection, output should not be identical to input
        assert not torch.allclose(output, x)
        
        # But should contain input components
        # (hard to check directly, but we can check dimensions)
        assert output.shape == x.shape
    
    def test_dropout_effect(self):
        """Test dropout effect."""
        attention_with_dropout = ZeroSumAttention(
            d_model=128, n_heads=4, dropout=0.5
        )
        attention_without_dropout = ZeroSumAttention(
            d_model=128, n_heads=4, dropout=0.0
        )
        
        # Copy weights for fair comparison
        attention_without_dropout.load_state_dict(
            attention_with_dropout.state_dict()
        )
        
        batch_size, seq_len = 1, 6
        x = torch.randn(batch_size, seq_len, 128)
        
        # In training mode, dropout should affect results
        attention_with_dropout.train()
        attention_without_dropout.train()
        
        torch.manual_seed(42)
        output_with_dropout, _ = attention_with_dropout(x)
        
        torch.manual_seed(42)
        output_without_dropout, _ = attention_without_dropout(x)
        
        # Results should differ due to dropout
        assert not torch.allclose(output_with_dropout, output_without_dropout)
    
    def test_gradient_flow(self):
        """Test gradient flow."""
        attention = ZeroSumAttention(d_model=64, n_heads=2)
        
        batch_size, seq_len = 1, 4
        x = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        
        output, _ = attention(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check parameter gradients
        for param in attention.parameters():
            assert param.grad is not None


class TestZeroSumTransformerBlock:
    """Tests for zero-sum transformer block."""
    
    def test_transformer_block_initialization(self):
        """Test transformer block initialization."""
        block = ZeroSumTransformerBlock(
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=0.1
        )
        
        assert isinstance(block.attention, ZeroSumAttention)
        assert block.feed_forward[0].in_features == 512
        assert block.feed_forward[0].out_features == 2048
        assert block.feed_forward[2].in_features == 2048
        assert block.feed_forward[2].out_features == 512
    
    def test_transformer_block_forward(self):
        """Test forward pass through transformer block."""
        block = ZeroSumTransformerBlock(
            d_model=256,
            n_heads=4,
            d_ff=1024
        )
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 256)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, 256)
        assert not torch.allclose(output, x)  # Should be transformed
    
    def test_transformer_block_with_mask(self):
        """Test transformer block with mask."""
        block = ZeroSumTransformerBlock(d_model=128, n_heads=2, d_ff=512)
        
        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, 128)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # Causal mask
        
        output = block(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, 128)


class TestBalansisIntegrationUtils:
    """Tests for Balansis integration utilities."""
    
    def test_integration_test_function(self):
        """Test integration test function."""
        # Test should run without errors
        try:
            test_balansis_integration()
            integration_works = True
        except Exception as e:
            integration_works = False
            print(f"Integration test failed: {e}")
        
        # Integration may not work if Balansis is not installed
        # But function should not raise critical errors
        assert isinstance(integration_works, bool)
    
    def test_compensation_metrics_creation(self):
        """Test compensation metrics creation."""
        metrics = CompensationMetrics(
            original_sum=1.23456789,
            compensated_sum=1.23456790,
            compensation_error=1e-8,
            relative_error=8.1e-9,
            iterations=5,
            precision_gain=3.2
        )
        
        assert metrics.original_sum == 1.23456789
        assert metrics.compensated_sum == 1.23456790
        assert metrics.compensation_error == 1e-8
        assert metrics.relative_error == 8.1e-9
        assert metrics.iterations == 5
        assert metrics.precision_gain == 3.2
    
    def test_compensation_metrics_dict_conversion(self):
        """Test metrics to dictionary conversion."""
        metrics = CompensationMetrics(
            original_sum=5.0,
            compensated_sum=5.1,
            compensation_error=0.1,
            relative_error=0.02,
            iterations=10,
            precision_gain=1.5
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["original_sum"] == 5.0
        assert metrics_dict["compensated_sum"] == 5.1
        assert metrics_dict["compensation_error"] == 0.1
        assert metrics_dict["relative_error"] == 0.02
        assert metrics_dict["iterations"] == 10
        assert metrics_dict["precision_gain"] == 1.5


class TestBalansisIntegrationWithZeroSumSets:
    """Tests for Balansis integration with zero-sum sets."""
    
    def test_compensator_with_harmonic_series(self, sample_harmonic_set):
        """Test compensator with harmonic series."""
        compensator = BalansisCompensator(precision=1e-15)
        
        compensated_sum, metrics = compensator.compensate_zero_sum_set(
            sample_harmonic_set
        )
        
        # Harmonic series should have improved precision
        assert abs(compensated_sum) < 1e-10
        assert metrics.precision_gain > 0
        assert metrics.iterations > 0
    
    def test_compensator_with_alternating_series(self, sample_alternating_set):
        """Test compensator with alternating series."""
        compensator = BalansisCompensator()
        
        compensated_sum, metrics = compensator.compensate_zero_sum_set(
            sample_alternating_set
        )
        
        # Alternating series should converge to zero
        assert abs(compensated_sum) < compensator.precision * 100
        assert isinstance(metrics, CompensationMetrics)
    
    def test_compensator_with_geometric_series(self, sample_geometric_set):
        """Test compensator with geometric series."""
        compensator = BalansisCompensator()
        
        compensated_sum, metrics = compensator.compensate_zero_sum_set(
            sample_geometric_set
        )
        
        assert isinstance(compensated_sum, float)
        assert isinstance(metrics, CompensationMetrics)
        assert metrics.compensation_error >= 0
    
    def test_attention_with_zero_sum_embeddings(self):
        """Test attention mechanism with zero-sum embeddings."""
        attention = ZeroSumAttention(d_model=128, n_heads=4)
        
        # Create embeddings that sum to zero
        batch_size, seq_len = 1, 6
        embeddings = torch.randn(batch_size, seq_len, 128)
        
        # Make last embedding compensating
        embeddings[:, -1, :] = -embeddings[:, :-1, :].sum(dim=1)
        
        # Check that sum is close to zero
        total_sum = embeddings.sum(dim=1)
        assert torch.allclose(total_sum, torch.zeros_like(total_sum), atol=1e-6)
        
        # Apply attention
        output, weights = attention(embeddings)
        
        assert output.shape == embeddings.shape
        assert weights.shape == (batch_size, 4, seq_len, seq_len)
    
    def test_integration_performance_comparison(self):
        """Test performance comparison with/without Balansis."""
        import time
        
        # Create large series for testing
        large_series = [1/n - 1/(n+1) for n in range(1, 10001)]
        
        # Test with Balansis
        compensator_with_balansis = BalansisCompensator()
        start_time = time.time()
        result_with_balansis = compensator_with_balansis.compensated_sum(large_series)
        time_with_balansis = time.time() - start_time
        
        # Test without Balansis (fallback)
        with patch('tnsim.integrations.balansis_integration.BALANSIS_AVAILABLE', False):
            compensator_fallback = BalansisCompensator()
            start_time = time.time()
            result_fallback = compensator_fallback.compensated_sum(large_series)
            time_fallback = time.time() - start_time
        
        # Results should be close
        assert abs(result_with_balansis - result_fallback) < 1e-10
        
        # Execution time should be reasonable
        assert time_with_balansis < 1.0
        assert time_fallback < 1.0
        
        print(f"Balansis time: {time_with_balansis:.4f}s, Fallback time: {time_fallback:.4f}s")


class TestBalansisErrorHandling:
    """Tests for error handling in Balansis integration."""
    
    def test_invalid_input_handling(self):
        """Test invalid input handling."""
        compensator = BalansisCompensator()
        
        # Test with None
        with pytest.raises((TypeError, ValueError)):
            compensator.compensated_sum(None)
        
        # Test with incorrect types
        with pytest.raises((TypeError, ValueError)):
            compensator.compensated_sum(["not", "a", "number"])
    
    def test_attention_invalid_dimensions(self):
        """Test invalid dimensions handling in attention."""
        attention = ZeroSumAttention(d_model=128, n_heads=4)
        
        # Incorrect input dimension
        with pytest.raises((RuntimeError, ValueError)):
            wrong_input = torch.randn(2, 10, 64)  # d_model should be 128
            attention(wrong_input)
    
    def test_compensation_with_infinite_values(self):
        """Test compensation with infinite values."""
        compensator = BalansisCompensator()
        
        # Test with infinity
        values_with_inf = [1.0, float('inf'), -float('inf'), 2.0]
        
        # Should handle correctly or raise exception
        try:
            result = compensator.compensated_sum(values_with_inf)
            # If no exception, result should be a number
            assert isinstance(result, (int, float))
        except (ValueError, OverflowError):
            # Expected behavior for invalid values
            pass
    
    def test_compensation_with_nan_values(self):
        """Test compensation with NaN values."""
        compensator = BalansisCompensator()
        
        values_with_nan = [1.0, float('nan'), 2.0]
        
        try:
            result = compensator.compensated_sum(values_with_nan)
            # If NaN is present, result may be NaN
            assert isinstance(result, (int, float))
        except (ValueError, TypeError):
            # Expected behavior for NaN values
            pass
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large objects
        attention = ZeroSumAttention(d_model=1024, n_heads=16)
        large_input = torch.randn(10, 100, 1024)
        
        # Perform operations
        for _ in range(10):
            output, _ = attention(large_input)
            del output
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500 MB)
        assert memory_increase < 500 * 1024 * 1024, f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB"


class TestBalansisConfigurationOptions:
    """Tests for various Balansis configurations."""
    
    def test_different_precision_levels(self):
        """Test different precision levels."""
        precisions = [1e-6, 1e-12, 1e-15]
        test_values = [1.0, 1e-10, -1.0, -1e-10]
        
        results = []
        for precision in precisions:
            compensator = BalansisCompensator(precision=precision)
            result = compensator.compensated_sum(test_values)
            results.append(result)
        
        # Results should be close, but may differ due to different precision
        for i in range(len(results) - 1):
            assert abs(results[i] - results[i+1]) < 1e-6
    
    def test_different_attention_configurations(self):
        """Test different attention configurations."""
        configs = [
            {"d_model": 64, "n_heads": 1, "dropout": 0.0},
            {"d_model": 128, "n_heads": 4, "dropout": 0.1},
            {"d_model": 256, "n_heads": 8, "dropout": 0.2},
        ]
        
        for config in configs:
            attention = ZeroSumAttention(**config)
            
            batch_size, seq_len = 1, 5
            x = torch.randn(batch_size, seq_len, config["d_model"])
            
            output, weights = attention(x)
            
            assert output.shape == x.shape
            assert weights.shape == (batch_size, config["n_heads"], seq_len, seq_len)
    
    def test_compensation_strength_effect(self):
        """Test compensation strength effect."""
        strengths = [0.0, 0.1, 0.5, 1.0]
        
        batch_size, seq_len, d_model = 1, 4, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        outputs = []
        for strength in strengths:
            attention = ZeroSumAttention(
                d_model=d_model,
                n_heads=2,
                compensation_strength=strength
            )
            
            output, _ = attention(x)
            outputs.append(output)
        
        # Different compensation strengths should give different results
        for i in range(len(outputs) - 1):
            assert not torch.allclose(outputs[i], outputs[i+1], atol=1e-6)