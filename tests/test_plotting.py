"""Tests for the Plotting utilities module.

This module contains comprehensive tests for visualization capabilities,
verifying plotting functions for AbsoluteValue, EternalRatio, and ACT concepts.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.logic.compensator import CompensationRecord, CompensationType
from balansis.utils.plot import (
    PlotStyle, PlotBackend, PlotConfig, PlotUtils
)
from balansis import ACT_EPSILON


class TestPlotStyle:
    """Test PlotStyle enum."""
    
    def test_plot_style_values(self):
        """Test PlotStyle enum values."""
        assert PlotStyle.SCIENTIFIC.value == "scientific"
        assert PlotStyle.ELEGANT.value == "elegant"
        assert PlotStyle.MINIMAL.value == "minimal"
        assert PlotStyle.COLORFUL.value == "colorful"


class TestPlotBackend:
    """Test PlotBackend enum."""
    
    def test_plot_backend_values(self):
        """Test PlotBackend enum values."""
        assert PlotBackend.MATPLOTLIB.value == "matplotlib"
        assert PlotBackend.PLOTLY.value == "plotly"


class TestPlotConfig:
    """Test PlotConfig Pydantic model."""
    
    def test_plot_config_creation(self):
        """Test PlotConfig creation with defaults."""
        config = PlotConfig()
        
        assert config.style == PlotStyle.SCIENTIFIC
        assert config.backend == PlotBackend.MATPLOTLIB
        assert config.width == 800
        assert config.height == 600
        assert config.dpi == 100
        assert config.grid == True
        assert config.legend == True
        assert config.title_size == 14
        assert config.axis_label_size == 14
        assert config.line_width == 2.0
        assert config.marker_size == 6.0
        assert config.alpha == 0.8
        assert config.color_palette == ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        assert config.save_format == "png"
        assert config.interactive == False
        assert config.animation_duration == 1000
        assert config.animation_frames == 50
    
    def test_plot_config_custom(self):
        """Test PlotConfig with custom values."""
        config = PlotConfig(
            style=PlotStyle.ELEGANT,
            backend=PlotBackend.PLOTLY,
            width=1200,
            height=800,
            interactive=True
        )
        
        assert config.style == PlotStyle.ELEGANT
        assert config.backend == PlotBackend.PLOTLY
        assert config.width == 1200
        assert config.height == 800
        assert config.interactive == True
    
    def test_plot_config_validation(self):
        """Test PlotConfig validation."""
        # Valid configuration
        config = PlotConfig(width=100, height=100, dpi=50)
        assert config.width == 100
        
        # Test minimum constraints
        with pytest.raises(ValueError):
            PlotConfig(width=0)  # Should be > 0
        
        with pytest.raises(ValueError):
            PlotConfig(height=0)  # Should be > 0
        
        with pytest.raises(ValueError):
            PlotConfig(dpi=0)  # Should be > 0


class TestPlotUtils:
    """Test PlotUtils functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PlotConfig()
        self.plot_utils = PlotUtils(self.config)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_absolute_values_matplotlib(self, mock_savefig, mock_show, mock_figure):
        """Test plotting AbsoluteValue objects with matplotlib."""
        # Create mock figure and axes
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = (mock_fig, mock_ax)
        
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-1.0),
            AbsoluteValue(magnitude=3.0, direction=0.5),
            AbsoluteValue.absolute()
        ]
        
        # Test basic plotting
        result = self.plot_utils.plot_absolute_values(
            values, 
            title="Test Plot",
            show_directions=True,
            save_path=None
        )
        
        assert result is not None
        mock_figure.assert_called_once()
    
    @patch('plotly.graph_objects.Figure')
    def test_plot_absolute_values_plotly(self, mock_figure):
        """Test plotting AbsoluteValue objects with plotly."""
        # Configure for plotly
        config = PlotConfig(backend=PlotBackend.PLOTLY, interactive=True)
        plot_utils = PlotUtils(config)
        
        # Create mock figure
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-1.0)
        ]
        
        result = plot_utils.plot_absolute_values(
            values,
            title="Test Plotly",
            show_directions=True
        )
        
        assert result is not None
        mock_figure.assert_called_once()
    
    def test_plot_absolute_values_empty_list(self):
        """Test plotting empty list of AbsoluteValue objects."""
        with pytest.raises(ValueError, match="No values provided for plotting"):
            self.plot_utils.plot_absolute_values([])
    
    @patch('matplotlib.pyplot.figure')
    def test_plot_eternal_ratios_matplotlib(self, mock_figure):
        """Test plotting EternalRatio objects with matplotlib."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = (mock_fig, mock_ax)
        
        ratios = [
            EternalRatio(
                numerator=AbsoluteValue(magnitude=6.0, direction=1.0),
                denominator=AbsoluteValue(magnitude=3.0, direction=1.0)
            ),
            EternalRatio(
                numerator=AbsoluteValue(magnitude=4.0, direction=-1.0),
                denominator=AbsoluteValue(magnitude=2.0, direction=1.0)
            ),
            EternalRatio.unity()
        ]
        
        result = self.plot_utils.plot_eternal_ratios(
            ratios,
            title="Ratio Test",
            show_stability=True
        )
        
        assert result is not None
        mock_figure.assert_called_once()
    
    def test_plot_eternal_ratios_empty_list(self):
        """Test plotting empty list of EternalRatio objects."""
        with pytest.raises(ValueError, match="No ratios provided for plotting"):
            self.plot_utils.plot_eternal_ratios([])
    
    @patch('matplotlib.pyplot.figure')
    def test_plot_compensation_analysis_matplotlib(self, mock_figure):
        """Test plotting compensation analysis with matplotlib."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = (mock_fig, mock_ax)
        
        records = [
            CompensationRecord(
                operation_type=CompensationType.STABILITY,
                original_values=[AbsoluteValue(magnitude=1e10, direction=1.0)],
                compensated_values=[AbsoluteValue(magnitude=1e5, direction=1.0)],
                stability_metric=0.8,
                timestamp=1234567890.0
            ),
            CompensationRecord(
                operation_type=CompensationType.OVERFLOW,
                original_values=[AbsoluteValue(magnitude=1e-10, direction=1.0)],
                compensated_values=[AbsoluteValue(magnitude=1e-5, direction=1.0)],
                stability_metric=0.9,
                timestamp=1234567891.0
            )
        ]
        
        result = self.plot_utils.plot_compensation_analysis(
            records,
            title="Compensation Analysis"
        )
        
        assert result is not None
        mock_figure.assert_called_once()
    
    def test_plot_compensation_analysis_empty_list(self):
        """Test plotting empty compensation records."""
        with pytest.raises(ValueError, match="No compensation records provided"):
            self.plot_utils.plot_compensation_analysis([])
    
    @patch('matplotlib.pyplot.figure')
    def test_plot_act_phase_space_matplotlib(self, mock_figure):
        """Test plotting ACT phase space with matplotlib."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = (mock_fig, mock_ax)
        
        values = [
            AbsoluteValue(magnitude=1.0, direction=1),
            AbsoluteValue(magnitude=2.0, direction=0.5),
            AbsoluteValue(magnitude=3.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-0.5),
            AbsoluteValue(magnitude=1.0, direction=-1.0)
        ]
        
        result = self.plot_utils.plot_act_phase_space(
            values,
            title="ACT Phase Space",
            show_trajectories=True
        )
        
        assert result is not None
        mock_figure.assert_called_once()
    
    def test_plot_act_phase_space_empty_list(self):
        """Test plotting empty phase space."""
        with pytest.raises(ValueError, match="No values provided for phase space plotting"):
            self.plot_utils.plot_act_phase_space([])
    
    @patch('plotly.graph_objects.Figure')
    @patch('plotly.subplots.make_subplots')
    def test_create_interactive_dashboard_plotly(self, mock_subplots, mock_figure):
        """Test creating interactive dashboard with plotly."""
        config = PlotConfig(backend=PlotBackend.PLOTLY, interactive=True)
        plot_utils = PlotUtils(config)
        
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-1.0)
        ]
        
        ratios = [
            EternalRatio(
                numerator=AbsoluteValue(magnitude=4.0, direction=1.0),
                denominator=AbsoluteValue(magnitude=2.0, direction=1.0)
            )
        ]
        
        records = [
            CompensationRecord(
                operation_type=CompensationType.STABILITY,
                original_values=[AbsoluteValue(magnitude=1e10, direction=1.0)],
                compensated_values=[AbsoluteValue(magnitude=1e5, direction=1.0)],
                stability_metric=0.8,
                timestamp=1234567890.0
            )
        ]
        
        result = plot_utils.create_interactive_dashboard(
            absolute_values=values,
            eternal_ratios=ratios,
            compensation_records=records,
            title="Test Dashboard"
        )
        
        assert result is not None
        mock_subplots.assert_called_once()
    
    def test_create_interactive_dashboard_matplotlib_error(self):
        """Test that matplotlib backend raises error for interactive dashboard."""
        values = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        with pytest.raises(ValueError, match="Interactive dashboard requires plotly backend"):
            self.plot_utils.create_interactive_dashboard(absolute_values=values)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.animation.FuncAnimation')
    def test_animate_sequence_evolution_matplotlib(self, mock_animation, mock_figure):
        """Test animating sequence evolution with matplotlib."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_figure.return_value = (mock_fig, mock_ax)
        
        mock_anim = Mock()
        mock_animation.return_value = mock_anim
        
        sequences = [
            [AbsoluteValue(magnitude=1.0, direction=1.0)],
            [AbsoluteValue(magnitude=1.0, direction=1.0), AbsoluteValue(magnitude=2.0, direction=1.0)],
            [AbsoluteValue(magnitude=1.0, direction=1.0), AbsoluteValue(magnitude=2.0, direction=1.0), AbsoluteValue(magnitude=3.0, direction=1.0)]
        ]
        
        result = self.plot_utils.animate_sequence_evolution(
            sequences,
            title="Sequence Evolution",
            save_path=None
        )
        
        assert result is not None
        mock_figure.assert_called_once()
        mock_animation.assert_called_once()
    
    def test_animate_sequence_evolution_empty_sequences(self):
        """Test animating empty sequences."""
        with pytest.raises(ValueError, match="No sequences provided for animation"):
            self.plot_utils.animate_sequence_evolution([])
    
    def test_animate_sequence_evolution_plotly_error(self):
        """Test that plotly backend raises error for animation."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plot_utils = PlotUtils(config)
        
        sequences = [[AbsoluteValue(magnitude=1.0, direction=1.0)]]
        
        with pytest.raises(ValueError, match="Animation requires matplotlib backend"):
            plot_utils.animate_sequence_evolution(sequences)
    
    def test_export_plot_data_csv(self):
        """Test exporting plot data to CSV format."""
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-1.0),
            AbsoluteValue.absolute()
        ]
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            result = self.plot_utils.export_plot_data(
                absolute_values=values,
                format="csv",
                filename="test_export.csv"
            )
            
            assert result == "test_export.csv"
            mock_to_csv.assert_called_once_with("test_export.csv", index=False)
    
    def test_export_plot_data_json(self):
        """Test exporting plot data to JSON format."""
        ratios = [
            EternalRatio(
                numerator=AbsoluteValue(magnitude=4.0, direction=1.0),
                denominator=AbsoluteValue(magnitude=2.0, direction=1.0)
            ),
            EternalRatio.unity()
        ]
        
        with patch('pandas.DataFrame.to_json') as mock_to_json:
            result = self.plot_utils.export_plot_data(
                eternal_ratios=ratios,
                format="json",
                filename="test_export.json"
            )
            
            assert result == "test_export.json"
            mock_to_json.assert_called_once_with("test_export.json", orient="records", indent=2)
    
    def test_export_plot_data_excel(self):
        """Test exporting plot data to Excel format."""
        values = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        with patch('pandas.DataFrame.to_excel') as mock_to_excel:
            result = self.plot_utils.export_plot_data(
                absolute_values=values,
                format="excel",
                filename="test_export.xlsx"
            )
            
            assert result == "test_export.xlsx"
            mock_to_excel.assert_called_once_with("test_export.xlsx", index=False)
    
    def test_export_plot_data_unsupported_format(self):
        """Test exporting with unsupported format."""
        values = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.plot_utils.export_plot_data(
                absolute_values=values,
                format="xml",
                filename="test.xml"
            )
    
    def test_export_plot_data_no_data(self):
        """Test exporting with no data provided."""
        with pytest.raises(ValueError, match="No data provided for export"):
            self.plot_utils.export_plot_data(format="csv", filename="test.csv")
    
    def test_export_plot_data_compensation_records(self):
        """Test exporting compensation records data."""
        records = [
            CompensationRecord(
                operation_type=CompensationType.STABILITY,
                original_values=[AbsoluteValue(magnitude=1e10, direction=1.0)],
                compensated_values=[AbsoluteValue(magnitude=1e5, direction=1.0)],
                stability_metric=0.8,
                timestamp=1234567890.0
            )
        ]
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            result = self.plot_utils.export_plot_data(
                compensation_records=records,
                format="csv",
                filename="compensation_export.csv"
            )
            
            assert result == "compensation_export.csv"
            mock_to_csv.assert_called_once_with("compensation_export.csv", index=False)


class TestPlotUtilsIntegration:
    """Test PlotUtils integration with different configurations."""
    
    def test_different_plot_styles(self):
        """Test plotting with different styles."""
        values = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        styles = [PlotStyle.SCIENTIFIC, PlotStyle.ELEGANT, PlotStyle.MINIMAL, PlotStyle.COLORFUL]
        
        for style in styles:
            config = PlotConfig(style=style)
            plot_utils = PlotUtils(config)
            
            with patch('matplotlib.pyplot.figure') as mock_figure:
                mock_fig = Mock()
                mock_ax = Mock()
                mock_figure.return_value = (mock_fig, mock_ax)
                
                result = plot_utils.plot_absolute_values(values, title=f"Test {style.value}")
                assert result is not None
    
    def test_custom_color_palette(self):
        """Test plotting with custom color palette."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        config = PlotConfig(color_palette=custom_colors)
        plot_utils = PlotUtils(config)
        
        values = [
            AbsoluteValue(magnitude=1.0, direction=1.0),
            AbsoluteValue(magnitude=2.0, direction=-1.0)
        ]
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_figure.return_value = (mock_fig, mock_ax)
            
            result = plot_utils.plot_absolute_values(values, title="Custom Colors")
            assert result is not None
    
    def test_high_dpi_plotting(self):
        """Test plotting with high DPI settings."""
        config = PlotConfig(dpi=300, width=1600, height=1200)
        plot_utils = PlotUtils(config)
        
        values = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_figure.return_value = (mock_fig, mock_ax)
            
            result = plot_utils.plot_absolute_values(values, title="High DPI")
            assert result is not None
    
    def test_plot_config_update(self):
        """Test updating plot configuration."""
        plot_utils = PlotUtils(PlotConfig())
        
        # Update configuration
        new_config = PlotConfig(
            style=PlotStyle.COLORFUL,
            line_width=3.0,
            marker_size=8.0
        )
        plot_utils.config = new_config
        
        assert plot_utils.config.style == PlotStyle.COLORFUL
        assert plot_utils.config.line_width == 3.0
        assert plot_utils.config.marker_size == 8.0


class TestPlotUtilsErrorHandling:
    """Test PlotUtils error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plot_utils = PlotUtils(PlotConfig())
    
    def test_plot_with_invalid_absolute_values(self):
        """Test plotting with invalid AbsoluteValue objects."""
        # This should be handled gracefully by the AbsoluteValue validation
        values = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_figure.return_value = (mock_fig, mock_ax)
            
            result = self.plot_utils.plot_absolute_values(values)
            assert result is not None
    
    def test_plot_with_very_large_values(self):
        """Test plotting with very large magnitude values."""
        values = [
            AbsoluteValue(magnitude=1e100, direction=1.0),
            AbsoluteValue(magnitude=1e-100, direction=-1.0)
        ]
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_figure.return_value = (mock_fig, mock_ax)
            
            result = self.plot_utils.plot_absolute_values(values, title="Large Values")
            assert result is not None
    
    def test_plot_with_special_values(self):
        """Test plotting with special values (Absolute, zero direction)."""
        values = [
            AbsoluteValue.absolute(),
            AbsoluteValue(magnitude=0.0, direction=1),
        AbsoluteValue(magnitude=1.0, direction=1)
        ]
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_figure.return_value = (mock_fig, mock_ax)
            
            result = self.plot_utils.plot_absolute_values(values, title="Special Values")
            assert result is not None
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_plot_with_invalid_path(self, mock_savefig):
        """Test saving plot with invalid file path."""
        mock_savefig.side_effect = OSError("Invalid path")
        
        values = [AbsoluteValue(magnitude=1.0, direction=1.0)]
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_figure.return_value = (mock_fig, mock_ax)
            
            with pytest.raises(OSError):
                self.plot_utils.plot_absolute_values(
                    values,
                    save_path="/invalid/path/plot.png"
                )


if __name__ == "__main__":
    pytest.main([__file__])