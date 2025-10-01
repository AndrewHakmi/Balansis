"""Tests for plotting utilities.

This module provides comprehensive tests for the plotting functionality
in balansis.utils.plot, covering all classes and methods.
"""

import pytest
import numpy as np
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Mock matplotlib and plotly to avoid import issues
with patch.dict('sys.modules', {
    'matplotlib': Mock(),
    'matplotlib.pyplot': Mock(),
    'matplotlib.animation': Mock(),
    'plotly': Mock(),
    'plotly.graph_objects': Mock(),
    'plotly.subplots': Mock(),
    'pandas': Mock()
}):
    from balansis.utils.plot import (
        PlotStyle, PlotBackend, PlotConfig, PlotUtils
    )
    from balansis.core.absolute import AbsoluteValue
    from balansis.core.eternity import EternalRatio
    from balansis.logic.compensator import CompensationRecord, CompensationType
    from pydantic import ValidationError
    
    # Create mock plotly objects
    go = Mock()
    go.Figure = Mock
    go.Scatter = Mock


class TestPlotStyle:
    """Test PlotStyle enum."""
    
    def test_plot_style_values(self):
        """Test PlotStyle enum values."""
        assert PlotStyle.SCIENTIFIC == "scientific"
        assert PlotStyle.ELEGANT == "elegant"
        assert PlotStyle.MINIMAL == "minimal"
        assert PlotStyle.COLORFUL == "colorful"
    
    def test_plot_style_membership(self):
        """Test PlotStyle membership."""
        assert "scientific" in PlotStyle
        assert "elegant" in PlotStyle
        assert "minimal" in PlotStyle
        assert "colorful" in PlotStyle
        assert "invalid" not in PlotStyle


class TestPlotBackend:
    """Test PlotBackend enum."""
    
    def test_plot_backend_values(self):
        """Test PlotBackend enum values."""
        assert PlotBackend.MATPLOTLIB == "matplotlib"
        assert PlotBackend.PLOTLY == "plotly"
    
    def test_plot_backend_membership(self):
        """Test PlotBackend membership."""
        assert "matplotlib" in PlotBackend
        assert "plotly" in PlotBackend
        assert "invalid" not in PlotBackend


class TestPlotConfig:
    """Test PlotConfig class."""
    
    def test_default_config(self):
        """Test default PlotConfig values."""
        config = PlotConfig()
        assert config.style == PlotStyle.SCIENTIFIC
        assert config.backend == PlotBackend.MATPLOTLIB
        assert config.width == 800
        assert config.height == 600
        assert config.dpi == 100
        assert len(config.color_palette) == 5
        assert config.font_size == 12
        assert config.line_width == 2.0
        assert config.marker_size == 6.0
        assert config.alpha == 0.8
        assert config.grid is True
        assert config.legend is True
        assert config.title_size == 14
        assert config.axis_label_size == 14
        assert config.interactive is False
        assert config.save_format == "png"
        assert config.animation_duration == 1000
        assert config.animation_frames == 50
    
    def test_custom_config(self):
        """Test custom PlotConfig values."""
        config = PlotConfig(
            style=PlotStyle.ELEGANT,
            backend=PlotBackend.PLOTLY,
            width=1200,
            height=800,
            dpi=150,
            font_size=14,
            alpha=0.9
        )
        assert config.style == PlotStyle.ELEGANT
        assert config.backend == PlotBackend.PLOTLY
        assert config.width == 1200
        assert config.height == 800
        assert config.dpi == 150
        assert config.font_size == 14
        assert config.alpha == 0.9
    
    def test_width_validation(self):
        """Test width validation."""
        with pytest.raises(ValidationError):
            PlotConfig(width=0)
        
        with pytest.raises(ValidationError):
            PlotConfig(width=-100)
    
    def test_height_validation(self):
        """Test height validation."""
        with pytest.raises(ValidationError):
            PlotConfig(height=0)
        
        with pytest.raises(ValidationError):
            PlotConfig(height=-100)
    
    def test_dpi_validation(self):
        """Test DPI validation."""
        with pytest.raises(ValidationError):
            PlotConfig(dpi=0)
        
        with pytest.raises(ValidationError):
            PlotConfig(dpi=-50)
    
    def test_alpha_validation(self):
        """Test alpha validation."""
        with pytest.raises(ValidationError):
            PlotConfig(alpha=-0.1)
        
        with pytest.raises(ValidationError):
            PlotConfig(alpha=1.1)
        
        # Valid alpha values
        config1 = PlotConfig(alpha=0.0)
        assert config1.alpha == 0.0
        
        config2 = PlotConfig(alpha=1.0)
        assert config2.alpha == 1.0
        
        config3 = PlotConfig(alpha=0.5)
        assert config3.alpha == 0.5
    
    def test_validate_assignment(self):
        """Test that validation occurs on assignment."""
        config = PlotConfig()
        
        with pytest.raises(ValidationError):
            config.width = -100
        
        with pytest.raises(ValidationError):
            config.alpha = 2.0


class TestPlotUtils:
    """Test PlotUtils class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PlotConfig()
        self.plotter = PlotUtils(self.config)
        
        # Create test data
        self.absolute_values = [
            AbsoluteValue(1.0, 1),
            AbsoluteValue(2.0, -1),
            AbsoluteValue(0.0, 0),
            AbsoluteValue(3.0, 1),
            AbsoluteValue(1.5, -1)
        ]
        
        self.eternal_ratios = [
            EternalRatio(AbsoluteValue(2.0, 1), AbsoluteValue(1.0, 1)),
            EternalRatio(AbsoluteValue(3.0, 1), AbsoluteValue(2.0, 1)),
            EternalRatio(AbsoluteValue(1.0, -1), AbsoluteValue(2.0, 1))
        ]
        
        self.compensation_records = [
            CompensationRecord(
                compensation_type=CompensationType.STABILITY,
                original_value=AbsoluteValue(1e-15, 1),
                compensated_value=AbsoluteValue(1e-12, 1),
                compensation_factor=1000.0,
                stability_gain=0.95
            ),
            CompensationRecord(
                compensation_type=CompensationType.OVERFLOW,
                original_value=AbsoluteValue(1e200, 1),
                compensated_value=AbsoluteValue(1e50, 1),
                compensation_factor=0.001,
                stability_gain=0.8
            )
        ]
    
    def test_init_default(self):
        """Test PlotUtils initialization with defaults."""
        plotter = PlotUtils()
        assert plotter.config is not None
        assert plotter.operations is not None
        assert plotter.compensator is not None
    
    def test_init_custom_config(self):
        """Test PlotUtils initialization with custom config."""
        config = PlotConfig(style=PlotStyle.ELEGANT)
        plotter = PlotUtils(config)
        assert plotter.config.style == PlotStyle.ELEGANT
    
    @patch('matplotlib.pyplot.style.use')
    @patch('matplotlib.pyplot.rcParams')
    def test_setup_matplotlib_style(self, mock_rcparams, mock_style_use):
        """Test matplotlib style setup."""
        mock_rcparams.update = Mock()
        
        # Test different styles
        for style in PlotStyle:
            config = PlotConfig(style=style, backend=PlotBackend.MATPLOTLIB)
            plotter = PlotUtils(config)
            
            # Verify rcParams.update was called
            mock_rcparams.update.assert_called()
            
            # Verify style.use was called
            mock_style_use.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_absolute_values_matplotlib(self, mock_subplots):
        """Test plotting AbsoluteValues with matplotlib."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.scatter.return_value = Mock()
        
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
        plotter = PlotUtils(config)
        
        result = plotter.plot_absolute_values(self.absolute_values)
        
        # Verify matplotlib methods were called
        mock_subplots.assert_called_once()
        mock_ax.scatter.assert_called_once()
        mock_ax.set_title.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        
        assert result == mock_fig
    
    def test_plot_absolute_values_plotly(self):
        """Test plotting AbsoluteValues with plotly."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plotter = PlotUtils(config)
        
        result = plotter.plot_absolute_values(self.absolute_values)
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_eternal_ratios_matplotlib(self, mock_subplots):
        """Test plotting EternalRatios with matplotlib."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
        plotter = PlotUtils(config)
        
        result = plotter.plot_eternal_ratios(self.eternal_ratios)
        
        # Verify matplotlib methods were called
        mock_subplots.assert_called_once()
        mock_ax1.scatter.assert_called_once()
        mock_ax2.scatter.assert_called_once()
        
        assert result == mock_fig
    
    def test_plot_eternal_ratios_plotly(self):
        """Test plotting EternalRatios with plotly."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plotter = PlotUtils(config)
        
        result = plotter.plot_eternal_ratios(self.eternal_ratios)
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    def test_plot_compensation_analysis_empty_records(self):
        """Test compensation analysis with empty records."""
        with pytest.raises(ValueError, match="No compensation records provided"):
            self.plotter.plot_compensation_analysis([])
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_compensation_analysis_matplotlib(self, mock_subplots):
        """Test compensation analysis with matplotlib."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
        plotter = PlotUtils(config)
        
        result = plotter.plot_compensation_analysis(self.compensation_records)
        
        # Verify matplotlib methods were called
        mock_subplots.assert_called_once()
        
        assert result == mock_fig
    
    def test_plot_compensation_analysis_plotly(self):
        """Test compensation analysis with plotly."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plotter = PlotUtils(config)
        
        result = plotter.plot_compensation_analysis(self.compensation_records)
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_act_phase_space_matplotlib(self, mock_subplots):
        """Test ACT phase space plotting with matplotlib."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.scatter.return_value = Mock()
        
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
        plotter = PlotUtils(config)
        
        result = plotter.plot_act_phase_space(self.absolute_values)
        
        # Verify matplotlib methods were called
        mock_subplots.assert_called_once()
        mock_ax.scatter.assert_called_once()
        
        assert result == mock_fig
    
    def test_plot_act_phase_space_plotly(self):
        """Test ACT phase space plotting with plotly."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plotter = PlotUtils(config)
        
        result = plotter.plot_act_phase_space(self.absolute_values)
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    def test_create_interactive_dashboard_wrong_backend(self):
        """Test interactive dashboard with wrong backend."""
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
        plotter = PlotUtils(config)
        
        with pytest.raises(ValueError, match="Interactive dashboard requires Plotly backend"):
            plotter.create_interactive_dashboard(
                self.absolute_values, self.eternal_ratios, self.compensation_records
            )
    
    def test_create_interactive_dashboard_plotly(self):
        """Test interactive dashboard with plotly."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plotter = PlotUtils(config)
        
        result = plotter.create_interactive_dashboard(
            self.absolute_values, self.eternal_ratios, self.compensation_records
        )
        
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0
    
    @patch('matplotlib.animation.FuncAnimation')
    @patch('matplotlib.pyplot.subplots')
    def test_animate_sequence_evolution_matplotlib(self, mock_subplots, mock_animation):
        """Test sequence animation with matplotlib."""
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.scatter.return_value = Mock()
        mock_ax.plot.return_value = (Mock(),)
        mock_ax.text.return_value = Mock()
        
        mock_anim = Mock()
        mock_animation.return_value = mock_anim
        
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
        plotter = PlotUtils(config)
        
        sequences = [self.absolute_values[:3], self.absolute_values[1:4]]
        result = plotter.animate_sequence_evolution(sequences)
        
        # Verify animation was created
        mock_animation.assert_called_once()
        assert result == mock_anim
    
    def test_animate_sequence_evolution_plotly(self):
        """Test sequence animation with plotly."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plotter = PlotUtils(config)
        
        sequences = [self.absolute_values[:3], self.absolute_values[1:4]]
        result = plotter.animate_sequence_evolution(sequences)
        
        assert isinstance(result, go.Figure)
        assert len(result.frames) > 0
    
    def test_export_plot_data_no_data(self):
        """Test export with no data provided."""
        with pytest.raises(ValueError, match="No data provided for export"):
            self.plotter.export_plot_data()
    
    @patch('pandas.DataFrame.to_csv')
    def test_export_plot_data_csv(self, mock_to_csv):
        """Test export to CSV format."""
        filename = self.plotter.export_plot_data(
            absolute_values=self.absolute_values,
            format='csv',
            filename='test.csv'
        )
        
        assert filename == 'test.csv'
        mock_to_csv.assert_called_once_with('test.csv', index=False)
    
    @patch('pandas.DataFrame.to_json')
    def test_export_plot_data_json(self, mock_to_json):
        """Test export to JSON format."""
        filename = self.plotter.export_plot_data(
            eternal_ratios=self.eternal_ratios,
            format='json',
            filename='test.json'
        )
        
        assert filename == 'test.json'
        mock_to_json.assert_called_once_with('test.json', orient="records", indent=2)
    
    @patch('pandas.DataFrame.to_excel')
    def test_export_plot_data_excel(self, mock_to_excel):
        """Test export to Excel format."""
        filename = self.plotter.export_plot_data(
            compensation_records=self.compensation_records,
            format='xlsx',
            filename='test.xlsx'
        )
        
        assert filename == 'test.xlsx'
        mock_to_excel.assert_called_once_with('test.xlsx', index=False)
    
    def test_export_plot_data_unsupported_format(self):
        """Test export with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported export format: xml"):
            self.plotter.export_plot_data(
                absolute_values=self.absolute_values,
                format='xml'
            )
    
    def test_repr(self):
        """Test string representation."""
        config = PlotConfig(backend=PlotBackend.PLOTLY, style=PlotStyle.ELEGANT)
        plotter = PlotUtils(config)
        
        repr_str = repr(plotter)
        assert "PlotUtils" in repr_str
        assert "plotly" in repr_str
        assert "elegant" in repr_str
    
    def test_str(self):
        """Test human-readable string representation."""
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB, style=PlotStyle.SCIENTIFIC)
        plotter = PlotUtils(config)
        
        str_repr = str(plotter)
        assert "PlotUtils" in str_repr
        assert "matplotlib" in str_repr
        assert "scientific" in str_repr


class TestPlotUtilsEdgeCases:
    """Test edge cases and error conditions for PlotUtils."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = PlotUtils()
    
    def test_plot_absolute_values_empty_list(self):
        """Test plotting empty list of AbsoluteValues."""
        # Should not raise an error, but create empty plot
        result = self.plotter.plot_absolute_values([])
        assert result is not None
    
    def test_plot_eternal_ratios_empty_list(self):
        """Test plotting empty list of EternalRatios."""
        # Should not raise an error, but create empty plot
        result = self.plotter.plot_eternal_ratios([])
        assert result is not None
    
    def test_plot_with_save_path(self):
        """Test plotting with save path."""
        values = [AbsoluteValue(1.0, 1), AbsoluteValue(2.0, -1)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot.png")
            
            # This should not raise an error
            result = self.plotter.plot_absolute_values(values, save_path=save_path)
            assert result is not None
    
    def test_animate_empty_sequences(self):
        """Test animation with empty sequences."""
        # Should handle empty sequences gracefully
        result = self.plotter.animate_sequence_evolution([])
        assert result is not None
    
    def test_dashboard_with_empty_records(self):
        """Test dashboard creation with empty compensation records."""
        config = PlotConfig(backend=PlotBackend.PLOTLY)
        plotter = PlotUtils(config)
        
        values = [AbsoluteValue(1.0, 1)]
        ratios = [EternalRatio(AbsoluteValue(2.0, 1), AbsoluteValue(1.0, 1))]
        
        # Should handle empty records gracefully
        result = plotter.create_interactive_dashboard(values, ratios, [])
        assert isinstance(result, go.Figure)