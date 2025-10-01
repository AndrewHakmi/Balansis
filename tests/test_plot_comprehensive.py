"""Comprehensive tests for plotting utilities to achieve 95% coverage.

This module provides extensive tests for all plotting functionality
in balansis.utils.plot, covering edge cases and error handling.
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


class TestPlotUtilsComprehensive:
    """Comprehensive tests for PlotUtils functionality."""
    
    @pytest.fixture
    def mock_plotter(self):
        """Create a PlotUtils instance with mocked dependencies."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True):
            return PlotUtils()
    
    @pytest.fixture
    def sample_absolute_values(self):
        """Create sample AbsoluteValue objects."""
        return [
            AbsoluteValue(magnitude=1.0, direction=1),
            AbsoluteValue(magnitude=2.0, direction=-1),
            AbsoluteValue(magnitude=0.0, direction=1),  # Absolute value
            AbsoluteValue(magnitude=3.0, direction=1)
        ]
    
    @pytest.fixture
    def sample_eternal_ratios(self):
        """Create sample EternalRatio objects."""
        return [
            EternalRatio(numerator=AbsoluteValue(magnitude=1.0, direction=1), denominator=AbsoluteValue(magnitude=2.0, direction=1)),
            EternalRatio(numerator=AbsoluteValue(magnitude=3.0, direction=-1), denominator=AbsoluteValue(magnitude=1.0, direction=1)),
            EternalRatio(numerator=AbsoluteValue(magnitude=0.0, direction=1), denominator=AbsoluteValue(magnitude=1.0, direction=1))
        ]
    
    @pytest.fixture
    def sample_compensation_records(self):
        """Create sample CompensationRecord objects."""
        return [
            CompensationRecord(
                operation_type="addition",
                compensation_type=CompensationType.STABILITY,
                original_values=[AbsoluteValue(magnitude=1.0, direction=1)],
                compensated_values=[AbsoluteValue(magnitude=1.1, direction=1)],
                compensation_factor=1.1,
                stability_metric=0.8,
                timestamp=1.0
            ),
            CompensationRecord(
                operation_type="multiplication",
                compensation_type=CompensationType.OVERFLOW,
                original_values=[AbsoluteValue(magnitude=1e100, direction=1)],
                compensated_values=[AbsoluteValue(magnitude=1e50, direction=1)],
                compensation_factor=0.5,
                stability_metric=0.9,
                timestamp=2.0
            )
        ]
    
    def test_plot_eternal_ratios_matplotlib(self, mock_plotter, sample_eternal_ratios):
        """Test plotting EternalRatios with matplotlib backend."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.savefig'):
            
            # Mock figure and axes
            mock_fig = Mock()
            mock_ax1, mock_ax2 = Mock(), Mock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
            plotter = PlotUtils(config=config)
            
            result = plotter.plot_eternal_ratios(sample_eternal_ratios)
            
            assert result == mock_fig
            mock_subplots.assert_called_once()
            mock_ax1.scatter.assert_called()
            mock_ax2.scatter.assert_called()
    
    def test_plot_eternal_ratios_plotly(self, mock_plotter, sample_eternal_ratios):
        """Test plotting EternalRatios with plotly backend."""
        with patch('plotly.subplots.make_subplots') as mock_subplots, \
             patch('plotly.graph_objects.Scatter') as mock_scatter:
            
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            result = plotter.plot_eternal_ratios(sample_eternal_ratios)
            
            assert result == mock_fig
            mock_subplots.assert_called_once()
    
    def test_plot_compensation_analysis_empty_records(self, mock_plotter):
        """Test compensation analysis with empty records list."""
        with pytest.raises(ValueError, match="No compensation records provided"):
            mock_plotter.plot_compensation_analysis([])
    
    def test_plot_compensation_analysis_matplotlib(self, mock_plotter, sample_compensation_records):
        """Test compensation analysis with matplotlib backend."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.suptitle'):
            
            mock_fig = Mock()
            mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
            plotter = PlotUtils(config=config)
            
            result = plotter.plot_compensation_analysis(sample_compensation_records)
            
            assert result == mock_fig
            mock_subplots.assert_called_once()
    
    def test_plot_compensation_analysis_plotly(self, mock_plotter, sample_compensation_records):
        """Test compensation analysis with plotly backend."""
        with patch('plotly.subplots.make_subplots') as mock_subplots, \
             patch('plotly.graph_objects.Bar') as mock_bar, \
             patch('plotly.graph_objects.Scatter') as mock_scatter, \
             patch('plotly.graph_objects.Histogram') as mock_hist:
            
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            result = plotter.plot_compensation_analysis(sample_compensation_records)
            
            assert result == mock_fig
    
    def test_plot_act_phase_space_matplotlib(self, mock_plotter, sample_absolute_values):
        """Test ACT phase space plotting with matplotlib."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.colorbar') as mock_colorbar, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
            plotter = PlotUtils(config=config)
            
            result = plotter.plot_act_phase_space(sample_absolute_values)
            
            assert result == mock_fig
            mock_ax.scatter.assert_called()
            mock_ax.axvline.assert_called()
            mock_ax.axhline.assert_called()
    
    def test_plot_act_phase_space_plotly(self, mock_plotter, sample_absolute_values):
        """Test ACT phase space plotting with plotly."""
        with patch('plotly.graph_objects.Figure') as mock_figure, \
             patch('plotly.graph_objects.Scatter') as mock_scatter:
            
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            result = plotter.plot_act_phase_space(sample_absolute_values)
            
            assert result == mock_fig
    
    def test_create_interactive_dashboard_wrong_backend(self, mock_plotter, sample_absolute_values, sample_eternal_ratios, sample_compensation_records):
        """Test interactive dashboard with wrong backend."""
        config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
        plotter = PlotUtils(config=config)
        
        with pytest.raises(ValueError, match="Interactive dashboard requires Plotly backend"):
            plotter.create_interactive_dashboard(
                sample_absolute_values, sample_eternal_ratios, sample_compensation_records
            )
    
    def test_create_interactive_dashboard_plotly(self, mock_plotter, sample_absolute_values, sample_eternal_ratios, sample_compensation_records):
        """Test interactive dashboard with plotly backend."""
        with patch('plotly.subplots.make_subplots') as mock_subplots, \
             patch('plotly.graph_objects.Scatter') as mock_scatter, \
             patch('plotly.graph_objects.Bar') as mock_bar:
            
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            result = plotter.create_interactive_dashboard(
                sample_absolute_values, sample_eternal_ratios, sample_compensation_records
            )
            
            assert result == mock_fig
    
    def test_animate_sequence_evolution_matplotlib(self, mock_plotter):
        """Test sequence animation with matplotlib."""
        sequences = [
            [AbsoluteValue(magnitude=1.0, direction=1), AbsoluteValue(magnitude=2.0, direction=-1)],
            [AbsoluteValue(magnitude=1.5, direction=1), AbsoluteValue(magnitude=2.5, direction=-1)]
        ]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.animation.FuncAnimation') as mock_anim:
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_animation = Mock()
            mock_anim.return_value = mock_animation
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
            plotter = PlotUtils(config=config)
            
            result = plotter.animate_sequence_evolution(sequences)
            
            assert result == mock_animation
    
    def test_animate_sequence_evolution_plotly(self, mock_plotter):
        """Test sequence animation with plotly."""
        sequences = [
            [AbsoluteValue(magnitude=1.0, direction=1), AbsoluteValue(magnitude=2.0, direction=-1)],
            [AbsoluteValue(magnitude=1.5, direction=1), AbsoluteValue(magnitude=2.5, direction=-1)]
        ]
        
        with patch('plotly.graph_objects.Figure') as mock_figure, \
             patch('plotly.graph_objects.Frame') as mock_frame, \
             patch('plotly.graph_objects.Scatter') as mock_scatter:
            
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            result = plotter.animate_sequence_evolution(sequences)
            
            assert result == mock_fig
    
    def test_animate_empty_sequences(self, mock_plotter):
        """Test animation with empty sequences."""
        sequences = [[], []]
        
        with patch('plotly.graph_objects.Figure') as mock_figure:
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            result = plotter.animate_sequence_evolution(sequences)
            assert result == mock_fig
    
    def test_export_plot_data_no_data(self, mock_plotter):
        """Test export with no data provided."""
        with pytest.raises(ValueError, match="No data provided for export"):
            mock_plotter.export_plot_data()
    
    def test_export_plot_data_csv(self, mock_plotter, sample_absolute_values):
        """Test exporting data to CSV format."""
        with patch('pandas.DataFrame') as mock_df:
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            
            result = mock_plotter.export_plot_data(
                absolute_values=sample_absolute_values,
                format='csv',
                filename='test.csv'
            )
            
            assert result == 'test.csv'
            mock_df_instance.to_csv.assert_called_once_with('test.csv', index=False)
    
    def test_export_plot_data_json(self, mock_plotter, sample_eternal_ratios):
        """Test exporting data to JSON format."""
        with patch('pandas.DataFrame') as mock_df:
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            
            result = mock_plotter.export_plot_data(
                eternal_ratios=sample_eternal_ratios,
                format='json',
                filename='test.json'
            )
            
            assert result == 'test.json'
            mock_df_instance.to_json.assert_called_once_with('test.json', orient="records", indent=2)
    
    def test_export_plot_data_excel(self, mock_plotter, sample_compensation_records):
        """Test exporting data to Excel format."""
        with patch('pandas.DataFrame') as mock_df:
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            
            result = mock_plotter.export_plot_data(
                compensation_records=sample_compensation_records,
                format='xlsx',
                filename='test.xlsx'
            )
            
            assert result == 'test.xlsx'
            mock_df_instance.to_excel.assert_called_once_with('test.xlsx', index=False)
    
    def test_export_plot_data_unsupported_format(self, mock_plotter, sample_absolute_values):
        """Test export with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported export format: xml"):
            mock_plotter.export_plot_data(
                absolute_values=sample_absolute_values,
                format='xml'
            )
    
    def test_export_plot_data_all_types(self, mock_plotter, sample_absolute_values, sample_eternal_ratios, sample_compensation_records):
        """Test exporting all data types together."""
        with patch('pandas.DataFrame') as mock_df:
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            
            result = mock_plotter.export_plot_data(
                absolute_values=sample_absolute_values,
                eternal_ratios=sample_eternal_ratios,
                compensation_records=sample_compensation_records,
                format='csv',
                filename='all_data.csv'
            )
            
            assert result == 'all_data.csv'
            mock_df_instance.to_csv.assert_called_once_with('all_data.csv', index=False)
    
    def test_plot_utils_repr(self, mock_plotter):
        """Test PlotUtils string representations."""
        repr_str = repr(mock_plotter)
        str_str = str(mock_plotter)
        
        assert "PlotUtils" in repr_str
        assert "backend" in repr_str
        assert "style" in repr_str
        assert "PlotUtils" in str_str
        assert "backend" in str_str
        assert "style" in str_str
    
    def test_setup_matplotlib_style_fallback(self, mock_plotter):
        """Test matplotlib style setup with fallback."""
        with patch('matplotlib.pyplot.style.use') as mock_style_use:
            # Simulate style not available
            mock_style_use.side_effect = [OSError("Style not found"), None]
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB, style=PlotStyle.SCIENTIFIC)
            plotter = PlotUtils(config=config)
            
            # The _setup_matplotlib_style should be called during initialization
            # and should handle the fallback gracefully
            assert plotter.config.style == PlotStyle.SCIENTIFIC
    
    def test_plot_eternal_ratios_with_save_path(self, mock_plotter, sample_eternal_ratios):
        """Test plotting eternal ratios with save path."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.tight_layout'):
            
            mock_fig = Mock()
            mock_ax1, mock_ax2 = Mock(), Mock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
            plotter = PlotUtils(config=config)
            
            plotter.plot_eternal_ratios(sample_eternal_ratios, save_path="test.png")
            
            mock_savefig.assert_called_once_with("test.png", dpi=plotter.config.dpi, bbox_inches='tight')
    
    def test_plot_phase_space_with_save_path(self, mock_plotter, sample_absolute_values):
        """Test phase space plotting with save path."""
        with patch('plotly.graph_objects.Figure') as mock_figure:
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            plotter.plot_act_phase_space(sample_absolute_values, save_path="test.html")
            
            mock_fig.write_html.assert_called_once_with("test.html")
    
    def test_animation_save_matplotlib(self, mock_plotter):
        """Test saving matplotlib animation."""
        sequences = [[AbsoluteValue(1.0, 1.0)]]
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.animation.FuncAnimation') as mock_anim:
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_animation = Mock()
            mock_anim.return_value = mock_animation
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
            plotter = PlotUtils(config=config)
            
            plotter.animate_sequence_evolution(sequences, save_path="test.gif")
            
            mock_animation.save.assert_called_once_with("test.gif", writer='pillow', fps=5)
    
    def test_dashboard_with_empty_records(self, mock_plotter, sample_absolute_values, sample_eternal_ratios):
        """Test dashboard creation with empty compensation records."""
        with patch('plotly.subplots.make_subplots') as mock_subplots:
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            result = plotter.create_interactive_dashboard(
                sample_absolute_values, sample_eternal_ratios, []
            )
            
            assert result == mock_fig