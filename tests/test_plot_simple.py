"""Simple tests for plotting utilities without direct imports.

This test file uses mocking to test plot functionality without requiring
actual plotting dependencies to be installed.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from decimal import Decimal

# Mock plotting modules before any imports
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.patches'] = MagicMock()
sys.modules['matplotlib.colors'] = MagicMock()
sys.modules['matplotlib.animation'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()
sys.modules['plotly.figure_factory'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import the modules we need to test
from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.logic.compensator import CompensationRecord


class TestPlotEnums:
    """Test plot enumeration classes."""
    
    def test_plot_style_enum(self):
        """Test PlotStyle enum values."""
        # Import inside test to avoid import issues
        from balansis.utils.plot import PlotStyle
        
        assert PlotStyle.SCIENTIFIC == "scientific"
        assert PlotStyle.ELEGANT == "elegant"
        assert PlotStyle.MINIMAL == "minimal"
        assert PlotStyle.COLORFUL == "colorful"
        
        # Test enum membership
        assert "scientific" in [style.value for style in PlotStyle]
        assert len(list(PlotStyle)) == 4
    
    def test_plot_backend_enum(self):
        """Test PlotBackend enum values."""
        from balansis.utils.plot import PlotBackend
        
        assert PlotBackend.MATPLOTLIB == "matplotlib"
        assert PlotBackend.PLOTLY == "plotly"
        
        # Test enum membership
        assert "matplotlib" in [backend.value for backend in PlotBackend]
        assert len(list(PlotBackend)) == 2


class TestPlotConfig:
    """Test PlotConfig class."""
    
    def test_default_config(self):
        """Test default PlotConfig values."""
        from balansis.utils.plot import PlotConfig, PlotStyle, PlotBackend
        
        config = PlotConfig()
        
        assert config.style == PlotStyle.SCIENTIFIC
        assert config.backend == PlotBackend.MATPLOTLIB
        assert config.width == 800
        assert config.height == 600
        assert config.dpi == 100
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
        from balansis.utils.plot import PlotConfig, PlotStyle, PlotBackend
        
        config = PlotConfig(
            style=PlotStyle.ELEGANT,
            backend=PlotBackend.PLOTLY,
            width=1200,
            height=800,
            dpi=150,
            alpha=0.6
        )
        
        assert config.style == PlotStyle.ELEGANT
        assert config.backend == PlotBackend.PLOTLY
        assert config.width == 1200
        assert config.height == 800
        assert config.dpi == 150
        assert config.alpha == 0.6
    
    def test_config_validation(self):
        """Test PlotConfig validation."""
        from balansis.utils.plot import PlotConfig
        
        # Test invalid width
        with pytest.raises(ValueError, match="Width must be positive"):
            PlotConfig(width=0)
        
        with pytest.raises(ValueError, match="Width must be positive"):
            PlotConfig(width=-100)
        
        # Test invalid height
        with pytest.raises(ValueError, match="Height must be positive"):
            PlotConfig(height=0)
        
        with pytest.raises(ValueError, match="Height must be positive"):
            PlotConfig(height=-50)
        
        # Test invalid dpi
        with pytest.raises(ValueError, match="DPI must be positive"):
            PlotConfig(dpi=0)
        
        with pytest.raises(ValueError, match="DPI must be positive"):
            PlotConfig(dpi=-72)
        
        # Test invalid alpha
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            PlotConfig(alpha=-0.1)
        
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            PlotConfig(alpha=1.1)
        
        # Test valid boundary values
        config = PlotConfig(alpha=0.0)
        assert config.alpha == 0.0
        
        config = PlotConfig(alpha=1.0)
        assert config.alpha == 1.0


class TestPlotUtilsBasic:
    """Test basic PlotUtils functionality."""
    
    def test_plot_utils_initialization(self):
        """Test PlotUtils initialization."""
        # Mock the dependencies to avoid import errors
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True):
            
            from balansis.utils.plot import PlotUtils, PlotConfig
            
            # Test default initialization
            plotter = PlotUtils()
            assert plotter.config is not None
            assert plotter.operations is not None
            assert plotter.compensator is not None
            
            # Test custom config
            custom_config = PlotConfig(width=1000, height=700)
            plotter = PlotUtils(config=custom_config)
            assert plotter.config.width == 1000
            assert plotter.config.height == 700
    
    def test_dependency_validation_missing_numpy(self):
        """Test dependency validation when numpy is missing."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', False):
            from balansis.utils.plot import PlotUtils
            
            with pytest.raises(ImportError, match="NumPy is required"):
                PlotUtils()
    
    def test_dependency_validation_missing_matplotlib(self):
        """Test dependency validation when matplotlib is missing."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', False):
            
            from balansis.utils.plot import PlotUtils, PlotConfig, PlotBackend
            
            config = PlotConfig(backend=PlotBackend.MATPLOTLIB)
            with pytest.raises(ImportError, match="Matplotlib is required"):
                PlotUtils(config=config)
    
    def test_dependency_validation_missing_plotly(self):
        """Test dependency validation when plotly is missing."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', False):
            
            from balansis.utils.plot import PlotUtils, PlotConfig, PlotBackend
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            with pytest.raises(ImportError, match="Plotly is required"):
                PlotUtils(config=config)


class TestPlotUtilsMocked:
    """Test PlotUtils with mocked dependencies."""
    
    @pytest.fixture
    def mock_plotter(self):
        """Create a PlotUtils instance with mocked dependencies."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True):
            
            from balansis.utils.plot import PlotUtils
            return PlotUtils()
    
    @pytest.fixture
    def sample_absolute_values(self):
        """Create sample AbsoluteValue objects for testing."""
        return [
            AbsoluteValue(magnitude=Decimal('1'), direction=Decimal('1')),
            AbsoluteValue(magnitude=Decimal('2'), direction=Decimal('-1')),
            AbsoluteValue(magnitude=Decimal('3'), direction=Decimal('1')),
            AbsoluteValue(magnitude=Decimal('0'), direction=Decimal('1')),
            AbsoluteValue(magnitude=Decimal('4'), direction=Decimal('-1'))
        ]
    
    @pytest.fixture
    def sample_eternal_ratios(self, sample_absolute_values):
        """Create sample EternalRatio objects for testing."""
        return [
            EternalRatio(sample_absolute_values[0], sample_absolute_values[1]),
            EternalRatio(sample_absolute_values[2], sample_absolute_values[3]),
            EternalRatio(sample_absolute_values[4], sample_absolute_values[0])
        ]
    
    def test_plot_absolute_values_matplotlib(self, mock_plotter, sample_absolute_values):
        """Test plotting AbsoluteValues with matplotlib backend."""
        with patch('balansis.utils.plot.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            result = mock_plotter.plot_absolute_values(
                sample_absolute_values,
                title="Test Plot",
                xlabel="Index",
                ylabel="Magnitude"
            )
            
            # Verify matplotlib functions were called
            mock_plt.subplots.assert_called_once()
            mock_ax.scatter.assert_called_once()
            mock_ax.set_title.assert_called_once_with("Test Plot", fontsize=14, pad=20)
            mock_ax.set_xlabel.assert_called_once_with("Index", fontsize=14)
            mock_ax.set_ylabel.assert_called_once_with("Magnitude", fontsize=14)
    
    def test_plot_absolute_values_plotly(self, sample_absolute_values):
        """Test plotting AbsoluteValues with plotly backend."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True), \
             patch('balansis.utils.plot.go') as mock_go:
            
            from balansis.utils.plot import PlotUtils, PlotConfig, PlotBackend
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig
            
            result = plotter.plot_absolute_values(
                sample_absolute_values,
                title="Test Plotly Plot"
            )
            
            # Verify plotly functions were called
            mock_go.Figure.assert_called_once()
            mock_fig.add_trace.assert_called()
            mock_fig.update_layout.assert_called()
    
    def test_empty_input_handling(self, mock_plotter):
        """Test handling of empty input lists."""
        # Test with empty list
        with patch('balansis.utils.plot.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            result = mock_plotter.plot_absolute_values([])
            
            # Should still create plot but with empty data
            mock_plt.subplots.assert_called_once()
            mock_ax.scatter.assert_called_once()


class TestPlotUtilsEdgeCases:
    """Test edge cases for PlotUtils."""
    
    def test_invalid_save_path(self):
        """Test handling of invalid save paths."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.plt') as mock_plt:
            
            from balansis.utils.plot import PlotUtils
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            mock_plt.savefig.side_effect = OSError("Invalid path")
            
            plotter = PlotUtils()
            values = [AbsoluteValue(Decimal('1'), Decimal('1'))]
            
            # Should handle save error gracefully
            with pytest.raises(OSError):
                plotter.plot_absolute_values(values, save_path="/invalid/path/plot.png")
    
    def test_matplotlib_style_fallback(self):
        """Test matplotlib style fallback when seaborn is not available."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.plt') as mock_plt:
            
            from balansis.utils.plot import PlotUtils, PlotConfig, PlotStyle
            
            # Mock style.use to raise OSError for seaborn styles
            mock_plt.style.use.side_effect = OSError("Style not found")
            
            config = PlotConfig(style=PlotStyle.SCIENTIFIC)
            plotter = PlotUtils(config=config)
            
            # Should fallback to default style
            mock_plt.style.use.assert_called_with('default')


class TestAdvancedPlotting:
    """Test advanced plotting methods for comprehensive coverage."""
    
    @pytest.fixture
    def mock_plotter(self):
        """Create a PlotUtils instance with mocked dependencies."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True):
            
            from balansis.utils.plot import PlotUtils
            return PlotUtils()
    
    @pytest.fixture
    def sample_compensation_records(self):
        """Create sample CompensationRecord objects for testing."""
        from balansis.logic.compensator import CompensationType
        return [
            CompensationRecord(
                operation_type="addition",
                compensation_type=CompensationType.STABILITY,
                original_values=[AbsoluteValue(magnitude=Decimal('1'), direction=Decimal('1'))],
                compensated_values=[AbsoluteValue(magnitude=Decimal('1.1'), direction=Decimal('1'))],
                compensation_factor=0.1,
                stability_metric=0.95,
                timestamp=1.0
            ),
            CompensationRecord(
                operation_type="subtraction",
                compensation_type=CompensationType.BALANCE,
                original_values=[AbsoluteValue(magnitude=Decimal('2'), direction=Decimal('-1'))],
                compensated_values=[AbsoluteValue(magnitude=Decimal('1.9'), direction=Decimal('-1'))],
                compensation_factor=0.1,
                stability_metric=0.88,
                timestamp=2.0
            )
        ]
    
    def test_plot_compensation_analysis_matplotlib(self, mock_plotter, sample_compensation_records):
        """Test compensation analysis plotting with matplotlib."""
        with patch('balansis.utils.plot.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax1, mock_ax2, mock_ax3, mock_ax4 = Mock(), Mock(), Mock(), Mock()
            mock_axes = [[mock_ax1, mock_ax2], [mock_ax3, mock_ax4]]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = mock_plotter.plot_compensation_analysis(
                sample_compensation_records,
                title="Compensation Analysis Test"
            )
            
            # Verify matplotlib functions were called
            mock_plt.subplots.assert_called_once()
            # Check that all subplots were configured
            for ax in [mock_ax1, mock_ax2, mock_ax3, mock_ax4]:
                ax.set_title.assert_called()
                ax.set_ylabel.assert_called()
    
    def test_plot_compensation_analysis_plotly(self, sample_compensation_records):
        """Test compensation analysis plotting with plotly."""
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True), \
             patch('balansis.utils.plot.make_subplots') as mock_subplots, \
             patch('balansis.utils.plot.go') as mock_go:
            
            from balansis.utils.plot import PlotUtils, PlotConfig, PlotBackend
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            
            result = plotter.plot_compensation_analysis(
                sample_compensation_records,
                title="Plotly Compensation Analysis"
            )
            
            # Verify plotly functions were called
            mock_subplots.assert_called_once()
            mock_fig.add_trace.assert_called()
            mock_fig.update_layout.assert_called()
    
    def test_plot_act_phase_space_matplotlib(self, mock_plotter):
        """Test ACT phase space plotting with matplotlib."""
        sample_values = [
            AbsoluteValue(Decimal('1'), Decimal('1')),
            AbsoluteValue(Decimal('2'), Decimal('-1')),
            AbsoluteValue(Decimal('3'), Decimal('1'))
        ]
        
        with patch('balansis.utils.plot.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            result = mock_plotter.plot_act_phase_space(
                sample_values,
                title="Phase Space Test"
            )
            
            # Verify matplotlib functions were called
            mock_plt.subplots.assert_called_once()
            mock_ax.scatter.assert_called_once()
            mock_ax.set_title.assert_called_once_with("Phase Space Test", fontsize=14, pad=20)
    
    def test_plot_act_phase_space_plotly(self):
        """Test ACT phase space plotting with plotly."""
        sample_values = [
            AbsoluteValue(Decimal('1'), Decimal('1')),
            AbsoluteValue(Decimal('2'), Decimal('-1'))
        ]
        
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True), \
             patch('balansis.utils.plot.go') as mock_go:
            
            from balansis.utils.plot import PlotUtils, PlotConfig, PlotBackend
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig
            
            result = plotter.plot_act_phase_space(
                sample_values,
                title="Plotly Phase Space"
            )
            
            # Verify plotly functions were called
            mock_go.Figure.assert_called_once()
            mock_fig.add_trace.assert_called()
            mock_fig.update_layout.assert_called()
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation."""
        sample_values = [AbsoluteValue(Decimal('1'), Decimal('1'))]
        sample_ratios = [EternalRatio(sample_values[0], sample_values[0])]
        sample_records = [
            CompensationRecord(
                original_value=sample_values[0],
                compensated_value=sample_values[0],
                compensation_factor=Decimal('0.1'),
                operation_type="test",
                stability_metric=Decimal('0.9')
            )
        ]
        
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True), \
             patch('balansis.utils.plot.make_subplots') as mock_subplots, \
             patch('balansis.utils.plot.go') as mock_go:
            
            from balansis.utils.plot import PlotUtils
            
            plotter = PlotUtils()
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            
            result = plotter.create_interactive_dashboard(
                absolute_values=sample_values,
                eternal_ratios=sample_ratios,
                compensation_records=sample_records,
                title="Test Dashboard"
            )
            
            # Verify dashboard creation
            mock_subplots.assert_called_once()
            mock_fig.add_trace.assert_called()
            mock_fig.update_layout.assert_called()
    
    def test_animate_sequence_evolution_matplotlib(self, mock_plotter):
        """Test sequence animation with matplotlib."""
        sequences = [
            [AbsoluteValue(Decimal('1'), Decimal('1'))],
            [AbsoluteValue(Decimal('2'), Decimal('-1'))]
        ]
        
        with patch('balansis.utils.plot.plt') as mock_plt, \
             patch('balansis.utils.plot.FuncAnimation') as mock_anim:
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            mock_animation = Mock()
            mock_anim.return_value = mock_animation
            
            result = mock_plotter.animate_sequence_evolution(
                sequences,
                title="Animation Test"
            )
            
            # Verify animation setup
            mock_plt.subplots.assert_called_once()
            mock_anim.assert_called_once()
    
    def test_animate_sequence_evolution_plotly(self):
        """Test sequence animation with plotly."""
        sequences = [
            [AbsoluteValue(Decimal('1'), Decimal('1'))],
            [AbsoluteValue(Decimal('2'), Decimal('-1'))]
        ]
        
        with patch('balansis.utils.plot.NUMPY_AVAILABLE', True), \
             patch('balansis.utils.plot.MATPLOTLIB_AVAILABLE', True), \
             patch('balansis.utils.plot.PLOTLY_AVAILABLE', True), \
             patch('balansis.utils.plot.go') as mock_go:
            
            from balansis.utils.plot import PlotUtils, PlotConfig, PlotBackend
            
            config = PlotConfig(backend=PlotBackend.PLOTLY)
            plotter = PlotUtils(config=config)
            
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig
            
            result = plotter.animate_sequence_evolution(
                sequences,
                title="Plotly Animation"
            )
            
            # Verify plotly animation setup
            mock_go.Figure.assert_called_once()
            mock_fig.add_trace.assert_called()
            mock_fig.update_layout.assert_called()
    
    def test_export_plot_data_csv(self, mock_plotter):
        """Test exporting plot data to CSV format."""
        sample_values = [AbsoluteValue(Decimal('1'), Decimal('1'))]
        
        with patch('balansis.utils.plot.pd') as mock_pd:
            mock_df = Mock()
            mock_pd.DataFrame.return_value = mock_df
            
            result = mock_plotter.export_plot_data(
                sample_values,
                filename="test_export.csv",
                format="csv"
            )
            
            # Verify CSV export
            mock_pd.DataFrame.assert_called_once()
            mock_df.to_csv.assert_called_once_with("test_export.csv", index=False)
    
    def test_export_plot_data_json(self, mock_plotter):
        """Test exporting plot data to JSON format."""
        sample_values = [AbsoluteValue(Decimal('1'), Decimal('1'))]
        
        with patch('balansis.utils.plot.pd') as mock_pd:
            mock_df = Mock()
            mock_pd.DataFrame.return_value = mock_df
            
            result = mock_plotter.export_plot_data(
                sample_values,
                filename="test_export.json",
                format="json"
            )
            
            # Verify JSON export
            mock_pd.DataFrame.assert_called_once()
            mock_df.to_json.assert_called_once_with("test_export.json", orient="records", indent=2)
    
    def test_export_plot_data_excel(self, mock_plotter):
        """Test exporting plot data to Excel format."""
        sample_values = [AbsoluteValue(Decimal('1'), Decimal('1'))]
        
        with patch('balansis.utils.plot.pd') as mock_pd:
            mock_df = Mock()
            mock_pd.DataFrame.return_value = mock_df
            
            result = mock_plotter.export_plot_data(
                sample_values,
                filename="test_export.xlsx",
                format="excel"
            )
            
            # Verify Excel export
            mock_pd.DataFrame.assert_called_once()
            mock_df.to_excel.assert_called_once_with("test_export.xlsx", index=False)
    
    def test_export_plot_data_invalid_format(self, mock_plotter):
        """Test handling of invalid export format."""
        sample_values = [AbsoluteValue(Decimal('1'), Decimal('1'))]
        
        with pytest.raises(ValueError, match="Unsupported format"):
            mock_plotter.export_plot_data(
                sample_values,
                filename="test_export.txt",
                format="txt"
            )
    
    def test_empty_data_handling_advanced(self, mock_plotter):
        """Test advanced plotting methods with empty data."""
        # Test compensation analysis with empty records
        with patch('balansis.utils.plot.plt') as mock_plt:
            mock_fig = Mock()
            mock_axes = [Mock(), Mock(), Mock()]
            mock_plt.subplots.return_value = (mock_fig, mock_axes)
            
            result = mock_plotter.plot_compensation_analysis([])
            mock_plt.subplots.assert_called_once()
        
        # Test phase space with empty values
        with patch('balansis.utils.plot.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            result = mock_plotter.plot_act_phase_space([])
            mock_plt.subplots.assert_called_once()
        
        # Test animation with empty sequences
        with patch('balansis.utils.plot.plt') as mock_plt, \
             patch('balansis.utils.plot.FuncAnimation') as mock_anim:
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            result = mock_plotter.animate_sequence_evolution([])
            mock_plt.subplots.assert_called_once()
    
    def test_error_handling_advanced_plotting(self, mock_plotter):
        """Test error handling in advanced plotting methods."""
        sample_values = [AbsoluteValue(Decimal('1'), Decimal('1'))]
        
        # Test matplotlib error handling
        with patch('balansis.utils.plot.plt') as mock_plt:
            mock_plt.subplots.side_effect = Exception("Matplotlib error")
            
            with pytest.raises(Exception, match="Matplotlib error"):
                mock_plotter.plot_act_phase_space(sample_values)
        
        # Test export error handling
        with patch('balansis.utils.plot.pd') as mock_pd:
            mock_df = Mock()
            mock_pd.DataFrame.return_value = mock_df
            mock_df.to_csv.side_effect = IOError("File write error")
            
            with pytest.raises(IOError, match="File write error"):
                mock_plotter.export_plot_data(
                    sample_values,
                    filename="test_export.csv",
                    format="csv"
                )