# Parameter Comparison Charts Guide

The test suite has generated **12 separate chart files** analyzing the fetal development simulator parameters.

## 📊 Generated Charts

### 1. Individual Parameter Effect Charts (5 files)
Each parameter has its own dedicated chart showing effects on all 5 metrics:

- **`chart_temperature_effects.png`** - Temperature's impact on development
- **`chart_adhesion_effects.png`** - Adhesion's impact on development  
- **`chart_growth_rate_effects.png`** - Growth rate's impact on development
- **`chart_differentiation_rate_effects.png`** - Differentiation rate's impact on development
- **`chart_morphogen_diffusion_effects.png`** - Morphogen diffusion's impact on development

Each chart contains 5 subplots showing:
- Total Cell Count
- Number of Organs Developed
- Differentiation Speed
- Spatial Spread
- Cell Type Diversity

### 2. Individual Metric Comparison Charts (5 files)
Each metric has its own chart comparing all 5 parameters:

- **`chart_metric_cells_comparison.png`** - All parameters' effects on total cell count
- **`chart_metric_organ_count_comparison.png`** - All parameters' effects on organ development
- **`chart_metric_diff_speed_comparison.png`** - All parameters' effects on differentiation speed
- **`chart_metric_spatial_comparison.png`** - All parameters' effects on spatial spread
- **`chart_metric_diversity_comparison.png`** - All parameters' effects on cell diversity

### 3. Overview Charts (2 files)

- **`chart_complete_overview.png`** - 6-panel overview showing all metrics and parameters together
- **`chart_sensitivity_heatmap.png`** - Heatmap showing which parameters most affect each metric

## 🔬 Parameters Tested

| Parameter | Values Tested | Description |
|-----------|--------------|-------------|
| **Temperature** | 1.0, 5.0, 12.0, 20.0, 30.0 | Cell movement randomness |
| **Adhesion** | 1.0, 5.0, 10.0, 20.0, 30.0 | Cell-cell stickiness |
| **Growth Rate** | 0.05, 0.15, 0.25, 0.5, 1.0 | Development speed |
| **Differentiation Rate** | 0.01, 0.05, 0.12, 0.25, 0.4 | Cell specialization rate |
| **Morphogen Diffusion** | 0.05, 0.3, 0.6, 1.2, 2.0 | Pattern formation strength |

## 📈 Metrics Measured

1. **Total Cell Count** - Total number of cells (excluding empty space)
2. **Organs Developed** - Count of distinct organ types that formed
3. **Differentiation Speed** - Rate at which cell types diversify
4. **Spatial Spread** - How cells distribute across the grid
5. **Cell Type Diversity** - Number of unique cell types present

## 🎯 Key Findings

All simulations ran to **Week 10** (~70 iterations) to capture early organogenesis.

### Test Results Summary:
- ✅ **176/182 tests passed (96.7%)**
- Temperature range: Affects cell count (13,309 - 14,254 cells)
- Adhesion range: Strong inverse correlation with cell count
- Differentiation rate: Higher rates produce more cell diversity
- All parameters successfully developed **7-8 organs** by week 10

### Chart Features:
- High-resolution (300 DPI) for publication quality
- Min/max values marked with colored triangles
- Value labels on all data points
- Normalized x-axes for cross-parameter comparison
- Professional color scheme and styling

## 📁 File Locations

All charts saved in: `C:\Users\almir\COMPUTATIONAL-BIOLOGY\`

## 🔍 How to Use

1. **Compare specific parameters**: Open individual effect charts (e.g., `chart_temperature_effects.png`)
2. **Compare effects on one metric**: Open metric comparison charts (e.g., `chart_metric_diversity_comparison.png`)
3. **See overall patterns**: Check `chart_complete_overview.png`
4. **Identify sensitive parameters**: Review `chart_sensitivity_heatmap.png`

Each chart is publication-ready at 300 DPI resolution with clear labels and legends.
