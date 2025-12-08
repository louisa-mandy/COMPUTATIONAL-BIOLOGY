"""
Comprehensive Test Suite for Fetal Development Simulator
Tests all parameters, outputs, and simulation states
Generates comparison charts for parameter effects
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from Baby_Organ import (
    FetalDevelopmentSimulator, 
    CellType, 
    COLORS, 
    MILESTONES, 
    STAGE_NAMES,
    GRID_SIZE
)

class SimulatorTester:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
        
    def log_test(self, test_name, passed, details=""):
        """Log test results"""
        status = "PASS" if passed else "FAIL"
        result = f"[{status}] {test_name}"
        if details:
            result += f" - {details}"
        self.test_results.append(result)
        
        if passed:
            self.tests_passed += 1
            print(f"✓ {result}")
        else:
            self.tests_failed += 1
            print(f"✗ {result}")
    
    def test_initialization(self):
        """Test simulator initialization"""
        print("\n=== Testing Initialization ===")
        
        try:
            sim = FetalDevelopmentSimulator()
            self.log_test("Simulator creation", True)
        except Exception as e:
            self.log_test("Simulator creation", False, str(e))
            return None
        
        # Test grid initialization
        self.log_test("Grid shape", sim.grid.shape == (GRID_SIZE, GRID_SIZE), 
                     f"Shape: {sim.grid.shape}")
        
        # Test initial cell types
        unique_types = np.unique(sim.grid)
        has_embryo = int(CellType.EMBRYO) in unique_types
        has_placenta = int(CellType.PLACENTA) in unique_types
        self.log_test("Initial embryo cells", has_embryo, 
                     f"Unique types: {unique_types}")
        self.log_test("Initial placenta cells", has_placenta)
        
        # Test morphogen fields
        self.log_test("Morphogen head shape", sim.morphogen_head.shape == (GRID_SIZE, GRID_SIZE))
        self.log_test("Morphogen body shape", sim.morphogen_body.shape == (GRID_SIZE, GRID_SIZE))
        
        return sim
    
    def test_parameters(self, sim):
        """Test all simulation parameters"""
        print("\n=== Testing Parameters ===")
        
        expected_params = ['temperature', 'adhesion', 'volume_constraint', 
                          'growth_rate', 'differentiation_rate', 
                          'morphogen_diffusion', 'speed']
        
        for param in expected_params:
            exists = param in sim.params
            self.log_test(f"Parameter '{param}' exists", exists)
            
            if exists:
                value = sim.params[param]
                is_numeric = isinstance(value, (int, float))
                self.log_test(f"Parameter '{param}' is numeric", is_numeric, 
                             f"Value: {value}, Type: {type(value)}")
                
                is_positive = value > 0
                self.log_test(f"Parameter '{param}' is positive", is_positive)
    
    def test_parameter_ranges(self, sim):
        """Test parameter value ranges"""
        print("\n=== Testing Parameter Ranges ===")
        
        # Test slider definitions
        for slider in sim.sliders:
            param_name = slider['name']
            current_value = sim.params[param_name]
            min_val = slider['min']
            max_val = slider['max']
            
            in_range = min_val <= current_value <= max_val
            self.log_test(f"'{param_name}' within slider range", in_range,
                         f"{min_val} <= {current_value} <= {max_val}")
    
    def test_cell_types(self):
        """Test all cell type definitions"""
        print("\n=== Testing Cell Types ===")
        
        cell_types = [
            CellType.EMPTY, CellType.EMBRYO, CellType.HEAD, CellType.BRAIN,
            CellType.BODY, CellType.ARM_LEFT, CellType.ARM_RIGHT, 
            CellType.LEG_LEFT, CellType.LEG_RIGHT, CellType.PLACENTA,
            CellType.UMBILICAL, CellType.HEART, CellType.LIVER, 
            CellType.STOMACH, CellType.INTESTINE, CellType.KIDNEY,
            CellType.LUNG, CellType.EYE, CellType.BLADDER
        ]
        
        for cell_type in cell_types:
            # Check color definition
            has_color = cell_type in COLORS
            self.log_test(f"CellType.{cell_type.name} has color", has_color)
            
            if has_color:
                color = COLORS[cell_type]
                is_rgb = isinstance(color, tuple) and len(color) == 3
                self.log_test(f"CellType.{cell_type.name} color is RGB", is_rgb,
                             f"Color: {color}")
                
                if is_rgb:
                    valid_rgb = all(0 <= c <= 255 for c in color)
                    self.log_test(f"CellType.{cell_type.name} RGB values valid", 
                                 valid_rgb)
    
    def test_week_calculation(self, sim):
        """Test week calculation from iterations"""
        print("\n=== Testing Week Calculation ===")
        
        test_cases = [
            (0, 1),    # iteration 0 -> week 1
            (7, 2),    # iteration 7 -> week 2
            (70, 11),  # iteration 70 -> week 11
            (280, 40), # iteration 280 -> week 40
            (300, 40), # iteration 300 -> week 40 (capped)
        ]
        
        for iteration, expected_week in test_cases:
            sim.iteration = iteration
            actual_week = sim.week()
            correct = actual_week == expected_week
            self.log_test(f"Iteration {iteration} -> Week {expected_week}", correct,
                         f"Got week {actual_week}")
    
    def test_milestones(self):
        """Test milestone definitions"""
        print("\n=== Testing Milestones ===")
        
        # Check all weeks 1-40 have a milestone
        all_weeks_covered = []
        for week in range(1, 41):
            covered = any(week in wk_range for wk_range in MILESTONES.keys())
            all_weeks_covered.append(covered)
        
        self.log_test("All weeks 1-40 have milestones", all(all_weeks_covered),
                     f"Covered: {sum(all_weeks_covered)}/40 weeks")
        
        # Check milestone text is not empty
        for wk_range, text in MILESTONES.items():
            is_valid = isinstance(text, str) and len(text) > 0
            self.log_test(f"Milestone for {wk_range} has text", is_valid)
    
    def test_stage_names(self):
        """Test stage name definitions"""
        print("\n=== Testing Stage Names ===")
        
        expected_stages = [1, 2, 3, 4, 5]
        for stage in expected_stages:
            exists = stage in STAGE_NAMES
            self.log_test(f"Stage {stage} has name", exists,
                         f"Name: {STAGE_NAMES.get(stage, 'N/A')}")
    
    def test_morphogen_update(self, sim):
        """Test morphogen field updates"""
        print("\n=== Testing Morphogen Updates ===")
        
        # Store initial state
        initial_head = sim.morphogen_head.copy()
        initial_body = sim.morphogen_body.copy()
        
        # Update morphogens
        sim.update_morphogens()
        
        # Check that morphogens changed
        head_changed = not np.array_equal(initial_head, sim.morphogen_head)
        body_changed = not np.array_equal(initial_body, sim.morphogen_body)
        
        self.log_test("Morphogen head updated", head_changed)
        self.log_test("Morphogen body updated", body_changed)
        
        # Check morphogen values are non-negative
        head_nonneg = np.all(sim.morphogen_head >= 0)
        body_nonneg = np.all(sim.morphogen_body >= 0)
        
        self.log_test("Morphogen head non-negative", head_nonneg)
        self.log_test("Morphogen body non-negative", body_nonneg)
    
    def test_anatomical_template(self, sim):
        """Test anatomical template generation for different weeks"""
        print("\n=== Testing Anatomical Templates ===")
        
        test_weeks = [1, 5, 10, 15, 20, 30, 40]
        
        for week in test_weeks:
            template = sim.get_anatomical_template(week)
            is_dict = isinstance(template, dict)
            self.log_test(f"Week {week} template is dict", is_dict)
            
            if is_dict:
                has_content = len(template) > 0
                self.log_test(f"Week {week} template has content", has_content,
                             f"Keys: {list(template.keys())}")
                
                # Check organ appearance timing
                if week >= 5:
                    has_organs = any(key in template for key in 
                                   ['heart', 'liver', 'lungs', 'stomach'])
                    self.log_test(f"Week {week} has organ templates", has_organs)
    
    def test_volume_calculation(self, sim):
        """Test cell volume calculations"""
        print("\n=== Testing Volume Calculations ===")
        
        volumes = sim.calculate_volumes()
        
        is_dict = isinstance(volumes, dict)
        self.log_test("Volumes is dictionary", is_dict)
        
        if is_dict:
            has_volumes = len(volumes) > 0
            self.log_test("Volumes calculated", has_volumes,
                         f"Cell types: {len(volumes)}")
            
            # Check all values are positive integers
            all_positive = all(v > 0 for v in volumes.values())
            all_int = all(isinstance(v, (int, np.integer)) for v in volumes.values())
            
            self.log_test("All volumes positive", all_positive)
            self.log_test("All volumes integer", all_int)
            
            # Print volume summary
            print(f"  Volume summary: {volumes}")
    
    def test_energy_calculation(self, sim):
        """Test energy calculation"""
        print("\n=== Testing Energy Calculations ===")
        
        volumes = sim.calculate_volumes()
        
        # Test energy at random positions
        test_positions = [
            (GRID_SIZE // 2, GRID_SIZE // 2),  # Center
            (10, 10),                           # Corner region
            (GRID_SIZE - 10, GRID_SIZE - 10),  # Opposite corner
        ]
        
        for x, y in test_positions:
            current_type = int(sim.grid[x, y])
            energy = sim.calculate_energy(x, y, current_type, volumes)
            
            is_numeric = isinstance(energy, (int, float))
            self.log_test(f"Energy at ({x}, {y}) is numeric", is_numeric,
                         f"Energy: {energy:.4f}")
            
            is_finite = np.isfinite(energy)
            self.log_test(f"Energy at ({x}, {y}) is finite", is_finite)
    
    def test_differentiation(self, sim):
        """Test cell differentiation process"""
        print("\n=== Testing Cell Differentiation ===")
        
        # Get initial cell type counts
        initial_types = set(np.unique(sim.grid))
        
        # Advance simulation to trigger differentiation
        sim.iteration = 50  # ~Week 8
        sim.update_morphogens()
        sim.apply_morphogen_differentiation()
        
        # Get new cell type counts
        final_types = set(np.unique(sim.grid))
        
        # Check if new cell types appeared
        new_types = final_types - initial_types
        self.log_test("New cell types differentiated", len(new_types) > 0,
                     f"New types: {new_types}")
        
        # Check for expected types at week 8
        expected_types = {int(CellType.HEAD), int(CellType.BODY)}
        has_expected = expected_types.intersection(final_types)
        self.log_test("Expected cell types present", len(has_expected) > 0,
                     f"Found: {has_expected}")
    
    def test_monte_carlo_step(self, sim):
        """Test Monte Carlo simulation step"""
        print("\n=== Testing Monte Carlo Step ===")
        
        # Store initial state
        initial_grid = sim.grid.copy()
        
        # Perform MC step
        try:
            sim.monte_carlo_step()
            self.log_test("Monte Carlo step executes", True)
        except Exception as e:
            self.log_test("Monte Carlo step executes", False, str(e))
            return
        
        # Check grid changed (cells moved)
        grid_changed = not np.array_equal(initial_grid, sim.grid)
        self.log_test("Grid state changed after MC step", grid_changed)
        
        # Check grid validity
        all_valid_types = all(
            val in [ct.value for ct in CellType] or val == 0
            for val in np.unique(sim.grid)
        )
        self.log_test("All cell types valid after MC step", all_valid_types)
    
    def test_reset_functionality(self, sim):
        """Test simulation reset"""
        print("\n=== Testing Reset Functionality ===")
        
        # Advance simulation
        sim.iteration = 100
        sim.is_running = True
        sim.monte_carlo_step()
        
        # Reset
        sim.reset_simulation()
        
        self.log_test("Iteration reset to 0", sim.iteration == 0)
        self.log_test("Running state reset", sim.is_running == False)
        self.log_test("Stage reset to 1", sim.current_stage == 1)
        
        # Check grid reinitialized
        has_embryo = int(CellType.EMBRYO) in np.unique(sim.grid)
        self.log_test("Embryo cells present after reset", has_embryo)
    
    def test_simulation_progression(self, sim):
        """Test complete simulation progression"""
        print("\n=== Testing Simulation Progression ===")
        
        sim.reset_simulation()
        
        # Simulate progression through weeks
        test_weeks = [5, 10, 15, 20, 30]
        
        for target_week in test_weeks:
            # Advance to target week
            target_iteration = (target_week - 1) * 7
            while sim.iteration < target_iteration and sim.week() < 40:
                sim.update_morphogens()
                sim.monte_carlo_step()
                sim.apply_morphogen_differentiation()
                sim.iteration += 1
            
            actual_week = sim.week()
            week_correct = abs(actual_week - target_week) <= 1
            self.log_test(f"Progressed to week ~{target_week}", week_correct,
                         f"Actual week: {actual_week}")
            
            # Check cell diversity increases
            unique_types = len(np.unique(sim.grid))
            self.log_test(f"Week {actual_week} has diverse cell types", 
                         unique_types > 2,
                         f"Unique types: {unique_types}")
    
    def test_organ_development(self, sim):
        """Test organ development timing"""
        print("\n=== Testing Organ Development ===")
        
        sim.reset_simulation()
        
        # Test organ appearance at appropriate weeks
        organ_tests = [
            (5, CellType.HEART, "Heart"),
            (6, CellType.LIVER, "Liver"),
            (7, CellType.STOMACH, "Stomach"),
            (8, CellType.INTESTINE, "Intestines"),
            (9, CellType.KIDNEY, "Kidneys"),
            (6, CellType.LUNG, "Lungs"),
            (5, CellType.EYE, "Eyes"),
        ]
        
        for week, organ_type, organ_name in organ_tests:
            # Advance to target week
            sim.iteration = (week - 1) * 7
            sim.update_morphogens()
            sim.apply_morphogen_differentiation()
            
            # Check if organ appears
            has_organ = int(organ_type) in np.unique(sim.grid)
            self.log_test(f"{organ_name} appears by week {week}", has_organ)
    
    def test_ui_elements(self, sim):
        """Test UI element definitions"""
        print("\n=== Testing UI Elements ===")
        
        # Test sliders
        self.log_test("Sliders defined", len(sim.sliders) > 0,
                     f"Slider count: {len(sim.sliders)}")
        
        for slider in sim.sliders:
            required_keys = ['name', 'label', 'min', 'max', 'rect', 'description']
            has_all_keys = all(key in slider for key in required_keys)
            self.log_test(f"Slider '{slider.get('name', 'unknown')}' complete", 
                         has_all_keys)
        
        # Test fonts
        has_fonts = all([
            sim.font is not None,
            sim.small_font is not None,
            sim.title_font is not None
        ])
        self.log_test("Fonts initialized", has_fonts)
    
    def test_camera_controls(self, sim):
        """Test camera/panning functionality"""
        print("\n=== Testing Camera Controls ===")
        
        initial_cam_x = sim.cam_x
        initial_cam_y = sim.cam_y
        
        # Simulate camera movement
        sim.cam_x += 50
        sim.cam_y -= 30
        
        x_changed = sim.cam_x != initial_cam_x
        y_changed = sim.cam_y != initial_cam_y
        
        self.log_test("Camera X movement", x_changed,
                     f"{initial_cam_x} -> {sim.cam_x}")
        self.log_test("Camera Y movement", y_changed,
                     f"{initial_cam_y} -> {sim.cam_y}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Passed: {self.tests_passed} ({pass_rate:.1f}%)")
        print(f"Failed: {self.tests_failed}")
        print("="*60)
        
        if self.tests_failed > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if "[FAIL]" in result:
                    print(f"  {result}")
        
        return self.tests_failed == 0
    
    def generate_parameter_comparison_charts(self):
        """Generate comprehensive comparison charts for all parameters"""
        print("\n=== Generating Parameter Comparison Charts ===")
        
        # Parameter ranges to test
        param_configs = {
            'temperature': {'values': [1.0, 5.0, 12.0, 20.0, 30.0], 'label': 'Temperature'},
            'adhesion': {'values': [1.0, 5.0, 10.0, 20.0, 30.0], 'label': 'Adhesion'},
            'growth_rate': {'values': [0.05, 0.15, 0.25, 0.5, 1.0], 'label': 'Growth Rate'},
            'differentiation_rate': {'values': [0.01, 0.05, 0.12, 0.25, 0.4], 'label': 'Differentiation Rate'},
            'morphogen_diffusion': {'values': [0.05, 0.3, 0.6, 1.2, 2.0], 'label': 'Morphogen Diffusion'}
        }
        
        results = {}
        
        # Run simulations for each parameter
        for param_name, config in param_configs.items():
            print(f"\n  Testing {config['label']}...")
            results[param_name] = self._test_parameter_variations(param_name, config['values'])
        
        # Generate visualizations
        self._plot_comparison_charts(results, param_configs)
        
        print("\n✓ Charts generated and saved!")
    
    def _test_parameter_variations(self, param_name, values):
        """Test variations of a single parameter"""
        results = {
            'values': values,
            'cell_counts': [],
            'organ_counts': [],
            'differentiation_speed': [],
            'spatial_spread': [],
            'final_diversity': []
        }
        
        base_iterations = 70  # ~Week 10
        
        for value in values:
            print(f"    {param_name} = {value:.3f}...", end=" ")
            
            # Create simulator with modified parameter
            sim = FetalDevelopmentSimulator()
            sim.params[param_name] = value
            sim.reset_simulation()
            
            # Track metrics over time
            cell_type_progression = []
            organ_appearance_time = {}
            
            # Run simulation
            for i in range(base_iterations):
                sim.update_morphogens()
                sim.monte_carlo_step()
                sim.apply_morphogen_differentiation()
                sim.iteration += 1
                
                # Track cell diversity
                unique_types = len(np.unique(sim.grid))
                cell_type_progression.append(unique_types)
                
                # Track organ appearance
                for organ in [CellType.HEART, CellType.LIVER, CellType.LUNG, 
                             CellType.STOMACH, CellType.INTESTINE, CellType.KIDNEY]:
                    if int(organ) in np.unique(sim.grid) and organ not in organ_appearance_time:
                        organ_appearance_time[organ] = i
            
            # Calculate metrics
            volumes = sim.calculate_volumes()
            total_cells = sum(volumes.values())
            
            # Count organs
            organ_types = [CellType.HEART, CellType.LIVER, CellType.LUNG, 
                          CellType.STOMACH, CellType.INTESTINE, CellType.KIDNEY, 
                          CellType.EYE, CellType.BLADDER]
            organ_count = sum(1 for org in organ_types if int(org) in volumes)
            
            # Differentiation speed (how fast cell types increase)
            diff_speed = np.mean(np.diff(cell_type_progression)) if len(cell_type_progression) > 1 else 0
            
            # Spatial spread (how spread out cells are)
            non_empty = np.argwhere(sim.grid != int(CellType.EMPTY))
            if len(non_empty) > 0:
                spatial_spread = np.std(non_empty, axis=0).mean()
            else:
                spatial_spread = 0
            
            # Final diversity
            final_diversity = len(np.unique(sim.grid))
            
            results['cell_counts'].append(total_cells)
            results['organ_counts'].append(organ_count)
            results['differentiation_speed'].append(diff_speed)
            results['spatial_spread'].append(spatial_spread)
            results['final_diversity'].append(final_diversity)
            
            print(f"Done (cells: {total_cells}, organs: {organ_count})")
        
        return results
    
    def _plot_comparison_charts(self, results, param_configs):
        """Create separate visualization for each parameter's effects"""
        
        params = list(results.keys())
        metrics = [
            ('cell_counts', 'Total Cell Count'),
            ('organ_counts', 'Number of Organs Developed'),
            ('differentiation_speed', 'Differentiation Speed'),
            ('spatial_spread', 'Spatial Spread'),
            ('final_diversity', 'Cell Type Diversity')
        ]
        
        # Color scheme for each parameter
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # Create individual chart for each parameter
        for param_idx, param_name in enumerate(params):
            config = param_configs[param_name]
            data = results[param_name]
            
            # Create figure with 5 subplots (one for each metric)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{config["label"]} - Effects on Development Metrics', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Flatten axes for easier iteration
            axes_flat = axes.flatten()
            
            for metric_idx, (metric_key, metric_label) in enumerate(metrics):
                ax = axes_flat[metric_idx]
                
                # Plot line with markers
                ax.plot(data['values'], data[metric_key], 
                       marker='o', linewidth=3, markersize=10,
                       color=colors[param_idx], label=metric_label,
                       markerfacecolor='white', markeredgewidth=2,
                       markeredgecolor=colors[param_idx])
                
                # Fill area under curve
                ax.fill_between(data['values'], data[metric_key], 
                               alpha=0.2, color=colors[param_idx])
                
                ax.set_xlabel(config['label'], fontsize=12, fontweight='bold')
                ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
                ax.set_title(metric_label, fontsize=13, fontweight='bold', pad=12)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(2)
                ax.spines['bottom'].set_linewidth(2)
                
                # Add value labels on points
                for x, y in zip(data['values'], data[metric_key]):
                    ax.annotate(f'{y:.2f}', (x, y), 
                              textcoords="offset points", 
                              xytext=(0, 10), ha='center', 
                              fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='white', 
                                      edgecolor=colors[param_idx],
                                      alpha=0.8))
                
                # Add min/max annotations
                min_idx = np.argmin(data[metric_key])
                max_idx = np.argmax(data[metric_key])
                ax.scatter(data['values'][min_idx], data[metric_key][min_idx], 
                          color='red', s=100, zorder=5, marker='v', alpha=0.6)
                ax.scatter(data['values'][max_idx], data[metric_key][max_idx], 
                          color='green', s=100, zorder=5, marker='^', alpha=0.6)
            
            # Hide the last subplot (we only have 5 metrics)
            axes_flat[5].axis('off')
            
            plt.tight_layout()
            
            # Save individual parameter chart
            filename = f'chart_{param_name}_effects.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close(fig)
        
        # Create summary comparison chart
        self._plot_summary_comparison(results, param_configs)
        
        plt.close('all')
    
    def _plot_summary_comparison(self, results, param_configs):
        """Create individual charts for each metric comparing all parameters"""
        
        metrics = [
            ('cell_counts', 'Total Cell Count', 'cells'),
            ('organ_counts', 'Organs Developed', 'organ_count'),
            ('differentiation_speed', 'Differentiation Speed', 'diff_speed'),
            ('spatial_spread', 'Spatial Spread', 'spatial'),
            ('final_diversity', 'Cell Type Diversity', 'diversity')
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        markers = ['o', 's', '^', 'D', 'v']
        
        # Create individual chart for each metric
        for metric_key, metric_label, filename_suffix in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for param_idx, (param_name, data) in enumerate(results.items()):
                config = param_configs[param_name]
                
                # Normalize x-axis to 0-1 for comparison
                normalized_x = np.linspace(0, 1, len(data['values']))
                
                ax.plot(normalized_x, data[metric_key], 
                       marker=markers[param_idx], linewidth=2.5, markersize=9,
                       color=colors[param_idx], label=config['label'],
                       alpha=0.85, markeredgewidth=2, markeredgecolor='white')
            
            ax.set_xlabel('Parameter Value (normalized: min → max)', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=13, fontweight='bold')
            ax.set_title(f'{metric_label} - All Parameters Comparison', 
                        fontsize=15, fontweight='bold', pad=15)
            ax.legend(fontsize=11, loc='best', framealpha=0.9, 
                     edgecolor='gray', fancybox=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            
            # Add background color
            ax.set_facecolor('#F8F9FA')
            
            plt.tight_layout()
            filename = f'chart_metric_{filename_suffix}_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close(fig)
        
        # Create combined overview chart
        self._plot_combined_overview(results, param_configs)
        
        # Create heatmap comparison
        self._plot_heatmap_comparison(results, param_configs)
    
    def _plot_combined_overview(self, results, param_configs):
        """Create combined overview chart with all metrics and parameters"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Complete Parameter Effects Overview', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        metrics = [
            ('cell_counts', 'Total Cell Count'),
            ('organ_counts', 'Organs Developed'),
            ('differentiation_speed', 'Differentiation Speed'),
            ('spatial_spread', 'Spatial Spread'),
            ('final_diversity', 'Cell Type Diversity')
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        markers = ['o', 's', '^', 'D', 'v']
        
        axes_flat = axes.flatten()
        
        for idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes_flat[idx]
            
            for param_idx, (param_name, data) in enumerate(results.items()):
                config = param_configs[param_name]
                
                # Normalize x-axis to 0-1 for comparison
                normalized_x = np.linspace(0, 1, len(data['values']))
                
                ax.plot(normalized_x, data[metric_key], 
                       marker=markers[param_idx], linewidth=2, markersize=7,
                       color=colors[param_idx], label=config['label'],
                       alpha=0.8, markeredgewidth=1.5, markeredgecolor='white')
            
            ax.set_xlabel('Parameter Value (normalized)', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=10, fontweight='bold')
            ax.set_title(metric_label, fontsize=11, fontweight='bold', pad=8)
            ax.legend(fontsize=8, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#F8F9FA')
        
        # Hide the last subplot
        axes_flat[5].axis('off')
        
        plt.tight_layout()
        plt.savefig('chart_complete_overview.png', dpi=300, bbox_inches='tight')
        print("  Saved: chart_complete_overview.png")
        plt.close(fig)
    
    def _plot_heatmap_comparison(self, results, param_configs):
        """Create heatmap showing parameter sensitivities"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['cell_counts', 'organ_counts', 'differentiation_speed', 
                  'spatial_spread', 'final_diversity']
        metric_labels = ['Cell Count', 'Organs', 'Diff. Speed', 'Spatial Spread', 'Diversity']
        
        params = list(results.keys())
        param_labels = [param_configs[p]['label'] for p in params]
        
        # Create sensitivity matrix (range of values for each metric)
        sensitivity_matrix = np.zeros((len(params), len(metrics)))
        
        for i, param_name in enumerate(params):
            data = results[param_name]
            for j, metric in enumerate(metrics):
                values = np.array(data[metric])
                # Calculate sensitivity as coefficient of variation
                if values.mean() > 0:
                    sensitivity = values.std() / values.mean() * 100
                else:
                    sensitivity = 0
                sensitivity_matrix[i, j] = sensitivity
        
        # Create heatmap
        im = ax.imshow(sensitivity_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(params)))
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_yticklabels(param_labels, fontsize=11)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values to cells
        for i in range(len(params)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{sensitivity_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sensitivity (%)', rotation=270, labelpad=20, fontsize=11)
        
        ax.set_title('Parameter Sensitivity Heatmap\n(Coefficient of Variation)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig('chart_sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
        print("  Saved: chart_sensitivity_heatmap.png")
        plt.close(fig)


def run_all_tests():
    """Run complete test suite"""
    print("="*60)
    print("FETAL DEVELOPMENT SIMULATOR - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tester = SimulatorTester()
    
    # Initialize simulator
    sim = tester.test_initialization()
    
    if sim is None:
        print("\n❌ Critical: Could not initialize simulator. Aborting tests.")
        return False
    
    # Run all test suites
    tester.test_parameters(sim)
    tester.test_parameter_ranges(sim)
    tester.test_cell_types()
    tester.test_week_calculation(sim)
    tester.test_milestones()
    tester.test_stage_names()
    tester.test_morphogen_update(sim)
    tester.test_anatomical_template(sim)
    tester.test_volume_calculation(sim)
    tester.test_energy_calculation(sim)
    tester.test_differentiation(sim)
    tester.test_monte_carlo_step(sim)
    tester.test_reset_functionality(sim)
    tester.test_simulation_progression(sim)
    tester.test_organ_development(sim)
    tester.test_ui_elements(sim)
    tester.test_camera_controls(sim)
    
    # Print summary
    all_passed = tester.print_summary()
    
    # Generate parameter comparison charts
    print("\n" + "="*60)
    print("GENERATING PARAMETER COMPARISON CHARTS")
    print("="*60)
    tester.generate_parameter_comparison_charts()
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
