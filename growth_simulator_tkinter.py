import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import math

# =========================================
# OFF-LATTICE AGENT-BASED GROWTH ALGORITHM
# =========================================

# Color mapping for body parts (matching combio2.py)
BODY_PART_COLORS = {
    'embryo': '#ff7896',      # (255, 120, 150) - Pink
    'head': '#ffc896',        # (255, 200, 150) - Light orange/peach
    'brain': '#ff9664',       # (255, 150, 100) - Orange
    'body': '#ff8ca0',        # (255, 140, 160) - Pink
    'arm_left': '#c86482',    # (200, 100, 130) - Darker pink
    'arm_right': '#c86482',   # (200, 100, 130) - Darker pink
    'leg_left': '#b45a78',    # (180, 90, 120) - Even darker pink
    'leg_right': '#b45a78',   # (180, 90, 120) - Even darker pink
    'placenta': '#7864c8'     # (120, 100, 200) - Purple
}

class Cell:
    """Individual cell agent in off-lattice space."""
    def __init__(self, x, y, cell_type='germinal', generation=0):
        self.x = x
        self.y = y
        self.vx = 0.0  # velocity
        self.vy = 0.0
        self.radius = 0.08  # Smaller cell radius for more detailed shape
        self.age = 0.0
        self.division_timer = random.uniform(8, 18)  # Much faster division
        self.cell_type = cell_type  # 'germinal', 'embryonic', 'fetal'
        self.generation = generation
        self.max_divisions = 25 - generation  # allow more divisions
        self.divisions = 0
        self.neighbors = []
        self.body_part = 'embryo'  # 'head', 'brain', 'body', 'arm_left', 'arm_right', 'leg_left', 'leg_right', 'placenta', 'embryo'
        
    def update_age(self, dt):
        """Update cell age."""
        self.age += dt
        
    def can_divide(self):
        """Check if cell can divide."""
        return (self.age >= self.division_timer and 
                self.divisions < self.max_divisions and
                len(self.neighbors) < 12)  # Allow more crowding for smoother outline with smaller cells
    
    def divide(self):
        """Cell division: create daughter cell."""
        # Random direction for division
        angle = random.uniform(0, 2 * np.pi)
        separation = (self.radius * 2.0)  # Tighter separation for smaller cells
        
        # Daughter cell position
        dx = np.cos(angle) * separation
        dy = np.sin(angle) * separation
        
        daughter = Cell(
            self.x + dx,
            self.y + dy,
            self.cell_type,
            self.generation + 1
        )
        
        # Reset division timer
        self.division_timer = random.uniform(8, 18)
        self.age = 0.0
        self.divisions += 1
        
        return daughter
    
    def distance_to(self, other):
        """Calculate distance to another cell."""
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx*dx + dy*dy)
    
    def determine_body_part(self, stage):
        """Determine body part based on generation and position."""
        if stage == "Germinal":
            # Early stage - mostly embryo, some placenta
            if self.generation == 0:
                self.body_part = 'embryo'
            else:
                # Randomly assign some cells as placenta in early stages
                if random.random() < 0.1:
                    self.body_part = 'placenta'
                else:
                    self.body_part = 'embryo'
        
        elif stage == "Embryonic":
            # Head region (generations 0-3)
            if self.generation < 4:
                # Brain is in the center of head
                center_dist = np.sqrt(self.x**2 + (self.y - 1.8)**2)
                if center_dist < 0.3:
                    self.body_part = 'brain'
                else:
                    self.body_part = 'head'
            # Body region (generations 4+)
            else:
                self.body_part = 'body'
        
        elif stage == "Fetal":
            # Head region (generations 0-7)
            if self.generation < 8:
                # Brain is in the center of head
                center_dist = np.sqrt(self.x**2 + (self.y - 2.3)**2)
                if center_dist < 0.4:
                    self.body_part = 'brain'
                else:
                    self.body_part = 'head'
            # Neck/upper chest (generations 8-10)
            elif self.generation < 11:
                self.body_part = 'body'
            # Body with limbs (generations 11+)
            else:
                body_index = self.generation - 11
                # Arms (upper body region)
                if body_index > 3 and body_index < 11:
                    if body_index % 2 == 0:
                        self.body_part = 'arm_left'
                    else:
                        self.body_part = 'arm_right'
                # Legs (lower body region)
                elif body_index > 11:
                    if body_index % 2 == 0:
                        self.body_part = 'leg_left'
                    else:
                        self.body_part = 'leg_right'
                # Main body
                else:
                    self.body_part = 'body'
    
    def update_forces(self, cells, dt, stage="Germinal"):
        """Update forces from neighboring cells (off-lattice mechanics)."""
        fx, fy = 0.0, 0.0
        
        # Find neighbors within interaction range
        self.neighbors = []
        interaction_range = self.radius * 4
        
        for other in cells:
            if other is self:
                continue
                
            dist = self.distance_to(other)
            
            if dist < interaction_range:
                self.neighbors.append(other)
                
                # Repulsion force (prevent overlap) - stronger to spread cells out
                if dist < (self.radius + other.radius):
                    overlap = (self.radius + other.radius) - dist
                    if overlap > 0:
                        # Strong repulsion
                        direction = np.array([self.x - other.x, self.y - other.y])
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction = direction / norm
                            force_mag = overlap * 0.8  # Increased repulsion
                            fx += direction[0] * force_mag
                            fy += direction[1] * force_mag
                
                # Adhesion force (keep structure together) - much weaker to prevent clustering
                elif dist < interaction_range and dist > (self.radius + other.radius) * 1.5:
                    # Only weak adhesion at medium distances, not close neighbors
                    direction = np.array([other.x - self.x, other.y - self.y])
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        # Very weak attraction - reduced significantly
                        force_mag = 0.003 / (dist * dist + 0.2)  # Much weaker
                        fx += direction[0] * force_mag
                        fy += direction[1] * force_mag
        
        # Morphogenetic forces based on stage - stronger to form proper shapes
        if stage == "Germinal":
            # Circular/spherical cluster - attract to center (like morula/blastocyst)
            center_dist = np.sqrt(self.x**2 + self.y**2)
            if center_dist > 0.3:
                center_dir = np.array([-self.x, -self.y]) / center_dist
                # Stronger force to form tight circular cluster
                fx += center_dir[0] * 0.05
                fy += center_dir[1] * 0.05
            # Also add slight repulsion from center if too close to spread out
            elif center_dist < 0.2:
                center_dir = np.array([self.x, self.y]) / (center_dist + 0.01)
                fx += center_dir[0] * 0.01
                fy += center_dir[1] * 0.01
        
        elif stage == "Embryonic":
            # Embryonic stage: C-shaped embryo with distinct head and body
            # Reference shows clear C-shape with head at top, body curving down
            
            n_cells = len(cells)
            if n_cells < 10:
                # Very early embryonic - still forming, slight clustering
                center_x, center_y = 0.0, 0.3
                dx = center_x - self.x
                dy = center_y - self.y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.4:
                    fx += dx * 0.03 / (dist + 0.1)
                    fy += dy * 0.03 / (dist + 0.1)
            else:
                # Later embryonic - form clear C-shape
                # Head region (top, distinct)
                if self.generation < 4:
                    target_x, target_y = 0.0, 1.8
                    # Spread head cells in a pattern
                    head_angle = (self.generation % 4) * (2 * np.pi / 4)
                    head_radius = 0.4
                    target_x = target_x + head_radius * np.cos(head_angle)
                    target_y = target_y + head_radius * np.sin(head_angle)
                    dx = target_x - self.x
                    dy = target_y - self.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0.1:
                        fx += dx * 0.04 / (dist + 0.1)  # Stronger force
                        fy += dy * 0.04 / (dist + 0.1)
                else:  # Body in C-shape - spread out along curve
                    cell_index = self.generation - 4
                    angle = cell_index * 0.25  # Progressive angle for C-shape
                    # C-shape: start from head, curve down and around
                    curve_radius = 1.4
                    start_angle = np.pi/2 + 0.1
                    target_x = curve_radius * np.cos(angle + start_angle)
                    target_y = 1.5 - curve_radius * np.sin(angle + start_angle)
                    dx = target_x - self.x
                    dy = target_y - self.y
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0.1:
                        fx += dx * 0.035 / (dist + 0.1)  # Stronger to spread along curve
                        fy += dy * 0.035 / (dist + 0.1)
        
        elif stage == "Fetal":
            # Fetal/Baby shape: Based on reference - pronounced C-shape, large head, curled position
            # Reference shows: head at top, body curves down and around in tight C-shape
            # Cells should spread out to form the shape, not cluster
            
            # Head region (top, large - prominent in fetal stage)
            if self.generation < 8:
                # Create a large head region at the top center - spread cells out
                head_center_x, head_center_y = 0.0, 2.3
                # Distribute cells in a circular/oval head pattern - spread out
                head_ring = self.generation // 2  # Which ring of head cells
                head_position = self.generation % 2  # Position in ring
                
                if head_ring == 0:
                    # Core head cells - center
                    head_radius = 0.3
                    head_angle = head_position * np.pi
                elif head_ring == 1:
                    # Second ring
                    head_radius = 0.5
                    head_angle = head_position * np.pi
                elif head_ring == 2:
                    # Third ring
                    head_radius = 0.7
                    head_angle = (head_position % 3) * (2 * np.pi / 3)
                else:
                    # Outer head cells - spread around circumference
                    head_radius = 0.9 + (head_ring % 2) * 0.15
                    head_angle = (head_position % 4) * (2 * np.pi / 4)
                
                target_x = head_center_x + head_radius * np.cos(head_angle)
                target_y = head_center_y + head_radius * np.sin(head_angle)
                dx = target_x - self.x
                dy = target_y - self.y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.1:
                    fx += dx * 0.05 / (dist + 0.1)  # Stronger force to spread head
                    fy += dy * 0.05 / (dist + 0.1)
            
            # Neck/upper chest region (generations 8-10) - connects head to body
            elif self.generation < 11:
                neck_x, neck_y = 0.0, 1.4
                # Spread neck cells slightly
                neck_spread = (self.generation - 8) * 0.15
                target_x = neck_x + neck_spread * np.cos((self.generation % 2) * np.pi)
                target_y = neck_y
                dx = target_x - self.x
                dy = target_y - self.y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.1:
                    fx += dx * 0.04 / (dist + 0.1)
                    fy += dy * 0.04 / (dist + 0.1)
            
            # Body region - tight C-shape (generations 11+)
            # Reference shows very curled up position - spread cells along curve
            else:
                body_index = self.generation - 11
                # Create a tight C-shape: curve from upper body down and around
                angle = body_index * 0.2  # Progressive angle for C-shape
                # C-shape: start from upper body, curve down and around
                curve_radius = 1.5
                start_angle = np.pi/2 + 0.12  # Adjusted for fetal position
                target_x = curve_radius * np.cos(angle + start_angle)
                target_y = 1.1 - curve_radius * np.sin(angle + start_angle)
                dx = target_x - self.x
                dy = target_y - self.y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.1:
                    fx += dx * 0.04 / (dist + 0.1)  # Stronger to spread along curve
                    fy += dy * 0.04 / (dist + 0.1)
                
                # Add limb-like protrusions for body cells (arms and legs)
                # Reference shows limbs developing along the C-curve
                if body_index > 3 and body_index < 20:
                    # Arms (upper body region)
                    if body_index > 3 and body_index < 11:
                        if body_index % 2 == 0:  # Left arm
                            limb_x = target_x - 0.7
                            limb_y = target_y
                            dx_limb = limb_x - self.x
                            dy_limb = limb_y - self.y
                            dist_limb = np.sqrt(dx_limb*dx_limb + dy_limb*dy_limb)
                            if dist_limb < 1.0:
                                fx += dx_limb * 0.015 / (dist_limb + 0.1)  # Stronger limb force
                                fy += dy_limb * 0.015 / (dist_limb + 0.1)
                        else:  # Right arm
                            limb_x = target_x + 0.7
                            limb_y = target_y
                            dx_limb = limb_x - self.x
                            dy_limb = limb_y - self.y
                            dist_limb = np.sqrt(dx_limb*dx_limb + dy_limb*dy_limb)
                            if dist_limb < 1.0:
                                fx += dx_limb * 0.015 / (dist_limb + 0.1)
                                fy += dy_limb * 0.015 / (dist_limb + 0.1)
                    # Legs (lower body region)
                    elif body_index > 11:
                        if body_index % 2 == 0:  # Left leg
                            limb_x = target_x - 0.6
                            limb_y = target_y
                            dx_limb = limb_x - self.x
                            dy_limb = limb_y - self.y
                            dist_limb = np.sqrt(dx_limb*dx_limb + dy_limb*dy_limb)
                            if dist_limb < 0.9:
                                fx += dx_limb * 0.012 / (dist_limb + 0.1)
                                fy += dy_limb * 0.012 / (dist_limb + 0.1)
                        else:  # Right leg
                            limb_x = target_x + 0.6
                            limb_y = target_y
                            dx_limb = limb_x - self.x
                            dy_limb = limb_y - self.y
                            dist_limb = np.sqrt(dx_limb*dx_limb + dy_limb*dy_limb)
                            if dist_limb < 0.9:
                                fx += dx_limb * 0.012 / (dist_limb + 0.1)
                                fy += dy_limb * 0.012 / (dist_limb + 0.1)
        
        # Update velocity (with less damping for better movement)
        self.vx = self.vx * 0.85 + fx * dt  # Less damping
        self.vy = self.vy * 0.85 + fy * dt
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Boundary conditions (soft boundary) - larger for Fetal stage
        if stage == "Fetal":
            boundary = 4.5  # Larger boundary for fetal shape with large head
        elif stage == "Embryonic":
            boundary = 3.0  # Medium for embryonic
        else:
            boundary = 2.5  # Smaller for germinal
        if abs(self.x) > boundary:
            self.vx *= -0.5
            self.x = np.sign(self.x) * boundary
        if abs(self.y) > boundary:
            self.vy *= -0.5
            self.y = np.sign(self.y) * boundary


class GrowthSimulation:
    """Off-lattice agent-based growth simulation."""
    def __init__(self):
        # Start with single zygote cell
        initial_cell = Cell(0.0, 0.0, 'germinal', 0)
        initial_cell.body_part = 'embryo'
        self.cells = [initial_cell]
        self.time = 0.0  # Simulation time
        self.dt = 0.1
        self.stage = "Germinal"
        self.frame_count = 0
        self.weeks = 0.0
        self.months = 0.0
        
    def update_stage(self):
        """Update developmental stage based on cell count and time."""
        n = len(self.cells)
        
        # Convert simulation time to weeks
        # Scale: 1 simulation unit = 0.15 weeks (adjusted for proper progression)
        self.weeks = self.time * 0.15
        self.months = self.weeks / 4.33  # Average weeks per month
        
        # Stage thresholds based on cell count
        if n < 6:
            self.stage = "Germinal"
        elif n < 25:
            self.stage = "Embryonic"
        else:
            self.stage = "Fetal"
        
        # Update cell types based on stage
        for cell in self.cells:
            cell.cell_type = self.stage.lower()
    
    def get_stage_info(self):
        """Get stage information with time ranges."""
        if self.stage == "Germinal":
            return "Germinal (0-2 weeks)"
        elif self.stage == "Embryonic":
            return "Embryonic (3-8 weeks)"
        else:
            return "Fetal (9+ weeks)"
    
    def step(self):
        """Perform one simulation step."""
        self.time += self.dt
        
        # Update stage first to get current stage
        self.update_stage()
        current_stage = self.stage
        
        # Adjust division timer based on stage (faster in early stages, but still fast in all)
        stage_division_multiplier = {
            "Germinal": 0.7,  # Very fast division
            "Embryonic": 0.85,  # Fast division
            "Fetal": 1.0  # Normal (but still fast due to lower base timer)
        }
        
        # Update all cells
        new_cells = []
        for cell in self.cells:
            # Update age (with stage-based rate - faster aging = faster division)
            cell.update_age(self.dt * stage_division_multiplier.get(current_stage, 1.0))
            
            # Update forces (off-lattice mechanics with morphogenetic guidance)
            cell.update_forces(self.cells, self.dt, current_stage)
            
            # Determine body part based on position and generation
            cell.determine_body_part(current_stage)
            
            # Check for division
            if cell.can_divide():
                daughter = cell.divide()
                # Set daughter cell type
                daughter.cell_type = current_stage.lower()
                # Determine body part for daughter cell
                daughter.determine_body_part(current_stage)
                new_cells.append(daughter)
        
        # Add new cells
        self.cells.extend(new_cells)
        
        # Update stage again after adding cells
        self.update_stage()
        
        self.frame_count += 1
    
    def reset(self):
        """Reset simulation to initial state."""
        initial_cell = Cell(0.0, 0.0, 'germinal', 0)
        initial_cell.body_part = 'embryo'
        self.cells = [initial_cell]
        self.time = 0.0
        self.stage = "Germinal"
        self.frame_count = 0


class GrowthSimulatorGUI:
    """Tkinter GUI for the growth simulator."""
    def __init__(self, root):
        self.root = root
        self.root.title("🧬 Off-Lattice Agent-Based Growth Simulator")
        self.root.geometry("1400x1000")
        self.root.configure(bg='#f9fafb')
        
        # Initialize simulation
        self.sim = GrowthSimulation()
        self.running = False
        self.max_cells = 300  # Increased for smaller cells to create more detailed shape
        self.speed = 5
        self.steps_per_frame = 5  # Increased for faster progression
        
        # Create main container with scrollbars
        self.create_scrollable_frame()
        
        # Setup UI
        self.setup_ui()
        
        # Start update loop
        self.update()
    
    def create_scrollable_frame(self):
        """Create a scrollable frame for the entire window."""
        # Create a frame to hold canvas and scrollbars
        canvas_container = tk.Frame(self.root, bg='#f9fafb')
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Create main canvas with scrollbars
        self.main_canvas = tk.Canvas(canvas_container, bg='#f9fafb', highlightthickness=0)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create vertical scrollbar
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.main_canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create horizontal scrollbar frame
        h_scrollbar_frame = tk.Frame(self.root, bg='#f9fafb')
        h_scrollbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        h_scrollbar = ttk.Scrollbar(h_scrollbar_frame, orient=tk.HORIZONTAL, command=self.main_canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure canvas scrolling
        self.main_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Create scrollable frame
        self.scrollable_frame = tk.Frame(self.main_canvas, bg='#f9fafb')
        self.scrollable_frame_id = self.main_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor=tk.NW
        )
        
        # Bind events for scrolling
        self.scrollable_frame.bind('<Configure>', self.on_frame_configure)
        self.main_canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Bind mousewheel (Windows and Mac)
        self.main_canvas.bind_all('<MouseWheel>', self.on_mousewheel)
        # Bind mousewheel (Linux)
        self.main_canvas.bind_all('<Button-4>', self.on_mousewheel)
        self.main_canvas.bind_all('<Button-5>', self.on_mousewheel)
    
    def on_frame_configure(self, event):
        """Update scroll region when frame size changes."""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all'))
    
    def on_canvas_configure(self, event):
        """Update scrollable frame width when canvas size changes."""
        canvas_width = event.width
        self.main_canvas.itemconfig(self.scrollable_frame_id, width=canvas_width)
    
    def on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        # Windows and Mac
        if hasattr(event, 'delta'):
            if event.delta > 0:
                self.main_canvas.yview_scroll(-1, 'units')
            elif event.delta < 0:
                self.main_canvas.yview_scroll(1, 'units')
        # Linux
        elif event.num == 4:
            self.main_canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self.main_canvas.yview_scroll(1, 'units')
    
    def setup_ui(self):
        """Setup the user interface."""
        # Title
        title_frame = tk.Frame(self.scrollable_frame, bg='#f9fafb')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🧬 Off-Lattice Agent-Based Growth Simulator",
            font=('Consolas', 16, 'bold'),
            fg='#2a4d7a',
            bg='#f9fafb'
        )
        title_label.pack()
        
        desc_label = tk.Label(
            title_frame,
            text="This simulation uses an off-lattice agent-based algorithm to model development from zygote to baby.",
            font=('Consolas', 10),
            fg='#222',
            bg='#f9fafb',
            wraplength=800
        )
        desc_label.pack(pady=5)
        
        color_info_label = tk.Label(
            title_frame,
            text="Color Coding: Head (#ffc896) | Brain (#ff9664) | Body (#ff8ca0) | Arms (#c86482) | Legs (#b45a78) | Placenta (#7864c8) | Embryo (#ff7896)",
            font=('Consolas', 9),
            fg='#444',
            bg='#f9fafb',
            wraplength=900
        )
        color_info_label.pack(pady=2)
        
        # Control panel
        control_frame = tk.Frame(self.scrollable_frame, bg='#f9fafb')
        control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶️ Start Simulation",
            command=self.start_simulation,
            bg='#2a4d7a',
            fg='white',
            font=('Consolas', 10),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor='hand2'
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(
            control_frame,
            text="⏸️ Pause",
            command=self.pause_simulation,
            bg='#666',
            fg='white',
            font=('Consolas', 10),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(
            control_frame,
            text="🔄 Reset",
            command=self.reset_simulation,
            bg='#666',
            fg='white',
            font=('Consolas', 10),
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor='hand2'
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content area
        main_frame = tk.Frame(self.scrollable_frame, bg='#f9fafb')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for visualization
        canvas_frame = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=800,
            height=800,
            bg='#fafafa',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Statistics panel - make it wider and use scrollable frame
        stats_container = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2, width=400)
        stats_container.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        stats_container.pack_propagate(False)
        
        # Create scrollable frame for stats panel
        stats_canvas = tk.Canvas(stats_container, bg='white', highlightthickness=0)
        stats_scrollbar = ttk.Scrollbar(stats_container, orient=tk.VERTICAL, command=stats_canvas.yview)
        stats_scrollable_frame = tk.Frame(stats_canvas, bg='white')
        
        stats_scrollable_frame.bind(
            "<Configure>",
            lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))
        )
        
        stats_window = stats_canvas.create_window((0, 0), window=stats_scrollable_frame, anchor="nw")
        stats_canvas.configure(yscrollcommand=stats_scrollbar.set)
        
        def configure_stats_window(event):
            # Update the width of the scrollable frame to match canvas width
            canvas_width = event.width
            stats_canvas.itemconfig(stats_window, width=canvas_width)
        
        stats_canvas.bind('<Configure>', configure_stats_window)
        
        stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel to stats canvas when hovering
        def on_stats_mousewheel(event):
            if hasattr(event, 'delta'):
                if event.delta > 0:
                    stats_canvas.yview_scroll(-1, 'units')
                elif event.delta < 0:
                    stats_canvas.yview_scroll(1, 'units')
            elif event.num == 4:
                stats_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                stats_canvas.yview_scroll(1, 'units')
        
        stats_canvas.bind('<Enter>', lambda e: stats_canvas.focus_set())
        stats_canvas.bind('<MouseWheel>', on_stats_mousewheel)
        stats_canvas.bind('<Button-4>', on_stats_mousewheel)
        stats_canvas.bind('<Button-5>', on_stats_mousewheel)
        
        # Use stats_scrollable_frame instead of stats_frame for all content
        stats_frame = stats_scrollable_frame
        
        stats_title = tk.Label(
            stats_frame,
            text="Statistics",
            font=('Consolas', 12, 'bold'),
            fg='#2a4d7a',
            bg='white'
        )
        stats_title.pack(pady=10)
        
        self.stats_text = tk.Text(
            stats_frame,
            width=45,
            height=12,
            font=('Consolas', 9),
            bg='white',
            fg='#222',
            wrap=tk.WORD,
            relief=tk.FLAT
        )
        self.stats_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Parameters panel
        params_frame = tk.Frame(stats_frame, bg='white')
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        params_title = tk.Label(
            params_frame,
            text="Parameters",
            font=('Consolas', 11, 'bold'),
            fg='#2a4d7a',
            bg='white'
        )
        params_title.pack(anchor=tk.W, pady=(10, 5))
        
        # Max cells slider
        max_cells_label = tk.Label(
            params_frame,
            text="Max Cells:",
            font=('Consolas', 9),
            bg='white',
            fg='#222'
        )
        max_cells_label.pack(anchor=tk.W)
        
        self.max_cells_var = tk.IntVar(value=self.max_cells)
        max_cells_scale = tk.Scale(
            params_frame,
            from_=50,
            to=500,
            orient=tk.HORIZONTAL,
            variable=self.max_cells_var,
            font=('Consolas', 8),
            bg='white',
            fg='#222',
            command=self.update_max_cells
        )
        max_cells_scale.pack(fill=tk.X, pady=5)
        
        # Speed slider
        speed_label = tk.Label(
            params_frame,
            text="Speed:",
            font=('Consolas', 9),
            bg='white',
            fg='#222'
        )
        speed_label.pack(anchor=tk.W, pady=(10, 0))
        
        self.speed_var = tk.IntVar(value=self.speed)
        speed_scale = tk.Scale(
            params_frame,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            font=('Consolas', 8),
            bg='white',
            fg='#222',
            command=self.update_speed
        )
        speed_scale.pack(fill=tk.X, pady=5)
        
        # Algorithm info
        info_frame = tk.Frame(stats_frame, bg='white')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_title = tk.Label(
            info_frame,
            text="Algorithm Details",
            font=('Consolas', 11, 'bold'),
            fg='#2a4d7a',
            bg='white'
        )
        info_title.pack(anchor=tk.W, pady=(5, 3))
        
        info_text = """• Off-lattice: Cells exist in continuous 2D space (no grid)
• Agent-based: Each cell is an autonomous agent with position, velocity, and properties
• Growth rules: Cells divide based on age and crowding constraints
• Force-based mechanics: Repulsion prevents overlap, adhesion maintains structure
• Emergent behavior: Complex shapes emerge from simple local interactions"""
        
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=('Consolas', 8),
            bg='white',
            fg='#444',
            justify=tk.LEFT,
            anchor=tk.W
        )
        info_label.pack(anchor=tk.W)
        
        # Body part legend - make it more prominent and visible
        legend_frame = tk.Frame(stats_frame, bg='white')
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        legend_title = tk.Label(
            legend_frame,
            text="Body Parts & Colors",
            font=('Consolas', 11, 'bold'),
            fg='#2a4d7a',
            bg='white'
        )
        legend_title.pack(anchor=tk.W, pady=(5, 3))
        
        # Create legend items with descriptions - more compact
        body_parts = [
            ('Head', 'head', 'Head region'),
            ('Brain', 'brain', 'Brain tissue'),
            ('Body', 'body', 'Main body/trunk'),
            ('Arms', 'arm_left', 'Arm limbs'),
            ('Legs', 'leg_left', 'Leg limbs'),
            ('Placenta', 'placenta', 'Placental tissue'),
            ('Embryo', 'embryo', 'Early embryonic cells')
        ]
        
        for name, part_key, description in body_parts:
            part_frame = tk.Frame(legend_frame, bg='white')
            part_frame.pack(anchor=tk.W, pady=1)
            
            # Color box
            color_box = tk.Canvas(part_frame, width=18, height=18, bg='white', highlightthickness=0)
            color_box.pack(side=tk.LEFT, padx=(0, 5))
            color = BODY_PART_COLORS.get(part_key, '#ff7896')
            color_box.create_oval(1, 1, 17, 17, fill=color, outline='#333', width=1)
            
            # Label with color code and description - more compact
            label_text = f"{name} {color} - {description}"
            label = tk.Label(
                part_frame,
                text=label_text,
                font=('Consolas', 8),
                bg='white',
                fg='#444',
                anchor=tk.W
            )
            label.pack(side=tk.LEFT)
        
        # Update scroll region after UI is set up
        self.root.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all'))
    
    def update_max_cells(self, value):
        """Update max cells parameter."""
        self.max_cells = int(value)
    
    def update_speed(self, value):
        """Update speed parameter."""
        self.speed = int(value)
    
    def start_simulation(self):
        """Start the simulation."""
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
    
    def pause_simulation(self):
        """Pause the simulation."""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
    
    def reset_simulation(self):
        """Reset the simulation."""
        self.running = False
        self.sim.reset()
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
    
    def draw_simulation(self):
        """Draw the simulation on canvas."""
        self.canvas.delete("all")
        
        # Get canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Center and scale - adjust based on stage
        center_x = width / 2
        center_y = height / 2
        # Scale to fit the current stage appropriately
        if self.sim.stage == "Fetal":
            # Fetal stage needs more space for large head and C-shape
            scale = min(width, height) / 9  # Scale for -4.5 to 4.5 range
        elif self.sim.stage == "Embryonic":
            scale = min(width, height) / 7  # Scale for -3 to 3 range
        else:
            scale = min(width, height) / 6  # Scale for -2.5 to 2.5 range (Germinal)
        
        # Draw grid - adjust range based on stage
        if self.sim.stage == "Fetal":
            grid_range = range(-4, 5)  # -4 to 4
        elif self.sim.stage == "Embryonic":
            grid_range = range(-3, 4)  # -3 to 3
        else:
            grid_range = range(-2, 3)  # -2 to 2 (Germinal)
        
        for i in grid_range:
            x = center_x + i * scale
            self.canvas.create_line(x, 0, x, height, fill='#e0e0e0', width=1)
        for i in grid_range:
            y = center_y + i * scale
            self.canvas.create_line(0, y, width, y, fill='#e0e0e0', width=1)
        
        # Draw cells
        for cell in self.sim.cells:
            # Convert coordinates
            screen_x = center_x + cell.x * scale
            screen_y = center_y - cell.y * scale  # Flip Y axis
            
            # Determine color by body part (matching combio2.py)
            color = BODY_PART_COLORS.get(cell.body_part, '#ff7896')  # Default to embryo color
            
            # Draw cell - slightly larger for visibility but still small for detail
            radius_pixels = cell.radius * scale * 1.2  # Slight increase for visibility
            # Minimum size to ensure visibility
            radius_pixels = max(radius_pixels, 2.0)
            self.canvas.create_oval(
                screen_x - radius_pixels,
                screen_y - radius_pixels,
                screen_x + radius_pixels,
                screen_y + radius_pixels,
                fill=color,
                outline='darkblue',
                width=0.5  # Thinner outline for smaller cells
            )
        
        # Draw title with stage info and time
        stage_info = self.sim.get_stage_info()
        title_text = f"{stage_info} | Cells: {len(self.sim.cells)} | Weeks: {self.sim.weeks:.1f} | Months: {self.sim.months:.2f}"
        self.canvas.create_text(
            width / 2,
            20,
            text=title_text,
            font=('Consolas', 10, 'bold'),
            fill='#2a4d7a'
        )
    
    def update_stats(self):
        """Update statistics display."""
        if len(self.sim.cells) > 0:
            avg_gen = np.mean([c.generation for c in self.sim.cells])
            max_div = max([c.divisions for c in self.sim.cells])
            avg_neighbors = np.mean([len(c.neighbors) for c in self.sim.cells])
            stage_info = self.sim.get_stage_info()
            
            stats = f"""Current Stage: {stage_info}

Cell Count: {len(self.sim.cells)}

Development Time:
  Weeks: {self.sim.weeks:.1f}
  Months: {self.sim.months:.2f}

Average Generation: {avg_gen:.2f}

Max Divisions: {max_div}

Average Neighbors: {avg_neighbors:.2f}

Simulation Time: {self.sim.time:.2f}

Frame Count: {self.sim.frame_count}

Status: {'Running' if self.running else 'Paused'}
            """
        else:
            stats = "No cells in simulation."
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def update(self):
        """Main update loop."""
        if self.running and len(self.sim.cells) < self.max_cells:
            # Run simulation steps
            for _ in range(self.steps_per_frame):
                if len(self.sim.cells) < self.max_cells:
                    self.sim.step()
                else:
                    self.running = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.pause_btn.config(state=tk.DISABLED)
                    break
        
        # Update visualization
        self.draw_simulation()
        self.update_stats()
        
        # Schedule next update
        delay = int(100 / self.speed)  # Adjust delay based on speed
        self.root.after(delay, self.update)


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = GrowthSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

