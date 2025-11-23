# combio2_fixed_weeked.py
# Updated to map iterations -> gestational weeks and to scale templates/differentiation
# to more realistic week-by-week milestones (based on Mayo Clinic / NHS summaries).
#
# NOTE: This remains a conceptual visualization, not a precise biological model.

try:
    import pygame
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "pygame is not installed. Activate your venv or run:\n  pip install pygame\n"
    ) from e

import numpy as np
import sys
from enum import IntEnum
from scipy.ndimage import gaussian_filter

pygame.init()
pygame.font.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GRID_SIZE = 150
DRAW_SIZE = 600
CELL_SIZE = max(1, DRAW_SIZE // GRID_SIZE)

class CellType(IntEnum):
    EMPTY = 0
    EMBRYO = 1
    HEAD = 2
    BRAIN = 3
    BODY = 4
    ARM_LEFT = 5
    ARM_RIGHT = 6
    LEG_LEFT = 7
    LEG_RIGHT = 8
    PLACENTA = 9
    UMBILICAL = 10

COLORS = {
    CellType.EMPTY: (15, 15, 30),
    CellType.EMBRYO: (255, 120, 150),
    CellType.HEAD: (255, 200, 150),
    CellType.BRAIN: (255, 150, 100),
    CellType.BODY: (255, 140, 160),
    CellType.ARM_LEFT: (200, 100, 130),
    CellType.ARM_RIGHT: (200, 100, 130),
    CellType.LEG_LEFT: (180, 90, 120),
    CellType.LEG_RIGHT: (180, 90, 120),
    CellType.PLACENTA: (120, 100, 200),
    CellType.UMBILICAL: (150, 130, 210)
}

STAGE_NAMES = {
    1: "Blastocyst Stage (Week 1-2)",
    2: "Embryonic Stage (Week 3-8)",
    3: "Early Fetal Stage (Week 9-12)",
    4: "Mid Fetal Stage (Week 13-24)"
}

class FetalDevelopmentSimulator:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Fetal Development - Morphogen-Based CPM")
        self.clock = pygame.time.Clock()

        # Simulation parameters (adjustable)
        self.params = {
            'temperature': 12.0,
            'adhesion': 10.0,
            'volume_constraint': 8.0,
            'growth_rate': 0.25,
            'differentiation_rate': 0.12,
            'morphogen_diffusion': 0.5
        }

        # State
        self.grid = None
        self.morphogen_head = None
        self.morphogen_body = None
        self.iteration = 0  # now measured in days
        self.current_stage = 1
        self.is_running = False
        self.show_settings = False

        # UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 36)

        self.sliders = self._create_sliders()
        self.active_slider = None

        self.reset_simulation()

    def _create_sliders(self):
        x_start = 650
        y_start = 250
        y_spacing = 70
        return [
            {'name': 'temperature','label': 'Temperature','min': 1.0,'max': 30.0,
             'rect': pygame.Rect(x_start, y_start, 300, 10),'description': 'Cell movement randomness'},
            {'name': 'adhesion','label': 'Adhesion','min': 1.0,'max': 30.0,
             'rect': pygame.Rect(x_start, y_start + y_spacing, 300, 10),'description': 'Cell-cell stickiness'},
            {'name': 'volume_constraint','label': 'Volume Constraint','min': 0.1,'max': 40.0,
             'rect': pygame.Rect(x_start, y_start + y_spacing * 2, 300, 10),'description': 'Cell size maintenance'},
            {'name': 'growth_rate','label': 'Growth Rate','min': 0.05,'max': 1.0,
             'rect': pygame.Rect(x_start, y_start + y_spacing * 3, 300, 10),'description': 'Development speed'},
            {'name': 'differentiation_rate','label': 'Differentiation Rate','min': 0.01,'max': 0.4,
             'rect': pygame.Rect(x_start, y_start + y_spacing * 4, 300, 10),'description': 'Cell specialization'},
            {'name': 'morphogen_diffusion','label': 'Morphogen Diffusion','min': 0.05,'max': 2.0,
             'rect': pygame.Rect(x_start, y_start + y_spacing * 5, 300, 10),'description': 'Pattern formation'}
        ]

    def init_grid(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        center = GRID_SIZE // 2

        # initial blastocyst (small blob)
        radius = 4
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i*i + j*j <= radius*radius:
                    yy = center + i
                    xx = center + j
                    if 0 <= yy < GRID_SIZE and 0 <= xx < GRID_SIZE:
                        grid[yy, xx] = int(CellType.EMBRYO)

        # placenta ring (outer ring), sparser early then stronger
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if GRID_SIZE * 0.32 < dist < GRID_SIZE * 0.42 and np.random.random() < 0.35:
                    grid[i, j] = int(CellType.PLACENTA)

        self.morphogen_head = np.zeros((GRID_SIZE, GRID_SIZE))
        self.morphogen_body = np.zeros((GRID_SIZE, GRID_SIZE))
        return grid

    def week(self):
        """Return gestational week (starts at 1)."""
        return max(1, self.iteration // 7 + 1)

    def update_morphogens(self):
        """Morphogen sources adjusted by week (head morphogen strong early)."""
        center = GRID_SIZE // 2
        w = self.week()

        # reset small sources each update (so diffusion/decay create gradient)
        # head morphogen: source near top of embryo (anterior)
        head_offset = max(3, 5 + int( (8 - min(w, 8)) * 0.8 ))  # stronger source early (organogenesis)
        head_source_y = center - head_offset
        x0 = max(0, center - 5)
        x1 = min(GRID_SIZE, center + 5)
        y0 = max(0, head_source_y - 2)
        y1 = min(GRID_SIZE, head_source_y + 2)
        self.morphogen_head[y0:y1, x0:x1] += 1.0 * (1.2 if w <= 8 else 0.7)

        # body axis: vertical stripe around center
        body_source_x = center
        y0 = max(0, center - 10)
        y1 = min(GRID_SIZE, center + 10)
        x0 = max(0, body_source_x - 2)
        x1 = min(GRID_SIZE, body_source_x + 2)
        self.morphogen_body[y0:y1, x0:x1] += 0.9

        # diffuse more when diffusion param larger; clamp sigma
        sigma = max(0.1, float(self.params['morphogen_diffusion']))
        self.morphogen_head = gaussian_filter(self.morphogen_head, sigma=sigma)
        self.morphogen_body = gaussian_filter(self.morphogen_body, sigma=sigma)

        # decay to avoid runaway accumulation
        self.morphogen_head *= 0.96
        self.morphogen_body *= 0.96

    def get_anatomical_template(self, week):
        """Return anatomical templates scaled to a given week."""
        center = GRID_SIZE // 2
        template = {}

        # Use weeks to scale sizes; these are conceptual
        if week <= 2:
            template['embryo'] = (center, center, 6 + week)  # small
        elif week <= 8:
            # organogenesis: head appears early and grows quickly
            growth = (week - 2) / (8 - 2)
            head_size = max(2, int(8 + 12 * growth))
            body_length = max(2, int(6 + 20 * growth))
            template['head'] = (center - 12 - int(8 * growth), center, head_size)
            template['body'] = (center + 4, center, body_length)
        elif week <= 12:
            # early fetal: brain region more defined, limb buds begin
            growth = (week - 8) / (12 - 8)
            template['head'] = (center - 20, center, 14)
            template['brain'] = (center - 20, center, 6 + int(4 * growth))
            template['body'] = (center - 2, center, 20 + int(8 * growth))
            # limb buds appear and grow from week ~9 onwards
            limb_growth = max(1, int(4 * growth))
            template['arm_left'] = (center + 6, center - 8, 2 + limb_growth)
            template['arm_right'] = (center + 6, center + 8, 2 + limb_growth)
            template['leg_left'] = (center + 14, center - 6, 3 + limb_growth)
            template['leg_right'] = (center + 14, center + 6, 3 + limb_growth)
            template['umbilical'] = [(center + i, center) for i in range(18, 28)]
        else:
            # mid-fetal: growth, limbs more defined
            growth = min((week - 12) / (24 - 12), 1.0)
            template['head'] = (center - 26, center, 18)
            template['brain'] = (center - 26, center, 10)
            template['body'] = (center - 2, center, 28 + int(10 * growth))
            template['arm_left'] = (center + 2, center - 16, 10 + int(8 * growth))
            template['arm_right'] = (center + 2, center + 16, 10 + int(8 * growth))
            template['leg_left'] = (center + 24, center - 10, 12 + int(8 * growth))
            template['leg_right'] = (center + 24, center + 10, 12 + int(8 * growth))
            template['umbilical'] = [(center + i, center) for i in range(18, 38)]

            # placenta persists (already set in init), umbilical reaches to placenta edge
        return template

    def apply_morphogen_differentiation(self):
        """Differentiate cells using morphogen gradients + anatomical templates scaled by week."""
        w = self.week()

        # map week to stage (1..4)
        if w <= 2:
            self.current_stage = 1
        elif w <= 8:
            self.current_stage = 2
        elif w <= 12:
            self.current_stage = 3
        else:
            self.current_stage = 4

        template = self.get_anatomical_template(w)

        # differentiation rate scaled by clinical windows (organogenesis highest)
        base_diff = float(self.params['differentiation_rate'])
        if w <= 8:
            diff_rate = base_diff * 1.8
        elif w <= 12:
            diff_rate = base_diff * 1.2
        else:
            diff_rate = base_diff * 0.9

        # iterate grid and apply template-based changes
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # skip placenta
                if self.grid[i, j] == int(CellType.PLACENTA):
                    continue

                # only embryo/body/empty can differentiate into fetal types
                if self.grid[i, j] in (int(CellType.EMBRYO), int(CellType.BODY), int(CellType.HEAD), int(CellType.EMPTY)):
                    # head
                    if 'head' in template:
                        hy, hx, hr = template['head']
                        dist_head = np.hypot(i - hy, j - hx)
                        # head formation stronger when head morphogen high
                        morpho_head_level = self.morphogen_head[i, j]
                        prob = diff_rate * (1.0 + morpho_head_level)
                        if dist_head < hr and np.random.random() < prob:
                            self.grid[i, j] = int(CellType.HEAD)

                    # brain inside head (early)
                    if 'brain' in template and self.grid[i, j] == int(CellType.HEAD):
                        by, bx, br = template['brain']
                        dist_brain = np.hypot(i - by, j - bx)
                        if dist_brain < br and np.random.random() < diff_rate * 0.6:
                            self.grid[i, j] = int(CellType.BRAIN)

                    # body
                    if 'body' in template:
                        byy, bxx, br = template['body']
                        dist_b = np.hypot(i - byy, j - bxx)
                        if dist_b < br and np.random.random() < diff_rate * 0.9:
                            if self.grid[i, j] not in (int(CellType.HEAD), int(CellType.BRAIN)):
                                self.grid[i, j] = int(CellType.BODY)

                    # limbs
                    limbs = ['arm_left', 'arm_right', 'leg_left', 'leg_right']
                    limb_types = [CellType.ARM_LEFT, CellType.ARM_RIGHT, CellType.LEG_LEFT, CellType.LEG_RIGHT]
                    for limb_name, limb_type in zip(limbs, limb_types):
                        if limb_name in template:
                            ly, lx, lr = template[limb_name]
                            # roughly elongated distance function to create bud/limb shapes
                            if 'arm' in limb_name:
                                dist_limb = abs(i - ly) + 0.6 * abs(j - lx)
                            else:
                                dist_limb = 0.6 * abs(i - ly) + abs(j - lx)
                            if dist_limb < lr and np.random.random() < diff_rate * 0.6:
                                self.grid[i, j] = int(limb_type)

        # umbilical cord: populate positional list from template
        if 'umbilical' in template:
            for uy, ux in template['umbilical']:
                if 0 <= uy < GRID_SIZE and 0 <= ux < GRID_SIZE:
                    # umbilical can overwrite empty cells or embryo area with certain prob
                    if self.grid[uy, ux] in (int(CellType.EMPTY), int(CellType.EMBRYO)) and np.random.random() < 0.35:
                        self.grid[uy, ux] = int(CellType.UMBILICAL)

    def calculate_volumes(self):
        volumes = {}
        unique, counts = np.unique(self.grid, return_counts=True)
        for cell_type, count in zip(unique, counts):
            if cell_type != int(CellType.EMPTY):
                volumes[int(cell_type)] = int(count)
        return volumes

    def calculate_energy(self, x, y, new_type, volumes):
        current_type = int(self.grid[x, y])
        energy = 0.0

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbor_type = int(self.grid[nx, ny])
                adhesion_strength = float(self.params['adhesion'])
                if current_type == neighbor_type:
                    adhesion_strength *= 1.5
                if new_type == neighbor_type:
                    adhesion_strength *= 1.5
                if current_type != neighbor_type:
                    energy += adhesion_strength
                if new_type != neighbor_type:
                    energy -= adhesion_strength

        # volume constraint uses week-dependent target (grows with gestational age)
        if current_type != int(CellType.EMPTY):
            current_volume = volumes.get(int(current_type), 0)
            week = self.week()
            target_volume = 80 + week * (float(self.params['growth_rate']) * 10.0)  # conceptual scaling
            energy += float(self.params['volume_constraint']) * (current_volume - target_volume) ** 2 / 1000.0

        return energy

    def monte_carlo_step(self):
        """Perform one MC step (copy attempts) to let cells rearrange."""
        volumes = self.calculate_volumes()
        attempts = GRID_SIZE * 6
        for _ in range(attempts):
            x = np.random.randint(0, GRID_SIZE)
            y = np.random.randint(0, GRID_SIZE)
            dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][np.random.randint(4)]
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                current_type = int(self.grid[x, y])
                new_type = int(self.grid[nx, ny])

                # keep placenta and fetus mostly separate
                if (current_type == int(CellType.PLACENTA) and new_type != int(CellType.EMPTY)) or \
                   (new_type == int(CellType.PLACENTA) and current_type != int(CellType.EMPTY)):
                    continue

                if current_type != new_type:
                    energy_diff = self.calculate_energy(x, y, new_type, volumes)
                    temp = max(1e-6, float(self.params['temperature']))
                    if energy_diff < 0 or np.random.random() < np.exp(-energy_diff / temp):
                        self.grid[x, y] = new_type

    def reset_simulation(self):
        self.grid = self.init_grid()
        self.iteration = 0  # days
        self.current_stage = 1
        self.is_running = False

    # --- Drawing / UI functions below mostly unchanged ---
    def draw_grid(self):
        grid_surface = pygame.Surface((DRAW_SIZE, DRAW_SIZE))
        grid_surface.fill(COLORS[CellType.EMPTY])
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell_val = int(self.grid[i, j])
                try:
                    color = COLORS[CellType(cell_val)]
                except ValueError:
                    color = COLORS[CellType.EMPTY]
                rect = (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(grid_surface, color, rect)
        self.screen.blit(grid_surface, (20, 100))
        pygame.draw.circle(self.screen, (138, 43, 226), (320, 400), 280, 3)

    def draw_ui(self):
        for i in range(WINDOW_HEIGHT):
            color_val = int(15 + 15 * np.sin(i / WINDOW_HEIGHT * np.pi))
            pygame.draw.line(self.screen, (color_val, color_val, color_val + 15), (0, i), (WINDOW_WIDTH, i))
        title = self.title_font.render("Fetal Development Simulator", True, (255, 255, 255))
        subtitle = self.small_font.render("Morphogen-Based Cellular Potts Model (conceptual)", True, (200, 150, 255))
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 20))
        self.screen.blit(subtitle, (WINDOW_WIDTH // 2 - subtitle.get_width() // 2, 60))
        self.draw_grid()
        self.draw_buttons()
        self.draw_info_panel()
        self.draw_legend()
        if self.show_settings:
            self.draw_settings()

    def draw_buttons(self):
        button_y = 720
        start_color = (138, 43, 226) if not self.is_running else (100, 100, 100)
        pygame.draw.rect(self.screen, start_color, (150, button_y, 120, 40), border_radius=10)
        pygame.draw.rect(self.screen, (200, 150, 255), (150, button_y, 120, 40), 2, border_radius=10)
        start_text = self.font.render("Pause" if self.is_running else "Start", True, (255, 255, 255))
        self.screen.blit(start_text, (180, button_y + 10))

        pygame.draw.rect(self.screen, (70, 70, 90), (290, button_y, 120, 40), border_radius=10)
        pygame.draw.rect(self.screen, (150, 150, 180), (290, button_y, 120, 40), 2, border_radius=10)
        reset_text = self.font.render("Reset", True, (255, 255, 255))
        self.screen.blit(reset_text, (320, button_y + 10))

        settings_color = (138, 43, 226) if self.show_settings else (70, 70, 90)
        pygame.draw.rect(self.screen, settings_color, (430, button_y, 120, 40), border_radius=10)
        border_color = (200, 150, 255) if self.show_settings else (150, 150, 180)
        pygame.draw.rect(self.screen, border_color, (430, button_y, 120, 40), 2, border_radius=10)
        settings_text = self.font.render("Settings", True, (255, 255, 255))
        self.screen.blit(settings_text, (450, button_y + 10))

    def draw_info_panel(self):
        panel_x, panel_y = 650, 100
        pygame.draw.rect(self.screen, (60, 40, 100), (panel_x, panel_y, 520, 120), border_radius=10)
        pygame.draw.rect(self.screen, (138, 43, 226), (panel_x, panel_y, 520, 120), 3, border_radius=10)

        stage_title = self.font.render("Current Stage", True, (255, 255, 255))
        week_text = self.font.render(f"Week: {self.week()}", True, (255, 220, 180))
        stage_num = self.title_font.render(f"Stage {self.current_stage}", True, (255, 200, 150))
        stage_name = self.small_font.render(STAGE_NAMES[self.current_stage], True, (220, 180, 255))
        iteration_text = self.small_font.render(f"Day (iter): {self.iteration}", True, (200, 150, 255))

        self.screen.blit(stage_title, (panel_x + 20, panel_y + 15))
        self.screen.blit(stage_num, (panel_x + 20, panel_y + 45))
        self.screen.blit(stage_name, (panel_x + 20, panel_y + 80))
        self.screen.blit(week_text, (panel_x + 240, panel_y + 15))
        self.screen.blit(iteration_text, (panel_x + 240, panel_y + 45))

    def draw_legend(self):
        legend_x, legend_y = 650, 580
        pygame.draw.rect(self.screen, (40, 40, 60), (legend_x, legend_y, 520, 120), border_radius=10)
        pygame.draw.rect(self.screen, (138, 43, 226), (legend_x, legend_y, 520, 120), 2, border_radius=10)

        legend_title = self.font.render("Cell Types", True, (255, 255, 255))
        self.screen.blit(legend_title, (legend_x + 20, legend_y + 10))

        cell_info = [("Head", CellType.HEAD), ("Brain", CellType.BRAIN), ("Body", CellType.BODY),
                     ("Arms", CellType.ARM_LEFT), ("Legs", CellType.LEG_LEFT), ("Placenta", CellType.PLACENTA)]
        for i, (name, cell_type) in enumerate(cell_info):
            x = legend_x + 20 + (i % 3) * 170
            y = legend_y + 45 + (i // 3) * 35
            pygame.draw.rect(self.screen, COLORS[cell_type], (x, y, 25, 25), border_radius=5)
            pygame.draw.rect(self.screen, (180, 180, 200), (x, y, 25, 25), 1, border_radius=5)
            name_text = self.small_font.render(name, True, (220, 200, 255))
            self.screen.blit(name_text, (x + 35, y + 5))

    def draw_settings(self):
        for slider in self.sliders:
            label = self.font.render(slider['label'], True, (255, 255, 255))
            self.screen.blit(label, (slider['rect'].x, slider['rect'].y - 25))
            value = float(self.params[slider['name']])
            value_text = self.small_font.render(f"{value:.2f}", True, (200, 150, 255))
            self.screen.blit(value_text, (slider['rect'].x + slider['rect'].width + 10, slider['rect'].y - 5))
            pygame.draw.rect(self.screen, (70, 70, 90), slider['rect'], border_radius=5)
            normalized = (value - slider['min']) / max(1e-6, (slider['max'] - slider['min']))
            handle_x = int(slider['rect'].x + normalized * slider['rect'].width)
            pygame.draw.circle(self.screen, (138, 43, 226), (handle_x, slider['rect'].y + 5), 12)
            pygame.draw.circle(self.screen, (200, 150, 255), (handle_x, slider['rect'].y + 5), 12, 2)
            desc = self.small_font.render(slider['description'], True, (180, 160, 220))
            self.screen.blit(desc, (slider['rect'].x, slider['rect'].y + 15))

    def handle_click(self, pos):
        x, y = pos
        if 720 <= y <= 760:
            if 150 <= x <= 270:
                self.is_running = not self.is_running
            elif 290 <= x <= 410:
                self.reset_simulation()
            elif 430 <= x <= 550:
                self.show_settings = not self.show_settings
        if self.show_settings:
            for slider in self.sliders:
                if slider['rect'].collidepoint(pos):
                    self.active_slider = slider
                    self.update_slider(pos)

    def update_slider(self, pos):
        if self.active_slider:
            slider = self.active_slider
            x = max(slider['rect'].x, min(pos[0], slider['rect'].x + slider['rect'].width))
            normalized = (x - slider['rect'].x) / max(1, slider['rect'].width)
            value = slider['min'] + normalized * (slider['max'] - slider['min'])
            self.params[slider['name']] = value

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.active_slider = None
                elif event.type == pygame.MOUSEMOTION and self.active_slider:
                    self.update_slider(event.pos)

            if self.is_running:
                # Stop simulation at week 40
                if self.week() >= 40:
                    self.is_running = False
                else:
                    # update sequence
                    self.update_morphogens()
                    self.monte_carlo_step()
                    self.apply_morphogen_differentiation()
                    # advance one day per update; you can accelerate by changing increment
                    self.iteration += 1

            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    simulator = FetalDevelopmentSimulator()
    simulator.run()
