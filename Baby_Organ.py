
import pygame
import numpy as np
import sys
from enum import IntEnum
from scipy.ndimage import gaussian_filter

pygame.init()
pygame.font.init()

# ---- Configurable constants ----
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 820
GRID_SIZE = 150
DRAW_SIZE = 600
CELL_SIZE = max(1, DRAW_SIZE // GRID_SIZE)

RNG = np.random.default_rng(42)

# ---- Cell types ----
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
    # Organs (unique, non-overlapping colors)
    HEART = 20
    LIVER = 21
    STOMACH = 22
    INTESTINE = 23
    KIDNEY = 24
    LUNG = 25
    EYE = 26
    BLADDER = 27

# ---- Colors: ensure organs are distinct from head/body/limbs/placenta ----
COLORS = {
    CellType.EMPTY: (12, 12, 28),
    CellType.EMBRYO: (255, 140, 160),
    CellType.HEAD: (255, 200, 160),
    CellType.BRAIN: (255, 160, 120),
    CellType.BODY: (255, 150, 170),
    CellType.ARM_LEFT: (220, 130, 150),
    CellType.ARM_RIGHT: (220, 130, 150),
    CellType.LEG_LEFT: (200, 110, 140),
    CellType.LEG_RIGHT: (200, 110, 140),
    CellType.PLACENTA: (120, 90, 180),
    CellType.UMBILICAL: (160, 130, 210),
    # Organs palette (unique)
    CellType.HEART: (255, 0, 0),          # red
    CellType.LIVER: (0, 0, 255),          # blue
    CellType.STOMACH: (255, 255, 255),    # white
    CellType.INTESTINE: (139, 69, 19),    # brown
    CellType.KIDNEY: (255, 255, 0),       # bright yellow
    CellType.LUNG: (255, 105, 180),       # bright pink
    CellType.EYE: (0, 0, 0),              # black
    CellType.BLADDER: (0, 0, 139)         # dark blue
}

# ---- Week-based milestones ----
MILESTONES = {
    range(1, 3): "Weeks 1–2: Fertilization & implantation window.",
    range(3, 6): "Weeks 3–5: Embryonic disk, early heart tube, neural tube closure begins.",
    range(6, 9): "Weeks 6–8: Organogenesis peak — heart beats, limb buds, facial features emerge.",
    range(9, 13): "Weeks 9–12: Fetus forms. Organs grow and refine; first trimester ends.",
    range(13, 17): "Weeks 13–16: Movement increases; anatomy starts to look more human.",
    range(17, 21): "Weeks 17–20: Quickening usually felt; detailed anatomy scan around 18–20 weeks.",
    range(21, 25): "Weeks 21–24: Brain grows rapidly; viability threshold approaches ~24 wks.",
    range(25, 29): "Weeks 25–28: Lungs and fat accumulation accelerate.",
    range(29, 33): "Weeks 29–32: Rapid weight gain; nervous system maturing.",
    range(33, 37): "Weeks 33–36: Final maturation; baby often turns head-down.",
    range(37, 41): "Weeks 37–40+: Full-term. Baby gains final weight."
}

STAGE_NAMES = {
    1: "Implantation (W1–2)",
    2: "Embryonic (W3–8)",
    3: "Early Fetal (W9–12)",
    4: "Mid Fetal (W13–24)",
    5: "Late Fetal (W25–40)"
}

class FetalDevelopmentSimulator:
    def __init__(self, save_path=None):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Fetal Development Simulator - Organs Added")
        self.clock = pygame.time.Clock()

        self.params = {
            'temperature': 12.0,
            'adhesion': 10.0,
            'volume_constraint': 8.0,
            'growth_rate': 0.20,
            'differentiation_rate': 0.12,
            'morphogen_diffusion': 0.6
        }

        self.grid = None
        self.morphogen_head = None
        self.morphogen_body = None
        self.iteration = 0
        self.is_running = False
        self.show_settings = False

        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 34)

        self.sliders = self._create_sliders()
        self.active_slider = None

        self.save_path = save_path
        self.reset_simulation()
        # camera for panning the grid view
        self.cam_x = 0
        self.cam_y = 0
        self.dragging = False
        self.drag_start = (0, 0)

    def _create_sliders(self):
        x_start = 650
        y_start = 250
        y_spacing = 60
        return [
            {'name': 'temperature','label': 'Temperature','min': 1.0,'max': 30.0,
             'rect': pygame.Rect(x_start, y_start, 300, 10),'description': 'Cell movement randomness'},
            {'name': 'adhesion','label': 'Adhesion','min': 1.0,'max': 30.0,
             'rect': pygame.Rect(x_start, y_start + y_spacing, 300, 10),'description': 'Cell-cell stickiness'},
            {'name': 'growth_rate','label': 'Growth Rate','min': 0.05,'max': 1.0,
             'rect': pygame.Rect(x_start, y_start + y_spacing * 2, 300, 10),'description': 'Development speed'},
            {'name': 'differentiation_rate','label': 'Differentiation Rate','min': 0.01,'max': 0.4,
             'rect': pygame.Rect(x_start, y_start + y_spacing * 3, 300, 10),'description': 'Cell specialization'},
            {'name': 'morphogen_diffusion','label': 'Morphogen Diffusion','min': 0.05,'max': 2.0,
             'rect': pygame.Rect(x_start, y_start + y_spacing * 4, 300, 10),'description': 'Pattern formation'}
        ]

    def init_grid(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        center = GRID_SIZE // 2
        radius = 4
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i*i + j*j <= radius*radius:
                    yy = center + i
                    xx = center + j
                    if 0 <= yy < GRID_SIZE and 0 <= xx < GRID_SIZE:
                        grid[yy, xx] = int(CellType.EMBRYO)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                dist = np.hypot(i - center, j - center)
                if GRID_SIZE * 0.32 < dist < GRID_SIZE * 0.42 and RNG.random() < 0.35:
                    grid[i, j] = int(CellType.PLACENTA)

        self.morphogen_head = np.zeros((GRID_SIZE, GRID_SIZE))
        self.morphogen_body = np.zeros((GRID_SIZE, GRID_SIZE))
        return grid

    def week(self):
        return min(40, max(1, (self.iteration // 7) + 1))

    def update_morphogens(self):
        center = GRID_SIZE // 2
        w = self.week()
        self.morphogen_head *= 0.94
        self.morphogen_body *= 0.94

        head_strength = 1.4 if 3 <= w <= 8 else (1.0 if w <= 12 else 0.6)
        head_y = center - max(4, 8 - int(min(w, 8) * 0.6))
        x0 = max(0, center - 6); x1 = min(GRID_SIZE, center + 6)
        y0 = max(0, head_y - 2); y1 = min(GRID_SIZE, head_y + 2)
        self.morphogen_head[y0:y1, x0:x1] += head_strength

        body_source_x = center
        y0 = max(0, center - 12); y1 = min(GRID_SIZE, center + 12)
        x0 = max(0, body_source_x - 2); x1 = min(GRID_SIZE, body_source_x + 2)
        self.morphogen_body[y0:y1, x0:x1] += 1.0

        sigma = max(0.1, float(self.params['morphogen_diffusion']))
        self.morphogen_head = gaussian_filter(self.morphogen_head, sigma=sigma)
        self.morphogen_body = gaussian_filter(self.morphogen_body, sigma=sigma)

    def get_anatomical_template(self, week):
        center = GRID_SIZE // 2
        t = {}
        # basic mapping as before, with organ anchor points inside body/head templates
        if week <= 2:
            t['embryo'] = (center, center, 6 + week)
        elif week <= 8:
            growth = (week - 2) / (8 - 2)
            head_size = 6 + int(12 * growth)
            body_len = 6 + int(20 * growth)
            t['head'] = (center - 10 - int(8 * growth), center, head_size)
            t['body'] = (center + 6, center, body_len)
        elif week <= 12:
            growth = (week - 8) / (12 - 8)
            t['head'] = (center - 18, center, 12)
            t['brain'] = (center - 18, center, 4 + int(4 * growth))
            t['body'] = (center - 2, center, 20 + int(6 * growth))
            limb_growth = 1 + int(3 * growth)
            t['arm_left'] = (center + 6, center - 7, 2 + limb_growth)
            t['arm_right'] = (center + 6, center + 7, 2 + limb_growth)
            t['leg_left'] = (center + 14, center - 5, 3 + limb_growth)
            t['leg_right'] = (center + 14, center + 5, 3 + limb_growth)
            t['umbilical'] = [(center + i, center) for i in range(16, 26)]
            # organ anchors within torso for weeks 9-12
            t['heart'] = (center - 2, center - 4, 3)
            t['liver'] = (center + 2, center - 2, 4)
            t['stomach'] = (center + 4, center + 2, 3)
            t['intestine'] = (center + 8, center + 2, 5)
            t['kidney_l'] = (center + 2, center - 6, 2)
            t['kidney_r'] = (center + 2, center + 6, 2)
            t['lungs'] = (center - 0, center - 1, 4)
            t['eyes'] = [(center - 18, center - 3), (center - 18, center + 3)]
            t['bladder'] = (center + 12, center, 2)
        elif week <= 24:
            growth = (week - 12) / (24 - 12)
            t['head'] = (center - 24, center, 16)
            t['brain'] = (center - 24, center, 8)
            t['body'] = (center - 2, center, 26 + int(10 * growth))
            t['arm_left'] = (center + 2, center - 14, 6 + int(6 * growth))
            t['arm_right'] = (center + 2, center + 14, 6 + int(6 * growth))
            t['leg_left'] = (center + 24, center - 8, 8 + int(8 * growth))
            t['leg_right'] = (center + 24, center + 8, 8 + int(8 * growth))
            t['umbilical'] = [(center + i, center) for i in range(16, 36)]
            # organs scale up
            t['heart'] = (center - 4, center - 4, 4 + int(3*growth))
            t['liver'] = (center + 2, center - 2, 6 + int(4*growth))
            t['stomach'] = (center + 6, center + 2, 4 + int(2*growth))
            t['intestine'] = (center + 10, center + 2, 6 + int(4*growth))
            t['kidney_l'] = (center + 4, center - 6, 3 + int(2*growth))
            t['kidney_r'] = (center + 4, center + 6, 3 + int(2*growth))
            t['lungs'] = (center - 2, center - 2, 6 + int(3*growth))
            t['eyes'] = [(center - 24, center - 4), (center - 24, center + 4)]
            t['bladder'] = (center + 16, center, 3)
        else:
            growth = min((week - 24) / (40 - 24), 1.0)
            t['head'] = (center - 28, center, 18)
            t['brain'] = (center - 28, center, 9)
            t['body'] = (center - 2, center, 30 + int(12 * growth))
            t['arm_left'] = (center + 4, center - 16, 10 + int(8 * growth))
            t['arm_right'] = (center + 4, center + 16, 10 + int(8 * growth))
            t['leg_left'] = (center + 28, center - 10, 12 + int(8 * growth))
            t['leg_right'] = (center + 28, center + 10, 12 + int(8 * growth))
            t['umbilical'] = [(center + i, center) for i in range(16, 38)]
            # mature organ anchors
            t['heart'] = (center - 6, center - 4, 6)
            t['liver'] = (center + 4, center - 2, 8)
            t['stomach'] = (center + 8, center + 3, 6)
            t['intestine'] = (center + 14, center + 3, 8)
            t['kidney_l'] = (center + 6, center - 6, 4)
            t['kidney_r'] = (center + 6, center + 6, 4)
            t['lungs'] = (center - 4, center - 3, 8)
            t['eyes'] = [(center - 28, center - 5), (center - 28, center + 5)]
            t['bladder'] = (center + 20, center, 4)
        return t

    def apply_morphogen_differentiation(self):
        w = self.week()
        if w <= 2:
            self.current_stage = 1
        elif w <= 8:
            self.current_stage = 2
        elif w <= 12:
            self.current_stage = 3
        elif w <= 24:
            self.current_stage = 4
        else:
            self.current_stage = 5

        template = self.get_anatomical_template(w)
        base_diff = float(self.params['differentiation_rate'])

        # Organogenesis window increases differentiation probability
        if 3 <= w <= 8:
            diff_rate = base_diff * 1.9
        elif 9 <= w <= 12:
            diff_rate = base_diff * 1.3
        else:
            diff_rate = base_diff * 0.9

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i, j] == int(CellType.PLACENTA):
                    continue

                # HEAD formation
                if 'head' in template:
                    hy, hx, hr = template['head']
                    dist = np.hypot(i - hy, j - hx)
                    morpho_head_level = self.morphogen_head[i, j]
                    prob = diff_rate * (1.0 + 0.8 * morpho_head_level)
                    if dist < hr and RNG.random() < prob:
                        self.grid[i, j] = int(CellType.HEAD)

                # BRAIN inside head
                if 'brain' in template and self.grid[i, j] == int(CellType.HEAD):
                    by, bx, br = template['brain']
                    if np.hypot(i - by, j - bx) < br and RNG.random() < diff_rate * 0.55:
                        self.grid[i, j] = int(CellType.BRAIN)

                # BODY
                if 'body' in template:
                    byy, bxx, br = template['body']
                    if np.hypot(i - byy, j - bxx) < br and RNG.random() < diff_rate * 0.85:
                        if self.grid[i, j] not in (int(CellType.HEAD), int(CellType.BRAIN)):
                            self.grid[i, j] = int(CellType.BODY)

                # LIMBS
                limbs = [('arm_left', CellType.ARM_LEFT), ('arm_right', CellType.ARM_RIGHT),
                         ('leg_left', CellType.LEG_LEFT), ('leg_right', CellType.LEG_RIGHT)]
                for limb_name, limb_type in limbs:
                    if limb_name in template:
                        ly, lx, lr = template[limb_name]
                        if 'arm' in limb_name:
                            dist_l = abs(i - ly) + 0.6 * abs(j - lx)
                        else:
                            dist_l = 0.6 * abs(i - ly) + abs(j - lx)
                        if dist_l < lr and RNG.random() < diff_rate * 0.6:
                            self.grid[i, j] = int(limb_type)

                # ORGANS: appear only at realistic weeks and within torso/head areas
                # Eyes (week 5+)
                if 'eyes' in template and self.week() >= 5:
                    for ey, ex in template['eyes']:
                        if np.hypot(i - ey, j - ex) < 2 and RNG.random() < diff_rate * 0.9:
                            self.grid[i, j] = int(CellType.EYE)

                # Heart (appears early; visible around wk6)
                if 'heart' in template and self.week() >= 5:
                    hy, hx, hr = template['heart']
                    if np.hypot(i - hy, j - hx) < hr and RNG.random() < diff_rate * 1.3:
                        self.grid[i, j] = int(CellType.HEART)

                # Lungs (initial buds wk5-6, expand later)
                if 'lungs' in template and self.week() >= 6:
                    ly, lx, lr = template['lungs']
                    if np.hypot(i - ly, j - lx) < lr and RNG.random() < diff_rate * 0.9:
                        # avoid overwriting brain/head
                        if self.grid[i, j] not in (int(CellType.HEAD), int(CellType.BRAIN), int(CellType.HEART)):
                            self.grid[i, j] = int(CellType.LUNG)

                # Liver (prominent wk6+)
                if 'liver' in template and self.week() >= 6:
                    ly, lx, lr = template['liver']
                    if np.hypot(i - ly, j - lx) < lr and RNG.random() < diff_rate * 1.0:
                        if self.grid[i, j] not in (int(CellType.HEAD), int(CellType.BRAIN)):
                            self.grid[i, j] = int(CellType.LIVER)

                # Stomach
                if 'stomach' in template and self.week() >= 7:
                    sy, sx, sr = template['stomach']
                    if np.hypot(i - sy, j - sx) < sr and RNG.random() < diff_rate * 0.9:
                        self.grid[i, j] = int(CellType.STOMACH)

                # Intestines
                if 'intestine' in template and self.week() >= 8:
                    iy, ix, ir = template['intestine']
                    if np.hypot(i - iy, j - ix) < ir and RNG.random() < diff_rate * 0.8:
                        self.grid[i, j] = int(CellType.INTESTINE)

                # Kidneys (pair)
                if 'kidney_l' in template and self.week() >= 9:
                    ky, kx, kr = template['kidney_l']
                    if np.hypot(i - ky, j - kx) < kr and RNG.random() < diff_rate * 0.7:
                        self.grid[i, j] = int(CellType.KIDNEY)
                if 'kidney_r' in template and self.week() >= 9:
                    ky, kx, kr = template['kidney_r']
                    if np.hypot(i - ky, j - kx) < kr and RNG.random() < diff_rate * 0.7:
                        self.grid[i, j] = int(CellType.KIDNEY)

                # Bladder (visible later)
                if 'bladder' in template and self.week() >= 10:
                    by, bx, br = template['bladder']
                    if np.hypot(i - by, j - bx) < br and RNG.random() < diff_rate * 0.7:
                        self.grid[i, j] = int(CellType.BLADDER)

        # Umbilical cord placement
        if 'umbilical' in template:
            for uy, ux in template['umbilical']:
                if 0 <= uy < GRID_SIZE and 0 <= ux < GRID_SIZE:
                    if self.grid[uy, ux] in (int(CellType.EMPTY), int(CellType.EMBRYO)) and RNG.random() < 0.45:
                        self.grid[uy, ux] = int(CellType.UMBILICAL)

    def calculate_volumes(self):
        vols = {}
        unique, counts = np.unique(self.grid, return_counts=True)
        for typ, cnt in zip(unique, counts):
            if typ != int(CellType.EMPTY):
                vols[int(typ)] = int(cnt)
        return vols

    def calculate_energy(self, x, y, new_type, volumes):
        current_type = int(self.grid[x, y])
        energy = 0.0
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                neighbor_type = int(self.grid[nx, ny])
                adhesion = float(self.params['adhesion'])
                if current_type == neighbor_type:
                    adhesion *= 1.5
                if new_type == neighbor_type:
                    adhesion *= 1.5
                if current_type != neighbor_type:
                    energy += adhesion
                if new_type != neighbor_type:
                    energy -= adhesion

        if current_type != int(CellType.EMPTY):
            current_vol = volumes.get(int(current_type), 0)
            week = self.week()
            target_vol = 40 + week * (float(self.params['growth_rate']) * 8.0)
            energy += float(self.params['volume_constraint']) * ((current_vol - target_vol) ** 2) / 2000.0
        return energy

    def monte_carlo_step(self):
        volumes = self.calculate_volumes()
        attempts = GRID_SIZE * 5
        for _ in range(attempts):
            x = RNG.integers(0, GRID_SIZE)
            y = RNG.integers(0, GRID_SIZE)
            dx, dy = [(-1,0),(1,0),(0,-1),(0,1)][RNG.integers(0,4)]
            nx, ny = x+dx, y+dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            cur = int(self.grid[x,y])
            new = int(self.grid[nx,ny])
            if (cur == int(CellType.PLACENTA) and new != int(CellType.EMPTY)) or (new == int(CellType.PLACENTA) and cur != int(CellType.EMPTY)):
                continue
            if cur != new:
                dE = self.calculate_energy(x, y, new, volumes)
                T = max(1e-6, float(self.params['temperature']))
                if dE < 0 or RNG.random() < np.exp(-dE / T):
                    self.grid[x,y] = new

    def reset_simulation(self):
        self.grid = self.init_grid()
        self.morphogen_head.fill(0)
        self.morphogen_body.fill(0)
        self.iteration = 0
        self.current_stage = 1
        self.is_running = False

    # ---- Drawing ----
    def draw_grid(self):
        grid_surface = pygame.Surface((DRAW_SIZE, DRAW_SIZE))
        grid_surface.fill(COLORS[CellType.EMPTY])
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                val = int(self.grid[i,j])
                color = COLORS.get(CellType(val), COLORS[CellType.EMPTY]) if val in [ct.value for ct in CellType] else COLORS[CellType.EMPTY]
                rect = (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(grid_surface, color, rect)
        # blit grid with camera offset so user can pan around
        blit_x = 20 + int(self.cam_x)
        blit_y = 100 + int(self.cam_y)
        self.screen.blit(grid_surface, (blit_x, blit_y))
        # overlay circle follows the grid (center offset by camera)
        circle_cx = blit_x + DRAW_SIZE // 2
        circle_cy = blit_y + DRAW_SIZE // 2
        pygame.draw.circle(self.screen, (130, 60, 190), (circle_cx, circle_cy), 280, 3)

    def draw_ui(self):
        for i in range(WINDOW_HEIGHT):
            c = int(18 + 12 * np.sin(i / WINDOW_HEIGHT * np.pi))
            pygame.draw.line(self.screen, (c, c, c+10), (0,i), (WINDOW_WIDTH, i))

        title = self.title_font.render("Fetal Development Simulator — Organs", True, (255,255,255))
        subtitle = self.small_font.render("Conceptual visualization (not medical). Weeks map to clinical milestones.", True, (200,200,255))
        self.screen.blit(title, (WINDOW_WIDTH//2 - title.get_width()//2, 18))
        self.screen.blit(subtitle, (WINDOW_WIDTH//2 - subtitle.get_width()//2, 54))

        self.draw_grid()
        self.draw_buttons()
        self.draw_info_panel()
        self.draw_legend()   # cell types + organ legend (organ legend placed under cell types)
        if self.show_settings:
            self.draw_settings()

    def draw_buttons(self):
        by = 740
        start_color = (120, 60, 180) if not self.is_running else (90,90,90)
        pygame.draw.rect(self.screen, start_color, (150, by, 120, 40), border_radius=10)
        pygame.draw.rect(self.screen, (200,150,255), (150, by, 120, 40), 2, border_radius=10)
        st = self.font.render("Pause" if self.is_running else "Start", True, (255,255,255))
        self.screen.blit(st, (176, by+10))

        pygame.draw.rect(self.screen, (70,70,90), (290, by, 120, 40), border_radius=10)
        pygame.draw.rect(self.screen, (150,150,180), (290, by, 120, 40), 2, border_radius=10)
        rt = self.font.render("Reset", True, (255,255,255))
        self.screen.blit(rt, (322, by+10))

        settings_color = (120,60,180) if self.show_settings else (70,70,90)
        pygame.draw.rect(self.screen, settings_color, (430, by, 120, 40), border_radius=10)
        pygame.draw.rect(self.screen, (200,150,255) if self.show_settings else (150,150,180), (430, by, 120, 40), 2, border_radius=10)
        s = self.font.render("Settings", True, (255,255,255))
        self.screen.blit(s, (460, by+10))

        # Close button (in-UI)
        pygame.draw.rect(self.screen, (180, 60, 60), (570, by, 120, 40), border_radius=10)
        pygame.draw.rect(self.screen, (255, 180, 180), (570, by, 120, 40), 2, border_radius=10)
        close_text = self.font.render("Close", True, (255,255,255))
        self.screen.blit(close_text, (596, by+10))

    def draw_info_panel(self):
        px, py = 650, 100
        pygame.draw.rect(self.screen, (60,40,100), (px, py, 520, 160), border_radius=10)
        pygame.draw.rect(self.screen, (140,70,200), (px, py, 520, 160), 2, border_radius=10)

        stage_title = self.font.render("Stage", True, (255,255,255))
        week_text = self.font.render(f"Week: {self.week()}", True, (255,220,180))
        stage_name = self.small_font.render(STAGE_NAMES.get(self.current_stage, "—"), True, (220,200,255))
        iteration_text = self.small_font.render(f"Day (iter): {self.iteration}", True, (200,160,220))

        self.screen.blit(stage_title, (px + 20, py + 12))
        self.screen.blit(stage_name, (px + 20, py + 40))
        self.screen.blit(week_text, (px + 240, py + 12))
        self.screen.blit(iteration_text, (px + 240, py + 42))

        milestone_text = "No milestone."
        for wk_range, text in MILESTONES.items():
            if self.week() in wk_range:
                milestone_text = text
                break
        lines = []
        words = milestone_text.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 48:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        y0 = py + 80
        for i, ln in enumerate(lines[:4]):
            t = self.small_font.render(ln, True, (220,200,230))
            self.screen.blit(t, (px + 20, y0 + i*18))

    def draw_legend(self):
        lx, ly = 650, 480
        pygame.draw.rect(self.screen, (40,40,60), (lx, ly, 520, 220), border_radius=10)
        pygame.draw.rect(self.screen, (140,70,200), (lx, ly, 520, 220), 2, border_radius=10)
        title = self.font.render("Cell Types", True, (255,255,255))
        self.screen.blit(title, (lx+20, ly+8))

        # cell type items
        items = [("Head", CellType.HEAD), ("Brain", CellType.BRAIN), ("Body", CellType.BODY),
                 ("Arms", CellType.ARM_LEFT), ("Legs", CellType.LEG_LEFT), ("Placenta", CellType.PLACENTA)]
        for i, (name, ctype) in enumerate(items):
            x = lx + 20 + (i % 3) * 170
            y = ly + 40 + (i // 3) * 36
            pygame.draw.rect(self.screen, COLORS[ctype], (x, y, 26, 26), border_radius=6)
            pygame.draw.rect(self.screen, (180,180,200), (x, y, 26, 26), 1, border_radius=6)
            txt = self.small_font.render(name, True, (220,200,255))
            self.screen.blit(txt, (x + 34, y + 3))

        # Organ legend (placed under the cell types)
        org_x = lx + 20
        org_y = ly + 40 + 72
        org_items = [("Heart", CellType.HEART), ("Liver", CellType.LIVER), ("Stomach", CellType.STOMACH),
                     ("Intestines", CellType.INTESTINE), ("Kidney", CellType.KIDNEY), ("Lungs", CellType.LUNG),
                     ("Eyes", CellType.EYE), ("Bladder", CellType.BLADDER)]
        for i, (name, ctype) in enumerate(org_items):
            x = org_x + (i % 2) * 250
            y = org_y + (i // 2) * 28
            pygame.draw.rect(self.screen, COLORS[ctype], (x, y, 20, 20), border_radius=4)
            pygame.draw.rect(self.screen, (200,200,220), (x, y, 20, 20), 1, border_radius=4)
            txt = self.small_font.render(name, True, (220,200,255))
            self.screen.blit(txt, (x + 28, y + 1))

    def draw_settings(self):
        for slider in self.sliders:
            label = self.font.render(slider['label'], True, (255,255,255))
            self.screen.blit(label, (slider['rect'].x, slider['rect'].y - 24))
            value = float(self.params[slider['name']])
            vtxt = self.small_font.render(f"{value:.2f}", True, (200,150,255))
            self.screen.blit(vtxt, (slider['rect'].x + slider['rect'].width + 10, slider['rect'].y - 6))
            pygame.draw.rect(self.screen, (70,70,90), slider['rect'], border_radius=5)
            normalized = (value - slider['min']) / max(1e-6, (slider['max'] - slider['min']))
            hx = int(slider['rect'].x + normalized * slider['rect'].width)
            pygame.draw.circle(self.screen, (120,60,180), (hx, slider['rect'].y + 5), 11)
            pygame.draw.circle(self.screen, (200,150,255), (hx, slider['rect'].y + 5), 11, 2)
            desc = self.small_font.render(slider['description'], True, (180,160,220))
            self.screen.blit(desc, (slider['rect'].x, slider['rect'].y + 16))

    def handle_click(self, pos):
        x, y = pos
        if 740 <= y <= 780:
            if 150 <= x <= 270:
                self.is_running = not self.is_running
            elif 290 <= x <= 410:
                self.reset_simulation()
            elif 430 <= x <= 550:
                self.show_settings = not self.show_settings
            elif 570 <= x <= 690:
                # in-UI close button: quit pygame and exit
                pygame.quit()
                sys.exit()
        if self.show_settings:
            for s in self.sliders:
                if s['rect'].collidepoint(pos):
                    self.active_slider = s
                    self.update_slider(pos)

    def update_slider(self, pos):
        if not self.active_slider:
            return
        s = self.active_slider
        x = max(s['rect'].x, min(pos[0], s['rect'].x + s['rect'].width))
        normalized = (x - s['rect'].x) / max(1, s['rect'].width)
        value = s['min'] + normalized * (s['max'] - s['min'])
        self.params[s['name']] = value

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # left click -> UI / slider handling
                    if event.button == 1:
                        self.handle_click(event.pos)
                    # right or middle click -> start panning if clicked on grid area
                    elif event.button in (2, 3):
                        mx, my = event.pos
                        grid_x0, grid_y0 = 20 + int(self.cam_x), 100 + int(self.cam_y)
                        grid_x1, grid_y1 = grid_x0 + DRAW_SIZE, grid_y0 + DRAW_SIZE
                        if grid_x0 <= mx <= grid_x1 and grid_y0 <= my <= grid_y1:
                            self.dragging = True
                            self.drag_start = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    # stop dragging on middle/right release; left release stops slider interaction
                    if event.button in (2, 3):
                        self.dragging = False
                    elif event.button == 1:
                        self.active_slider = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        # move camera by mouse delta
                        dx = event.pos[0] - self.drag_start[0]
                        dy = event.pos[1] - self.drag_start[1]
                        self.cam_x += dx
                        self.cam_y += dy
                        self.drag_start = event.pos
                    elif self.active_slider:
                        self.update_slider(event.pos)
                elif event.type == pygame.KEYDOWN:
                    # arrow keys / WASD to nudge camera
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.cam_x += 20
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.cam_x -= 20
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.cam_y += 20
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.cam_y -= 20

            if self.is_running:
                if self.week() >= 40:
                    self.is_running = False
                else:
                    self.update_morphogens()
                    self.monte_carlo_step()
                    self.apply_morphogen_differentiation()
                    self.iteration += 1

            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    sim = FetalDevelopmentSimulator()
    sim.run()
