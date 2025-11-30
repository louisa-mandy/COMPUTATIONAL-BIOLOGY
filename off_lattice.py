# Off-Lattice Baby Organ Growth Simulation
# Integrated with a pygame UI adapted from the original Baby_Organ.py
# Features implemented:
# - Off-lattice (agent-based) cell model with repulsion, adhesion, and cohesion
# - Cell differentiation into organs using anatomical templates (week-based)
# - Cell division (mitosis) for growth
# - Adhesion forces (differential adhesion by cell type)
# - Pygame rendering with camera pan, settings panel, sliders, start/reset
# - Switchable mode placeholder (CPM vs Off-lattice) — currently uses Off-lattice

import pygame
import numpy as np
import sys
import psutil
import time
import math
from enum import IntEnum

pygame.init()
pygame.font.init()

# ---- Configurable constants ----
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 820
SIM_AREA_W = 600
SIM_AREA_H = 600
SIM_ORIGIN_X = 20
SIM_ORIGIN_Y = 100

RNG = np.random.default_rng(42)

# ---- Cell types ----
class CellType(IntEnum):
    STEM = 0
    HEAD = 1
    BRAIN = 2
    BODY = 3
    ARM = 4
    LEG = 5
    PLACENTA = 6
    UMBILICAL = 7
    HEART = 20
    LIVER = 21
    STOMACH = 22
    INTESTINE = 23
    KIDNEY = 24
    LUNG = 25
    EYE = 26
    BLADDER = 27

# Colors for agent rendering
COLORS = {
    CellType.STEM: "#785AB4",
    CellType.HEAD: "#FFC8A0",
    CellType.BRAIN: "#FFA078",
    CellType.BODY: "#FF96AA",
    CellType.ARM: "#DC8296",
    CellType.LEG: "#C86E8C",
    CellType.PLACENTA: "#785AB4",
    CellType.UMBILICAL: "#A082D2",
    CellType.HEART: "#FF0000",
    CellType.LIVER: "#0000FF",
    CellType.STOMACH: "#FFFFFF",
    CellType.INTESTINE: "#8B4513",
    CellType.KIDNEY: "#FFFF00",
    CellType.LUNG: "#FF69B4",
    CellType.EYE: "#000000",
    CellType.BLADDER: "#00008B",
}


# Week-based milestones (simplified)
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

# ---- Agent (cell) class for off-lattice simulation ----
class Cell:
    def __init__(self, x, y, cell_type=CellType.STEM):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.type = int(cell_type)
        self.age = 0.0

    def pos(self):
        return (self.x, self.y)

# ---- Off-lattice embryo engine ----
class OffLatticeEngine:
    def __init__(self, area_w=SIM_AREA_W, area_h=SIM_AREA_H, init_cells=500):
        self.area_w = area_w
        self.area_h = area_h
        self.cells = []
        self.iteration = 0
        self.time = 0.0

        # parameters that UI can change
        self.params = {
            'temperature': 1.0,
            'adhesion_base': 0.2,
            'growth_rate': 0.04,  # INCREASED from 0.02
            'differentiation_rate': 0.08,  # INCREASED from 0.05
            'morphogen_diffusion': 0.6,
            'speed': 1.0
        }

        self.adhesion_preferences = {}
        self._init_adhesion_rules()

        # Initialize cells in LARGER cluster near center
        cx, cy = area_w/2, area_h/2
        for _ in range(init_cells):
            a = RNG.random() * 2 * math.pi
            r = RNG.random() * 25  # INCREASED from 8 to 25
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            self.cells.append(Cell(x, y, CellType.STEM))

      

    def _init_adhesion_rules(self):
        # adhesion preference: same-type prefer to stick more (higher positive value)
        # organs may prefer certain neighbors; represented as scaling factor
        for t in CellType:
            self.adhesion_preferences[int(t)] = {}
            for u in CellType:
                # base: same-type gets +0.4, different-type gets -0.1
                self.adhesion_preferences[int(t)][int(u)] = 0.4 if t == u else -0.05
        
        # HEAD cells stick together VERY strongly
        self.adhesion_preferences[int(CellType.HEAD)][int(CellType.HEAD)] = 1.2
        self.adhesion_preferences[int(CellType.BRAIN)][int(CellType.BRAIN)] = 1.0
        self.adhesion_preferences[int(CellType.HEAD)][int(CellType.BRAIN)] = 0.9
        self.adhesion_preferences[int(CellType.BRAIN)][int(CellType.HEAD)] = 0.9
        
        # HEAD/BRAIN repel BODY
        self.adhesion_preferences[int(CellType.HEAD)][int(CellType.BODY)] = -0.3
        self.adhesion_preferences[int(CellType.BODY)][int(CellType.HEAD)] = -0.3
        self.adhesion_preferences[int(CellType.BRAIN)][int(CellType.BODY)] = -0.3
        self.adhesion_preferences[int(CellType.BODY)][int(CellType.BRAIN)] = -0.3
        
        # BODY cells stick together
        self.adhesion_preferences[int(CellType.BODY)][int(CellType.BODY)] = 0.8
        
        # make organs slightly more adhesive to body
        self.adhesion_preferences[int(CellType.HEART)][int(CellType.BODY)] = 1.4
        self.adhesion_preferences[int(CellType.LIVER)][int(CellType.BODY)] = 1.4
        self.adhesion_preferences[int(CellType.LUNG)][int(CellType.BODY)] = 1.4

        self.adhesion_preferences[int(CellType.STOMACH)][int(CellType.BODY)] = 1.2
        self.adhesion_preferences[int(CellType.INTESTINE)][int(CellType.BODY)] = 1.2
        self.adhesion_preferences[int(CellType.KIDNEY)][int(CellType.BODY)] = 1.2
        self.adhesion_preferences[int(CellType.BLADDER)][int(CellType.BODY)] = 1.0
        self.adhesion_preferences[int(CellType.BODY)][int(CellType.HEART)] = 1.4

        
    def week(self):
        # map iteration -> week roughly (7 iterations per week)
        return min(40, max(1, (self.iteration // 7) + 1))

    def step(self, dt=1.0):
        # compute forces
        n = len(self.cells)
        positions = np.array([[c.x, c.y] for c in self.cells])
        # spatial hashing / simple grid to optimize neighbor search
        cell_forces = [(0.0, 0.0) for _ in range(n)]
        grid_size = 12
        buckets = {}
        for i, c in enumerate(self.cells):
            bx = int(c.x // grid_size)
            by = int(c.y // grid_size)
            buckets.setdefault((bx,by), []).append(i)

        for i, c in enumerate(self.cells):
            fx, fy = 0.0, 0.0
            bx = int(c.x // grid_size)
            by = int(c.y // grid_size)
            neighbors_idx = []
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    neighbors_idx.extend(buckets.get((bx+dx, by+dy), []))

            for j in neighbors_idx:
                if j == i: continue
                other = self.cells[j]
                dx = c.x - other.x
                dy = c.y - other.y
                dist = math.hypot(dx, dy) + 1e-6
                # INCREASED preferred distance to prevent overlap
                preferred_dist = 4.5  # INCREASED from 2.8 to 4.5
                if dist < 0.1:
                    # numerical safety
                    continue
                # MUCH STRONGER repulsion to prevent layering
                if dist < preferred_dist:
                    rep = (preferred_dist - dist) * 3.5  # INCREASED from 1.0 to 3.5
                    fx += (dx/dist) * rep
                    fy += (dy/dist) * rep
                # adhesion (attractive) beyond preferred up to some radius
                elif dist < 18:  # Slightly increased adhesion range
                    # adhesion strength depends on types
                    a_pref = self.adhesion_preferences.get(c.type, {}).get(other.type, 0.0)
                    adh = a_pref * self.params['adhesion_base'] * (1 - (dist- preferred_dist)/14)
                    fx -= (dx/dist) * adh
                    fy -= (dy/dist) * adh
                    
            # Add centripetal force to pull HEAD cells toward head center
            if c.type == int(CellType.HEAD) or c.type == int(CellType.BRAIN):
                head_cx = self.area_w / 2
                head_cy = self.area_h / 2 - 70  # Head center position
                to_center_x = head_cx - c.x
                to_center_y = head_cy - c.y
                dist_to_center = math.hypot(to_center_x, to_center_y)
                if dist_to_center > 5:  # Only apply if not already at center
                    pull_strength = 0.2  # REDUCED from 0.3 to 0.2
                    fx += (to_center_x / dist_to_center) * pull_strength
                    fy += (to_center_y / dist_to_center) * pull_strength

            # small noise
            temp = self.params['temperature']
            fx += (RNG.random()-0.5) * temp * 0.5  # Reduced noise
            fy += (RNG.random()-0.5) * temp * 0.5

            cell_forces[i] = (fx, fy)

        # integrate velocities and positions
        for i, c in enumerate(self.cells):
            fx, fy = cell_forces[i]
            # simple overdamped dynamics with MORE damping
            c.vx = (c.vx + fx * 0.2) * 0.85  # INCREASED damping from 0.9 to 0.85
            c.vy = (c.vy + fy * 0.2) * 0.85
            c.x += c.vx * dt * self.params['speed']
            c.y += c.vy * dt * self.params['speed']
            # confine to area with soft boundary
            pad = 4
            c.x = max(pad, min(self.area_w - pad, c.x))
            c.y = max(pad, min(self.area_h - pad, c.y))
            c.age += dt

        self.time += dt
        self.iteration += 1

        # division & differentiation
        self._division_pass()
        self._differentiation_pass()
        
    def _division_pass(self):
        # probabilistic division based on growth_rate and local crowding
        new_cells = []
        for c in self.cells:
            prob = self.params['growth_rate']
            # reduce prob when overcrowded (count neighbors)
            nb = self._count_neighbors(c, radius=4)
            prob *= (1.0 - min(0.9, nb/8.0))
            if RNG.random() < prob:
                # spawn new cell near parent
                ang = RNG.random() * 2*math.pi
                r = 1.5 + RNG.random() * 1.0
                nx = c.x + math.cos(ang)*r
                ny = c.y + math.sin(ang)*r
                child = Cell(nx, ny, c.type if RNG.random() < 0.95 else CellType.STEM)
                new_cells.append(child)
        self.cells.extend(new_cells)

    def _differentiation_pass(self):
        w = self.week()
        template = self._anatomical_template(w)
        
        for c in self.cells:
            # Morphogen gradient influence based on position
            cx, cy = self.area_w/2, self.area_h/2
            
            # Distance from head region (top)
            head_dist = math.hypot(c.x - cx, c.y - (cy - 90))
            # Distance from body center
            body_dist = math.hypot(c.x - cx, c.y - cy)
            
            # Morphogen concentrations (exponential decay) - STRONGER influence
            morphogen_head = math.exp(-head_dist / (40 * self.params['morphogen_diffusion']))
            morphogen_body = math.exp(-body_dist / (50 * self.params['morphogen_diffusion']))
            
            # Template anchor zones - prioritize head/brain first
            best_target = None
            best_score = -1
            
            for name, anchor in template.items():
                ax, ay, ar = anchor
                dx = c.x - ax
                dy = c.y - ay
                dist = math.hypot(dx, dy)
                
                # Use softer boundary - cells can differentiate slightly outside radius
                if dist < ar * 1.2:  # 20% tolerance
                    # decide organ type mapping
                    target = None
                    priority = 0  # Higher priority = differentiate first
                    
                    if name.startswith('head'):
                        target = CellType.HEAD
                        priority = 100  # Highest priority
                    elif name.startswith('brain'):
                        target = CellType.BRAIN
                        priority = 95
                    elif name.startswith('body'):
                        target = CellType.BODY
                        priority = 50
                    elif name.startswith('arm'):
                        target = CellType.ARM
                        priority = 40
                    elif name.startswith('leg'):
                        target = CellType.LEG
                        priority = 40
                    elif name.startswith('heart'):
                        target = CellType.HEART
                        priority = 30
                    elif name.startswith('liver'):
                        target = CellType.LIVER
                        priority = 25
                    elif name.startswith('stomach'):
                        target = CellType.STOMACH
                        priority = 20
                    elif name.startswith('intestine'):
                        target = CellType.INTESTINE
                        priority = 20
                    elif name.startswith('kidney'):
                        target = CellType.KIDNEY
                        priority = 20
                    elif name.startswith('lungs') and w >= 9:
                        target = CellType.LUNG
                        priority = 25
                    elif name.startswith('eye'):
                        target = CellType.EYE
                        priority = 35
                    elif name.startswith('bladder'):
                        target = CellType.BLADDER
                        priority = 15

                    if target is not None:
                        # Morphogen-enhanced differentiation probability
                        morphogen_factor = 1.0
                        if target in (CellType.HEAD, CellType.BRAIN, CellType.EYE):
                            morphogen_factor = 1.0 + morphogen_head * 5.0  # Even stronger
                        else:
                            morphogen_factor = 1.0 + morphogen_body * 2.5
                        
                        # Distance-based probability (stronger near center of anchor)
                        distance_factor = 1.0 - (dist / (ar * 1.2))
                        
                        # Calculate score including priority
                        score = (priority * 
                                morphogen_factor * 
                                distance_factor *
                                (1.0 + 1.2 * math.exp(-c.age*0.005)))
                        
                        if score > best_score:
                            best_score = score
                            best_target = target
            
            # Apply differentiation based on best match
            if best_target is not None:
                prob = self.params['differentiation_rate'] * (best_score / 100.0)
                if RNG.random() < prob:
                    c.type = int(best_target)
                                            
    def _count_neighbors(self, cell, radius=5):
        cnt = 0
        r2 = radius*radius
        for o in self.cells:
            if o is cell: continue
            dx = o.x - cell.x
            dy = o.y - cell.y
            if dx*dx + dy*dy < r2:
                cnt += 1
        return cnt

    def _anatomical_template(self, week):
        # returns anchors in simulation coordinate space (x,y,radius)
        # VERTICAL ORIENTATION: Head at top, body center, legs at bottom
        cx, cy = self.area_w/2, self.area_h/2
        t = {}
        
        if week <= 2:
            # Weeks 1-2: Fertilization & implantation - just a small cell cluster
            t['embryo'] = (cx, cy, 15 + week * 3)
            
        elif week <= 5:
            # Weeks 3-5: Embryonic disk, early heart tube forming
            growth = (week - 2) / (5 - 2)
            
            # Very early differentiation - small disk-like structure
            t['embryo'] = (cx, cy, 20 + int(10 * growth))
            # Early heart tube starting to form
            if week >= 4:
                t['heart'] = (cx, cy, 8 + int(4 * growth))
                
        elif week <= 8:
            # Weeks 6-8: Organogenesis peak - heart beats, limb buds, facial features
            growth = (week - 5) / (8 - 5)
            
            # Head and body start separating
            head_offset = -20 - (15 * growth)
            body_offset = 5 + (10 * growth)
            
            t['head'] = (cx, cy + head_offset, 20 + int(12 * growth))
            t['body'] = (cx, cy + body_offset, 25 + int(20 * growth))
            
            # Heart is now beating (more prominent)
            t['heart'] = (cx - 8, cy - 5, 10 + int(5 * growth))
            
            # Limb buds emerge (week 6+)
            if week >= 6:
                arm_y = cy - 5
                t['arm_left'] = (cx - 30, arm_y, 12 + int(3 * growth))
                t['arm_right'] = (cx + 30, arm_y, 12 + int(3 * growth))
            
            # Leg buds appear slightly later
            if week >= 7:
                leg_y = cy + 25
                t['leg_left'] = (cx - 15, leg_y, 12 + int(3 * growth))
                t['leg_right'] = (cx + 15, leg_y, 12 + int(3 * growth))
            
            # Eyes start forming (facial features)
            if week >= 7:
                t['eye_l'] = (cx - 8, cy + head_offset - 5, 3)
                t['eye_r'] = (cx + 8, cy + head_offset - 5, 3)
                
        elif week <= 12:
            # Weeks 9-12: Fetus forms, organs grow and refine
            growth = (week - 8) / (12 - 8)
            
            # More defined fetal structure
            t['head'] = (cx, cy - 70, 35 + int(10 * growth))
            t['brain'] = (cx, cy - 70, 18 + int(6 * growth))
            t['body'] = (cx, cy, 50 + int(12 * growth))
            
            # Limbs more developed
            arm_y = cy - 25
            t['arm_left'] = (cx - 50, arm_y, 20 + int(5 * growth))
            t['arm_right'] = (cx + 50, arm_y, 20 + int(5 * growth))
            
            leg_y = cy + 45
            t['leg_left'] = (cx - 22, leg_y, 22 + int(5 * growth))
            t['leg_right'] = (cx + 22, leg_y, 22 + int(5 * growth))
            
            # Organs developing and refining
            t['heart'] = (cx - 15, cy - 30, 8 + int(2 * growth))
            t['liver'] = (cx + 18, cy - 18, 8 + int(2 * growth))
            t['lungs'] = (cx, cy - 32, 15 + int(3 * growth))
            t['stomach'] = (cx - 10, cy - 5, 13 + int(3 * growth))
            t['intestine'] = (cx + 8, cy + 8, 16 + int(4 * growth))
            t['kidney_l'] = (cx - 22, cy - 10, 8 + int(2 * growth))
            t['kidney_r'] = (cx + 22, cy - 10, 8 + int(2 * growth))
            t['bladder'] = (cx, cy + 22, 11 + int(2 * growth))
            
            # Eyes more defined
            t['eye_l'] = (cx - 10, cy - 75, 4)
            t['eye_r'] = (cx + 10, cy - 75, 4)
            
            # Umbilical cord
            t['umbilical'] = (cx, cy + 70, 8 + int(2 * growth))
            
        elif week <= 16:
            # Weeks 13-16: Movement increases, anatomy more human
            growth = (week - 12) / (16 - 12)
            
            # Growing fetus with human proportions
            t['head'] = (cx, cy - 90, 45 + int(8 * growth))
            t['brain'] = (cx, cy - 90, 22 + int(6 * growth))
            t['body'] = (cx, cy, 60 + int(15 * growth))
            
            # Limbs elongating
            t['arm_left'] = (cx - 60, cy - 30, 24 + int(6 * growth))
            t['arm_right'] = (cx + 60, cy - 30, 24 + int(6 * growth))
            t['leg_left'] = (cx - 24, cy + 55, 26 + int(6 * growth))
            t['leg_right'] = (cx + 24, cy + 55, 26 + int(6 * growth))
            
            # Organs maturing
            t['heart'] = (cx - 18, cy - 38, 16 + int(3 * growth))
            t['liver'] = (cx + 22, cy - 22, 19 + int(4 * growth))
            t['lungs'] = (cx, cy - 40, 17 + int(3 * growth))
            t['stomach'] = (cx - 12, cy - 8, 15 + int(3 * growth))
            t['intestine'] = (cx + 10, cy + 10, 18 + int(4 * growth))
            t['kidney_l'] = (cx - 26, cy - 12, 10 + int(2 * growth))
            t['kidney_r'] = (cx + 26, cy - 12, 10 + int(2 * growth))
            t['bladder'] = (cx, cy + 28, 12 + int(2 * growth))
            
            t['eye_l'] = (cx - 12, cy - 98, 4 + int(1 * growth))
            t['eye_r'] = (cx + 12, cy - 98, 4 + int(1 * growth))
            t['umbilical'] = (cx, cy + 85, 10 + int(2 * growth))
            
        elif week <= 20:
            # Weeks 17-20: Quickening felt, detailed anatomy scan period
            growth = (week - 16) / (20 - 16)
            
            # Well-formed fetus
            t['head'] = (cx, cy - 105, 50 + int(8 * growth))
            t['brain'] = (cx, cy - 105, 26 + int(6 * growth))
            t['body'] = (cx, cy + 5, 70 + int(12 * growth))
            
            # Active limbs
            t['arm_left'] = (cx - 68, cy - 35, 28 + int(6 * growth))
            t['arm_right'] = (cx + 68, cy - 35, 28 + int(6 * growth))
            t['leg_left'] = (cx - 26, cy + 65, 30 + int(6 * growth))
            t['leg_right'] = (cx + 26, cy + 65, 30 + int(6 * growth))
            
            # All organs visible on scan
            t['heart'] = (cx - 20, cy - 45, 18 + int(3 * growth))
            t['liver'] = (cx + 26, cy - 28, 22 + int(4 * growth))
            t['lungs'] = (cx, cy - 48, 20 + int(3 * growth))
            t['stomach'] = (cx - 14, cy - 12, 16 + int(3 * growth))
            t['intestine'] = (cx + 12, cy + 14, 22 + int(4 * growth))
            t['kidney_l'] = (cx - 28, cy - 16, 11 + int(2 * growth))
            t['kidney_r'] = (cx + 28, cy - 16, 11 + int(2 * growth))
            t['bladder'] = (cx, cy + 34, 13 + int(2 * growth))
            
            t['eye_l'] = (cx - 14, cy - 113, 5)
            t['eye_r'] = (cx + 14, cy - 113, 5)
            t['umbilical'] = (cx, cy + 95, 11 + int(2 * growth))
            
        elif week <= 24:
            # Weeks 21-24: Brain grows rapidly, viability threshold
            growth = (week - 20) / (24 - 20)
            
            # Brain development emphasis
            t['head'] = (cx, cy - 115, 55 + int(8 * growth))
            t['brain'] = (cx, cy - 115, 30 + int(8 * growth))  # Brain growing rapidly
            t['body'] = (cx, cy + 8, 78 + int(12 * growth))
            
            # Well-developed limbs
            t['arm_left'] = (cx - 72, cy - 38, 32 + int(6 * growth))
            t['arm_right'] = (cx + 72, cy - 38, 32 + int(6 * growth))
            t['leg_left'] = (cx - 28, cy + 72, 34 + int(6 * growth))
            t['leg_right'] = (cx + 28, cy + 72, 34 + int(6 * growth))
            
            # Organs maturing for viability
            t['heart'] = (cx - 22, cy - 50, 20 + int(3 * growth))
            t['liver'] = (cx + 28, cy - 32, 24 + int(4 * growth))
            t['lungs'] = (cx, cy - 52, 22 + int(4 * growth))  # Lungs developing
            t['stomach'] = (cx - 16, cy - 15, 18 + int(3 * growth))
            t['intestine'] = (cx + 14, cy + 16, 24 + int(4 * growth))
            t['kidney_l'] = (cx - 30, cy - 18, 12 + int(2 * growth))
            t['kidney_r'] = (cx + 30, cy - 18, 12 + int(2 * growth))
            t['bladder'] = (cx, cy + 38, 14 + int(2 * growth))
            
            t['eye_l'] = (cx - 16, cy - 123, 5 + int(1 * growth))
            t['eye_r'] = (cx + 16, cy - 123, 5 + int(1 * growth))
            t['umbilical'] = (cx, cy + 100, 12 + int(2 * growth))
            
        elif week <= 28:
            # Weeks 25-28: Lungs and fat accumulation accelerate
            growth = (week - 24) / (28 - 24)
            
            t['head'] = (cx, cy - 125, 60 + int(8 * growth))
            t['brain'] = (cx, cy - 125, 35 + int(5 * growth))
            t['body'] = (cx, cy + 10, 85 + int(12 * growth))  # Fat accumulation
            
            # Rounded limbs (fat deposits)
            t['arm_left'] = (cx - 76, cy - 42, 36 + int(6 * growth))
            t['arm_right'] = (cx + 76, cy - 42, 36 + int(6 * growth))
            t['leg_left'] = (cx - 30, cy + 78, 38 + int(6 * growth))
            t['leg_right'] = (cx + 30, cy + 78, 38 + int(6 * growth))
            
            # Lung development critical
            t['heart'] = (cx - 24, cy - 54, 22 + int(2 * growth))
            t['liver'] = (cx + 30, cy - 35, 26 + int(3 * growth))
            t['lungs'] = (cx, cy - 56, 25 + int(5 * growth))  # Lungs accelerating
            t['stomach'] = (cx - 17, cy - 18, 19 + int(2 * growth))
            t['intestine'] = (cx + 15, cy + 18, 26 + int(3 * growth))
            t['kidney_l'] = (cx - 32, cy - 20, 13 + int(2 * growth))
            t['kidney_r'] = (cx + 32, cy - 20, 13 + int(2 * growth))
            t['bladder'] = (cx, cy + 40, 15 + int(2 * growth))
            
            t['eye_l'] = (cx - 17, cy - 133, 6)
            t['eye_r'] = (cx + 17, cy - 133, 6)
            t['umbilical'] = (cx, cy + 108, 13 + int(2 * growth))
            
        elif week <= 32:
            # Weeks 29-32: Rapid weight gain, nervous system maturing
            growth = (week - 28) / (32 - 28)
            
            t['head'] = (cx, cy - 135, 65 + int(8 * growth))
            t['brain'] = (cx, cy - 135, 38 + int(6 * growth))  # Nervous system
            t['body'] = (cx, cy + 12, 92 + int(12 * growth))  # Rapid weight gain
            
            # Fuller limbs
            t['arm_left'] = (cx - 78, cy - 45, 40 + int(6 * growth))
            t['arm_right'] = (cx + 78, cy - 45, 40 + int(6 * growth))
            t['leg_left'] = (cx - 32, cy + 85, 42 + int(6 * growth))
            t['leg_right'] = (cx + 32, cy + 85, 42 + int(6 * growth))
            
            # Mature organs
            t['heart'] = (cx - 25, cy - 57, 23 + int(2 * growth))
            t['liver'] = (cx + 32, cy - 37, 28 + int(3 * growth))
            t['lungs'] = (cx, cy - 60, 28 + int(3 * growth))
            t['stomach'] = (cx - 18, cy - 20, 20 + int(2 * growth))
            t['intestine'] = (cx + 16, cy + 20, 28 + int(3 * growth))
            t['kidney_l'] = (cx - 34, cy - 22, 14 + int(2 * growth))
            t['kidney_r'] = (cx + 34, cy - 22, 14 + int(2 * growth))
            t['bladder'] = (cx, cy + 43, 16 + int(2 * growth))
            
            t['eye_l'] = (cx - 18, cy - 143, 6)
            t['eye_r'] = (cx + 18, cy - 143, 6)
            t['umbilical'] = (cx, cy + 115, 14)
            
        elif week <= 36:
            # Weeks 33-36: Final maturation, baby turns head-down
            growth = (week - 32) / (36 - 32)
            
            t['head'] = (cx, cy - 145, 70 + int(8 * growth))
            t['brain'] = (cx, cy - 145, 42 + int(6 * growth))
            t['body'] = (cx, cy + 15, 100 + int(12 * growth))
            
            # Full limbs
            t['arm_left'] = (cx - 80, cy - 48, 44 + int(6 * growth))
            t['arm_right'] = (cx + 80, cy - 48, 44 + int(6 * growth))
            t['leg_left'] = (cx - 34, cy + 92, 46 + int(6 * growth))
            t['leg_right'] = (cx + 34, cy + 92, 46 + int(6 * growth))
            
            # Fully mature organs
            t['heart'] = (cx - 26, cy - 60, 24)
            t['liver'] = (cx + 34, cy - 40, 30)
            t['lungs'] = (cx, cy - 63, 30)
            t['stomach'] = (cx - 19, cy - 22, 21)
            t['intestine'] = (cx + 17, cy + 22, 30)
            t['kidney_l'] = (cx - 36, cy - 24, 15)
            t['kidney_r'] = (cx + 36, cy - 24, 15)
            t['bladder'] = (cx, cy + 45, 17)
            
            t['eye_l'] = (cx - 19, cy - 153, 6)
            t['eye_r'] = (cx + 19, cy - 153, 6)
            t['umbilical'] = (cx, cy + 120, 14)
            
        else:
            # Weeks 37-40+: Full-term, final weight gain
            growth = min((week - 36) / (40 - 36), 1.0)
            
            t['head'] = (cx, cy - 150, 75 + int(10 * growth))
            t['brain'] = (cx, cy - 150, 46 + int(8 * growth))
            t['body'] = (cx, cy + 18, 108 + int(15 * growth))
            
            # Full-term limbs
            t['arm_left'] = (cx - 82, cy - 50, 48 + int(8 * growth))
            t['arm_right'] = (cx + 82, cy - 50, 48 + int(8 * growth))
            t['leg_left'] = (cx - 36, cy + 98, 50 + int(10 * growth))
            t['leg_right'] = (cx + 36, cy + 98, 50 + int(10 * growth))
            
            # Full-term organs
            t['heart'] = (cx - 27, cy - 62, 25)
            t['liver'] = (cx + 36, cy - 42, 32)
            t['lungs'] = (cx, cy - 65, 32)
            t['stomach'] = (cx - 20, cy - 24, 22)
            t['intestine'] = (cx + 18, cy + 24, 32)
            t['kidney_l'] = (cx - 38, cy - 26, 16)
            t['kidney_r'] = (cx + 38, cy - 26, 16)
            t['bladder'] = (cx, cy + 48, 18)
            
            t['eye_l'] = (cx - 20, cy - 158, 6)
            t['eye_r'] = (cx + 20, cy - 158, 6)
            t['umbilical'] = (cx, cy + 125, 15)
            
        return t

# ---- Pygame UI and integration ----
class FetalDevelopmentUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Fetal Development — Off-Lattice Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 34)

        self.engine = OffLatticeEngine(area_w=SIM_AREA_W, area_h=SIM_AREA_H, init_cells=600)
        # sync params with UI sliders
        self.params = self.engine.params

        self.is_running = False
        self.show_settings = False
        self.cam_x = 0
        self.cam_y = 0
        self.dragging = False
        self.drag_start = (0,0)

        # sliders
        self.sliders = self._create_sliders()
        self.active_slider = None
        self.settings_scroll = 0

        # mode (0: off-lattice, 1: cmp placeholder)
        self.mode = 0

        # performance metrics
        self.cpu_usage = 0.0
        self.last_perf_update = time.time()

    def _create_sliders(self):
        x_start = 650
        y_start = 320
        y_spacing = 60
        return [
            {'name': 'speed','label': 'Speed','min': 0.1,'max': 10.0,
            'rect': pygame.Rect(x_start, y_start, 300, 10),'description': 'Simulation speed multiplier'},

            {'name': 'temperature','label': 'Temperature','min': 1.0,'max': 30.0,
            'rect': pygame.Rect(x_start, y_start + y_spacing, 300, 10),'description': 'Random motility'},

            {'name': 'adhesion_base','label': 'Adhesion','min': 1.0,'max': 30.0,
            'rect': pygame.Rect(x_start, y_start + y_spacing * 2, 300, 10),'description': 'Cell-cell adhesion strength'},

            {'name': 'growth_rate','label': 'Growth Rate','min': 0.05,'max': 1.0,
            'rect': pygame.Rect(x_start, y_start + y_spacing * 3, 300, 10),'description': 'Division probability'},

            {'name': 'differentiation_rate','label': 'Differentiation','min': 0.01,'max': 0.4,
            'rect': pygame.Rect(x_start, y_start + y_spacing * 4, 300, 10),'description': 'Cell specialization rate'},

            {'name': 'morphogen_diffusion','label': 'Morphogen Diffusion','min': 0.05,'max': 2.0,
            'rect': pygame.Rect(x_start, y_start + y_spacing * 5, 300, 10),'description': 'Pattern formation gradient'},
        ]

    def week(self):
        return self.engine.week()

    def draw_simulation(self):
        # background for sim area
        pygame.draw.rect(self.screen, (12,12,28), (SIM_ORIGIN_X, SIM_ORIGIN_Y, SIM_AREA_W, SIM_AREA_H))
        # draw guides
        cx = SIM_ORIGIN_X + SIM_AREA_W//2 + int(self.cam_x)
        cy = SIM_ORIGIN_Y + SIM_AREA_H//2 + int(self.cam_y)
        pygame.draw.circle(self.screen, (130,60,190), (cx, cy), 280, 3)

        # draw cells
        for c in self.engine.cells:
            sx = SIM_ORIGIN_X + int(c.x + self.cam_x)
            sy = SIM_ORIGIN_Y + int(c.y + self.cam_y)
            ct = CellType(c.type)
            color = COLORS.get(ct, (200,200,200))
            # size by type
            radius = 2
            if ct in (CellType.HEART, CellType.LIVER, CellType.LUNG):
                radius = 4
            elif ct in (CellType.HEAD, CellType.BODY):
                radius = 3
            pygame.draw.circle(self.screen, color, (sx, sy), radius)
            # small outline
            pygame.draw.circle(self.screen, (20,20,20), (sx, sy), radius, 1)

    def draw_ui(self):
        # gradient background
        for i in range(WINDOW_HEIGHT):
            c = int(18 + 12 * math.sin(i / WINDOW_HEIGHT * math.pi))
            pygame.draw.line(self.screen, (c, c, c+10), (0,i), (WINDOW_WIDTH, i))

        title = self.title_font.render("Fetal Development — Off-lattice", True, (255,255,255))
        subtitle = self.small_font.render("Agent-based conceptual model (not medical).", True, (200,200,255))
        self.screen.blit(title, (WINDOW_WIDTH//2 - title.get_width()//2, 18))
        self.screen.blit(subtitle, (WINDOW_WIDTH//2 - subtitle.get_width()//2, 54))

        self.draw_simulation()
        self.draw_buttons()
        self.draw_info_panel()
        self.draw_legend()
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

        # Mode toggle (Off-lattice / CPM placeholder)
        pygame.draw.rect(self.screen, (80,80,140), (570, by, 180, 40), border_radius=10)
        mm = self.font.render("Mode: Off-lattice", True, (255,255,255))
        self.screen.blit(mm, (580, by+10))

    def draw_info_panel(self):
        px, py = 650, 100
        pygame.draw.rect(self.screen, (60,40,100), (px, py, 520, 200), border_radius=10)
        pygame.draw.rect(self.screen, (140,70,200), (px, py, 520, 200), 2, border_radius=10)

        stage_title = self.font.render("Stage", True, (255,255,255))
        week_text = self.font.render(f"Week: {self.week()}", True, (255,220,180))
        stage_name = self.small_font.render(STAGE_NAMES.get(self.engine._anatomical_template(self.week()) and self._stage_from_week(), "—"), True, (220,200,255))
        iteration_text = self.small_font.render(f"Day (iter): {self.engine.iteration}", True, (200,160,220))
        speed_text = self.small_font.render(f"Speed: {self.params['speed']:.1f}x", True, (200, 255, 200))
        cpu_text = self.small_font.render(f"CPU: {self.cpu_usage:.1f}%", True, (255, 200, 200))

        self.screen.blit(stage_title, (px + 20, py + 12))
        self.screen.blit(stage_name, (px + 20, py + 40))
        self.screen.blit(week_text, (px + 240, py + 12))
        self.screen.blit(iteration_text, (px + 240, py + 42))
        self.screen.blit(speed_text, (px + 240, py + 72))
        self.screen.blit(cpu_text, (px + 20, py + 135))

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
        y0 = py + 100
        for i, ln in enumerate(lines[:2]):
            t = self.small_font.render(ln, True, (220,200,230))
            self.screen.blit(t, (px + 20, y0 + i*18))

    def _stage_from_week(self):
        w = self.week()
        if w <= 2:
            return 1
        elif w <= 8:
            return 2
        elif w <= 12:
            return 3
        elif w <= 24:
            return 4
        else:
            return 5

    def draw_legend(self):
        lx, ly = 650, 500
        pygame.draw.rect(self.screen, (40,40,60), (lx, ly, 520, 200), border_radius=10)
        pygame.draw.rect(self.screen, (140,70,200), (lx, ly, 520, 200), 2, border_radius=10)
        title = self.font.render("Cell Types", True, (255,255,255))
        self.screen.blit(title, (lx+20, ly+8))

        org_items = [("Head", CellType.HEAD), ("Brain", CellType.BRAIN), ("Body", CellType.BODY),
                     ("Arm", CellType.ARM), ("Leg", CellType.LEG), ("Heart", CellType.HEART),
                     ("Liver", CellType.LIVER), ("Stomach", CellType.STOMACH), ("Intestines", CellType.INTESTINE),
                     ("Kidney", CellType.KIDNEY), ("Lungs", CellType.LUNG), ("Eyes", CellType.EYE),
                     ("Bladder", CellType.BLADDER), ("Placenta", CellType.PLACENTA), ("Umbilical", CellType.UMBILICAL)]
        for i, (name, ctype) in enumerate(org_items):
            x = lx + (i % 3) * 170
            y = ly + 40 + (i // 3) * 28
            pygame.draw.rect(self.screen, COLORS[ctype], (x, y, 20, 20), border_radius=4)
            pygame.draw.rect(self.screen, (200,200,220), (x, y, 20, 20), 1, border_radius=4)
            txt = self.small_font.render(name, True, (220,200,255))
            self.screen.blit(txt, (x + 28, y + 1))
                    
    def draw_settings(self):
        panel_rect = pygame.Rect(640, 270, 540, 340)
        pygame.draw.rect(self.screen, (40, 40, 60), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (140, 70, 200), panel_rect, 2, border_radius=10)
        clip_rect = pygame.Rect(650, 280, 520, 320)
        self.screen.set_clip(clip_rect)
        for slider in self.sliders:
            y_offset = slider['rect'].y + self.settings_scroll
            label = self.font.render(slider['label'], True, (255,255,255))
            self.screen.blit(label, (slider['rect'].x, y_offset - 24))
            value = float(self.params[slider['name']])
            vtxt = self.small_font.render(f"{value:.3f}", True, (200,150,255))
            self.screen.blit(vtxt, (slider['rect'].x + slider['rect'].width + 10, y_offset - 6))
            slider_rect = pygame.Rect(slider['rect'].x, y_offset, slider['rect'].width, slider['rect'].height)
            pygame.draw.rect(self.screen, (70,70,90), slider_rect, border_radius=5)
            normalized = (value - slider['min']) / max(1e-6, (slider['max'] - slider['min']))
            hx = int(slider['rect'].x + normalized * slider['rect'].width)
            pygame.draw.circle(self.screen, (120,60,180), (hx, y_offset + 5), 11)
            pygame.draw.circle(self.screen, (200,150,255), (hx, y_offset + 5), 11, 2)
            desc = self.small_font.render(slider['description'], True, (180,160,220))
            self.screen.blit(desc, (slider['rect'].x, y_offset + 16))
        self.screen.set_clip(None)

    def handle_click(self, pos):
        x, y = pos
        by = 740
        if by <= y <= by+40:
            if 150 <= x <= 270:
                self.is_running = not self.is_running
            elif 290 <= x <= 410:
                self._reset_sim()
            elif 430 <= x <= 550:
                self.show_settings = not self.show_settings
            elif 570 <= x <= 750:
                # mode toggle (placeholder)
                self.mode = 0 if self.mode == 1 else 1
        if self.show_settings:
            panel_rect = pygame.Rect(640, 270, 540, 340)
            if panel_rect.collidepoint(pos):
                for s in self.sliders:
                    adjusted_rect = pygame.Rect(s['rect'].x, s['rect'].y + self.settings_scroll, s['rect'].width, s['rect'].height)
                    if adjusted_rect.collidepoint(pos):
                        self.active_slider = s
                        self.update_slider(pos)

    def update_slider(self, pos):
        if not self.active_slider: return
        s = self.active_slider
        x = max(s['rect'].x, min(pos[0], s['rect'].x + s['rect'].width))
        normalized = (x - s['rect'].x) / max(1, s['rect'].width)
        value = s['min'] + normalized * (s['max'] - s['min'])
        self.params[s['name']] = value

    def _reset_sim(self):
        self.engine = OffLatticeEngine(area_w=SIM_AREA_W, area_h=SIM_AREA_H, init_cells=600)
        self.params = self.engine.params
        self.is_running = False

    def run(self):
        running = True
        last_time = time.time()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                    elif event.button in (2,3):
                        mx, my = event.pos
                        grid_x0, grid_y0 = SIM_ORIGIN_X + int(self.cam_x), SIM_ORIGIN_Y + int(self.cam_y)
                        grid_x1, grid_y1 = grid_x0 + SIM_AREA_W, grid_y0 + SIM_AREA_H
                        if grid_x0 <= mx <= grid_x1 and grid_y0 <= my <= grid_y1:
                            self.dragging = True
                            self.drag_start = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (2,3):
                        self.dragging = False
                    elif event.button == 1:
                        self.active_slider = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.drag_start[0]
                        dy = event.pos[1] - self.drag_start[1]
                        self.cam_x += dx
                        self.cam_y += dy
                        self.drag_start = event.pos
                    elif self.active_slider:
                        self.update_slider(event.pos)
                elif event.type == pygame.MOUSEWHEEL and self.show_settings:
                    panel_rect = pygame.Rect(640, 270, 540, 340)
                    if panel_rect.collidepoint(pygame.mouse.get_pos()):
                        self.settings_scroll += event.y * 20
                        self.settings_scroll = max(-200, min(0, self.settings_scroll))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.cam_x += 20
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.cam_x -= 20
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.cam_y += 20
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.cam_y -= 20

            # update performance occasionally
            current_time = time.time()
            if current_time - self.last_perf_update > 0.5:
                try:
                    self.cpu_usage = psutil.cpu_percent()
                except Exception:
                    self.cpu_usage = 0.0
                self.last_perf_update = current_time

            if self.is_running:
                # step engine multiple times depending on speed (allow fractional via dt)
                speed = max(0.01, float(self.params['speed']))
                steps = int(max(1, math.floor(speed)))
                frac = speed - math.floor(speed)
                for _ in range(steps):
                    self.engine.step(dt=1.0)
                if frac > 1e-6:
                    self.engine.step(dt=frac)

            # draw everything
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    print("Starting Off-Lattice UI...")
    try:
        ui = FetalDevelopmentUI()
        print("UI created successfully")
        ui.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
